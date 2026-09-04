"""Exact fixed-cardinality Bernoulli distributions for masked alloy species."""

from __future__ import annotations

import torch
from torch import Tensor


def _tables(logits: Tensor, quota: int) -> tuple[list[Tensor], list[Tensor]]:
    if logits.ndim != 1 or not 0 <= quota <= len(logits):
        raise ValueError("quota must be between zero and the number of logits")
    negative_infinity = logits.new_tensor(float("-inf"))
    prefix = [torch.cat((logits.new_zeros(1), negative_infinity.repeat(quota)))]
    for count, logit in enumerate(logits, 1):
        previous = prefix[-1]
        values = [previous[0]]
        for q in range(1, quota + 1):
            values.append(
                torch.logaddexp(previous[q], previous[q - 1] + logit)
                if q <= count
                else negative_infinity
            )
        prefix.append(torch.stack(values))
    suffix = [torch.empty(0, device=logits.device)] * (len(logits) + 1)
    suffix[-1] = prefix[0]
    for index in range(len(logits) - 1, -1, -1):
        previous = suffix[index + 1]
        values = [previous[0]]
        available = len(logits) - index
        for q in range(1, quota + 1):
            values.append(
                torch.logaddexp(previous[q], previous[q - 1] + logits[index])
                if q <= available
                else negative_infinity
            )
        suffix[index] = torch.stack(values)
    return prefix, suffix


def log_partition(logits: Tensor, quota: int) -> Tensor:
    """Log partition for choosing exactly ``quota`` items."""
    work = logits.double()
    if not 0 <= quota <= len(work):
        raise ValueError("quota must be between zero and the number of logits")
    values = work.new_zeros(1)
    for logit in work:
        if len(values) <= quota:
            values = torch.cat(
                (
                    values[:1],
                    torch.logaddexp(values[1:], values[:-1] + logit),
                    (values[-1] + logit).reshape(1),
                )
            )
        else:
            values = torch.cat(
                (values[:1], torch.logaddexp(values[1:], values[:-1] + logit))
            )
    return values[quota]


def constrained_marginals(logits: Tensor, quota: int) -> Tensor:
    """Exact inclusion marginals from prefix/suffix dynamic programming."""
    work = logits.double()
    if quota == 0:
        return torch.zeros_like(work)
    if quota == len(work):
        return torch.ones_like(work)
    prefix, suffix = _tables(work, quota)
    log_z = prefix[-1][quota]
    output = []
    for index, logit in enumerate(work):
        terms = [prefix[index][q] + suffix[index + 1][quota - 1 - q] for q in range(quota)]
        output.append((logit + torch.logsumexp(torch.stack(terms), 0) - log_z).exp())
    return torch.stack(output)


def constrained_nll(
    logits: Tensor,
    terminal_species: Tensor,
    masked: Tensor,
    target_cr: Tensor,
) -> Tensor:
    """Batch exact-cardinality NLL for terminal Cr subsets on masked sites."""
    losses = []
    for batch in range(len(logits)):
        sites = torch.where(masked[batch])[0]
        revealed_cr = terminal_species[batch, ~masked[batch]].eq(1).sum().item()
        quota = int(target_cr[batch].item()) - revealed_cr
        if not 0 <= quota <= len(sites):
            raise ValueError("masked state has an invalid remaining Cr quota")
        cr_logit = logits[batch, sites, 1] - logits[batch, sites, 0]
        selected = terminal_species[batch, sites].eq(1)
        if int(selected.sum()) != quota:
            raise ValueError("terminal species does not match the requested composition")
        losses.append(log_partition(cr_logit, quota) - cr_logit[selected].double().sum())
    return torch.stack(losses).mean()


def constrained_reveal_step(
    species: Tensor,
    logits: Tensor,
    target_cr: Tensor,
    reveal_probability: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Batched exact conditional reveals; log probability conditions on reveal sites."""
    if species.ndim != 2 or logits.shape != (*species.shape, 2):
        raise ValueError("require species [batch,sites] and binary logits [batch,sites,2]")
    if not 0 <= reveal_probability <= 1:
        raise ValueError("reveal_probability must be in [0,1]")
    target_cr = torch.as_tensor(target_cr, device=species.device).long().reshape(-1)
    batch, sites = species.shape
    output = species.clone()
    masked = species.eq(2)
    quota = target_cr - species.eq(1).sum(1)
    masked_count = masked.sum(1)
    if torch.any((quota < 0) | (quota > masked_count)):
        raise ValueError("masked state has an invalid remaining Cr quota")

    random = torch.rand((batch, sites), device=species.device, generator=generator)
    reveal = masked & (random < reveal_probability)
    # Selected sites first in random order, then unrevealed masked sites. Already
    # revealed sites sort last and do not enter the partition recurrence.
    keys = torch.rand((batch, sites), device=species.device, generator=generator)
    keys = keys + (~reveal) + 2 * (~masked)
    order = keys.argsort(1)
    active = masked.gather(1, order)
    cr_logit = (logits[..., 1] - logits[..., 0]).double().gather(1, order)

    negative_infinity = cr_logit.new_tensor(float("-inf"))
    suffix = cr_logit.new_full((batch, sites + 1, sites + 1), negative_infinity)
    suffix[:, sites, 0] = 0
    for position in range(sites - 1, -1, -1):
        previous = suffix[:, position + 1]
        included = torch.cat(
            (
                negative_infinity.expand(batch, 1),
                previous[:, :-1] + cr_logit[:, position, None],
            ),
            1,
        )
        updated = torch.logaddexp(previous, included)
        suffix[:, position] = torch.where(active[:, position, None], updated, previous)

    log_probability = torch.zeros(batch, dtype=torch.float64, device=species.device)
    reveal_count = reveal.sum(1)
    batch_index = torch.arange(batch, device=species.device)
    for position in range(int(reveal_count.max().item())):
        selected = position < reveal_count
        current_quota = quota.clamp_min(1)
        denominator = suffix[batch_index, position, quota]
        numerator = (
            cr_logit[:, position]
            + suffix[batch_index, position + 1, current_quota - 1]
        )
        probability_cr = torch.where(quota > 0, (numerator - denominator).exp(), 0.0)
        probability_cr = probability_cr.clamp(0, 1)
        choice = selected & (
            torch.rand(batch, device=species.device, generator=generator) < probability_cr
        )
        site = order[:, position]
        output[batch_index[selected], site[selected]] = choice[selected].long()
        probability = torch.where(choice, probability_cr, 1 - probability_cr)
        log_probability += torch.where(
            selected,
            probability.clamp_min(torch.finfo(torch.float64).tiny).log(),
            0.0,
        )
        quota -= choice.long()
    return output, log_probability
