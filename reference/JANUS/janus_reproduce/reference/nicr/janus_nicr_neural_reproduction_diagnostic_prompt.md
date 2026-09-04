# JANUS Ni-Cr Neural Reproduction Diagnostic Prompt

We need to diagnose the JANUS Ni-Cr neural-model reproduction before spending more compute.

## Current status

1. The reference MCMC composition-ladder reproduction appears healthy:
   - 30k sweeps completed for both FCC and BCC.
   - Full composition range is populated.
   - Coordinate / volume acceptance is roughly 0.30–0.38.
   - Interior pair-swap acceptance is roughly 0.34–0.45.
   - Random-start and slab-start walkers show reasonably small disagreement.
   - Energy and volume profiles are smooth.
   Therefore, do **not** treat the reference-MCMC side and the neural-model side as one failure.

2. The existing JANUS neural checkpoints currently fail badly under path-weight diagnostics:
   - Direct path-weight G_mix is extremely jagged.
   - Median within-rung ESS is approximately 1.11 for FCC and 1.00 for BCC.
   - This is essentially complete importance-weight collapse.
   - These curves must **not** be presented as valid Fig. 3b reproduction.

3. Important methodological point:
   The paper’s reported Cu-Ag / Ni-Cr composition-ladder results do **not** rely on independently estimating absolute Z_n from direct path-weight averages rung by rung.

   The reported ladder result uses neighboring-rung substitution works and importance-weighted BAR to estimate

       Delta_n = beta [G(n+1) - G(n)],

   followed by accumulation along the composition ladder.

   However, ESS ≈ 1 is still a serious problem because the weighted BAR expectation will then effectively be dominated by one trajectory.

Your task is to determine whether the failure comes from:

A. the trained sampler itself  
B. the forward/backward path-weight implementation  
C. a mismatch with the paper’s fixed-composition JANUS training/inference protocol  
D. or some combination of these

Do not hide failed results. Report every diagnostic, including negative results.

---

## STEP 1. Verify the exact Ni-Cr neural training protocol

Before running another expensive training job, audit the current implementation against the paper.

For the Ni-Cr Fig. 3b route, verify explicitly that:

1. Training is fixed-composition / canonical-rung training, not ordinary semi-grand generation followed by rare-composition reweighting.
2. During rollout for rung n, N_Cr = n is enforced exactly.
3. The conditioning feature uses the rung composition c0 = n / N.
4. One amortized model covers all composition rungs and the required temperature window for a given lattice.
5. FCC and BCC use separate lattice-specific models.
6. The continuous channels and discrete species channel are trained jointly as intended.
7. Replay-buffer behavior matches the intended JANUS training loop.
8. The physics-informed displacement and volume priors are using the intended phase-specific calibration.
9. The target EAM energy evaluation and the JANUS graph cutoff are not being conflated.
10. Inference uses the same discretization / integration convention expected by the training and path-weight equations.

Return a table:

| item | paper protocol | our implementation | match? | evidence/code location |
|---|---|---|---|---|

Do not merely say “looks correct.” Give exact config values and code locations.

---

## STEP 2. Diagnose the sampler BEFORE importance weighting

We need to know whether q_theta itself is poor before blaming the weights.

For representative compositions, at minimum:
- x_Cr = 0.25
- x_Cr = 0.50
- x_Cr = 0.75

for both FCC and BCC at 1200 K, compare raw unweighted JANUS terminal samples against the converged reference-MCMC rung samples.

Compare distributions of:
1. potential energy per atom
2. volume per atom
3. displacement magnitude statistics
4. at least one meaningful local chemical-order statistic / pair statistic
5. any other quantity already available from the MCMC QA pipeline

For each observable report:
- JANUS mean ± std
- MCMC mean ± std
- distribution overlap / histogram
- standardized mean discrepancy if useful

Produce plots with JANUS and MCMC overlaid.

Interpretation:
- If unweighted JANUS already strongly disagrees with MCMC, this is primarily a model-training / sampler problem.
- If unweighted JANUS is close to MCMC but path weights collapse, suspect the path-weight implementation or accumulated trajectory mismatch.

---

## STEP 3. Decompose log path weights

For every generated trajectory, save the individual contributions to log W instead of only the final total.

At minimum separate:
1. terminal target-density / energy contribution
2. prior-density contribution
3. discrete species-generation contribution
4. displacement-channel forward/backward Gaussian log-ratio
5. volume-channel forward/backward Gaussian log-ratio
6. any Jacobian or conditioning-related terms
7. total log W

For each representative rung and for all rungs if feasible, report:
- mean
- std
- min/max
- variance
- correlation with total log W

Also plot the distributions.

We need to identify which component is causing the enormous variance.

Explicitly check whether Var(log W_disc), Var(log W_u), Var(log W_v), or another term dominates.

Also verify:
- signs
- forward/backward orientation
- normalization constants that should cancel
- whether any term is accidentally accumulated once per integration step when it should only appear once

---

## STEP 4. Validate the path-weight implementation with controlled tests

Do not trust the current path-weight code until it passes controlled tests.

Construct simplified cases where the answer should be easy.

### 4.1 Near-identity / well-trained proposal
If possible evaluate trajectories where forward and backward dynamics are deliberately matched closely. log W should have much smaller variance.

### 4.2 Continuous-only test
Disable/fix species if possible and validate the displacement/volume Gaussian path-ratio machinery separately.

### 4.3 Discrete-only test
Freeze continuous channels and validate the species probability contribution separately.

### 4.4 Small toy system
If tractable, compare the path-weight estimate against an exact or high-quality MCMC quantity.

### 4.5 Numerical stability
Check that:
- all normalization uses log-sum-exp
- float32 vs float64 does not materially alter ESS or rung estimates

Do not proceed on the assumption that the equations are coded correctly just because no runtime error occurs.

---

## STEP 5. Compute the ACTUAL neural ladder estimator used for Fig. 3b

The current direct path-weight G_mix plot is only a diagnostic.

Implement / verify the neighboring-rung importance-weighted BAR estimator.

For edge n -> n+1:

1. From rung n terminal configurations, evaluate all eligible Ni->Cr substitution works.
2. From rung n+1 terminal configurations, evaluate all eligible Cr->Ni substitution works.
3. Use the normalized within-rung path weights Wbar_k^(n) and Wbar_l^(n+1) inside the weighted BAR equation.
4. Solve for Delta_n = beta [G(n+1) - G(n)].
5. Accumulate G(n) - G(0) = k_B T sum_{m < n} Delta_m.
6. Construct G_mix(x) by subtracting the pure-endpoint chord.

For every edge report:
- Delta_n
- ESS_n
- ESS_(n+1)
- forward one-sided estimate
- reverse one-sided estimate
- forward/reverse discrepancy
- BAR convergence/root status
- uncertainty/bootstrap error if available

Make an edge-level diagnostic plot versus composition.

Do **not** output only the final smooth curve. I want to see where the ladder becomes unreliable.

---

## STEP 6. Define explicit reliability criteria

Do not call a neural G_mix curve successful simply because BAR returns a finite number.

Mark each rung/edge as reliable or unreliable using diagnostics such as:
1. within-rung ESS
2. forward/reverse one-sided discrepancy
3. BAR overlap
4. bootstrap uncertainty
5. raw JANUS-vs-MCMC distribution mismatch

If ESS remains approximately 1, explicitly label the estimate unreliable even if BAR produces a value.

Do not smooth, spline, interpolate, or regularize failed edges in a way that hides the problem.

---

## STEP 7. Compare existing checkpoints vs the new common-setting models

Once the new models finish, evaluate old and new checkpoints with the **exact same**:
- sampling count
- random seeds if possible
- composition rungs
- temperature
- integration steps
- path-weight code
- BAR code
- plotting code
- axes

Produce a direct comparison of:
1. median / min / distribution of ESS
2. raw terminal-vs-MCMC mismatch
3. log W component variances
4. BAR edge diagnostics
5. final G_mix, only where statistically reliable

The goal is to determine whether the old checkpoints failed because of their model settings or whether the reproduction pipeline has a systematic implementation mismatch.

---

## STEP 8. Do NOT move to the FCC/BCC phase boundary too early

For Ni-Cr, first establish reliable relative G_mix curves independently for FCC and BCC.

Only after those curves are reliable should we add the pure-element FCC/BCC absolute free-energy offsets and perform the common-tangent construction for the phase boundary.

If the rung curves are unreliable, leave the phase-diagram panel empty and say why.

---

## Required deliverable

Send me one concise diagnostic report containing:

A. Protocol audit table  
B. Raw JANUS vs MCMC terminal-distribution plots  
C. log W component decomposition  
D. ESS vs composition  
E. Weighted-BAR edge diagnostics  
F. Neural G_mix vs reference MCMC G_mix at 1200 K  
G. Old-checkpoint vs new-model comparison  
H. A final root-cause assessment classified as:

1. training/model mismatch
2. path-weight bug
3. fixed-composition protocol mismatch
4. estimator/implementation bug
5. insufficient training/convergence
6. unresolved

For each conclusion, give concrete evidence.

Most importantly:
- Do not claim reproduction success until the weighted-BAR free-energy curve is statistically reliable and agrees with the reference within meaningful uncertainty.
- Do not hide low-ESS or failed results.
- Start **STEP 2 and STEP 3 immediately using the existing checkpoints while the new common-setting models are still running**.
- If the new models also show ESS ≈ 1, treat this as evidence for a systematic training/inference/path-weight implementation mismatch rather than a bad checkpoint or random seed.
