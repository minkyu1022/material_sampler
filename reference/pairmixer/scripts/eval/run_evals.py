"""
DATASET=test

DATASET=casp15
NAME=boltz
python scripts/eval/run_evals.py \
    logs/boltz_results_final/ablations/${DATASET}/${NAME}/boltz_results_queries/predictions \
    logs/boltz_results_final/targets/${DATASET} \
    logs/boltz_results_final/ablation_results_with_local_lddt/${DATASET}/${NAME} \
    --testset ${DATASET}

"""
import argparse
import os
import subprocess
from pathlib import Path

from tqdm import tqdm
import pymp

NODE_ID = int(os.environ.get("SLURM_PROCID", 0))
NUM_NODES = int(os.environ.get("SLURM_NNODES", 1))

OST_COMPARE_STRUCTURE = r"""
#!/bin/bash

ost compare-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--min-pep-length 4 \
--min-nuc-length 4 \
-o {output_path} \
--lddt --bb-lddt --qs-score --dockq \
--local-lddt \
--ics --ips --rigid-scores --patch-scores --tm-score
"""


OST_COMPARE_LIGAND = r"""
#!/bin/bash

ost compare-ligand-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--lddt-pli --rmsd \
--substructure-match \
-o {output_path}
"""


def evaluate_structure(
    name: str,
    pred: Path,
    reference: Path,
    outdir: str,
    executable: str = "/bin/bash",
) -> None:
    """Evaluate the structure."""
    # Evaluate polymer metrics
    out_path = Path(outdir) / f"{name}.json"

    if out_path.exists():
        print(  # noqa: T201
            f"Skipping recomputation of {name} as protein json file already exists"
        )
    else:
        subprocess.run(
            OST_COMPARE_STRUCTURE.format(
                model_file=str(pred),
                reference_file=str(reference),
                output_path=str(out_path),
            ),
            shell=True,  # noqa: S602
            check=False,
            executable=executable,
            capture_output=True,
        )

    # Evaluate ligand metrics
    out_path = Path(outdir) / f"{name}_ligand.json"
    if out_path.exists():
        print(f"Skipping recomputation of {name} as ligand json file already exists")  # noqa: T201
    else:
        subprocess.run(
            OST_COMPARE_LIGAND.format(
                model_file=str(pred),
                reference_file=str(reference),
                output_path=str(out_path),
            ),
            shell=True,  # noqa: S602
            check=False,
            executable=executable,
            capture_output=True,
        )


def main(args):
    # Aggregate the predictions and references
    files = sorted(list(args.data.iterdir()))
    if NUM_NODES > 1:
        files = files[NODE_ID::NUM_NODES]
        print(f"Running on node {NODE_ID}/{NUM_NODES} with {len(files)} examples {files[:10]=}")
    names = {f.stem.lower(): f for f in files}

    # Create the output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    evals_to_run = []
    for name, folder in list(names.items()):
        for model_id in range(5):
            # Split the input data
            if args.format == "af3":
                pred_path = folder / f"seed-1_sample-{model_id}" / "model.cif"
            elif args.format == "chai":
                pred_path = folder / f"pred.model_idx_{model_id}.cif"
            elif args.format == "boltz":
                name_file = (
                    f"{name[0].upper()}{name[1:]}"
                    if 'casp' in args.testset
                    else name.lower()
                )
                pred_path = folder / f"{name_file}_model_{model_id}.cif"

            if 'casp' in args.testset:
                ref_path = args.pdb / f"{name[0].upper()}{name[1:]}.cif"
            elif args.testset == "test":
                ref_path = args.pdb / f"{name.lower()}.cif.gz"

            full_name = f"{name}_model_{model_id}"
            evals_to_run.append((full_name, pred_path, ref_path, args.outdir, args.executable))

    if args.num_threads > 1:
        print(f"Starting {len(evals_to_run)} evals in parallel with {args.num_threads} threads")
        with pymp.Parallel(args.num_threads) as p:
            for i in p.xrange(len(evals_to_run)):
                evaluate_structure(*evals_to_run[i])
    else:
        for i in tqdm(range(len(evals_to_run))):
            evaluate_structure(*evals_to_run[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--format", type=str, default="boltz")
    parser.add_argument("--testset", type=str, default="casp")
    parser.add_argument("--executable", type=str, default="/bin/bash")
    parser.add_argument("--num_threads", type=int, default=144)
    args = parser.parse_args()
    main(args)