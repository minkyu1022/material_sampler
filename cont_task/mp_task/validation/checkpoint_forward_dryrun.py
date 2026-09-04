#!/usr/bin/env python3
import csv, json, time
from pathlib import Path
import torch
from pymatgen.core import Structure
from src.data.mp20_tokens import lattice_to_Y, VZ
from src.eval_crystalite_ckpt import _apply_ema_state_dict, _build_model_from_ckpt, _load_checkpoint
from src.models.lattice_repr import y1_to_lattice_latent
from src.models.type_encoding import build_type_encoding

root=Path(__file__).resolve().parents[3]
checkpoint=root/'cont_task/mp_task/pre_train/checkpoints/csp_mp20_best.pt'
with (root/'cont_task/mp_task/data/mp20/raw/val.csv').open(newline='') as f:
    row=next(csv.DictReader(f))
structure=Structure.from_str(row['cif'],fmt='cif').get_reduced_structure()
nmax=20
assert len(structure)<=nmax
a0=torch.zeros((1,nmax),dtype=torch.long)
frac=torch.zeros((1,nmax,3),dtype=torch.float32)
pad=torch.ones((1,nmax),dtype=torch.bool)
a0[0,:len(structure)]=torch.tensor([site.specie.Z for site in structure])
frac[0,:len(structure)]=torch.tensor(structure.frac_coords%1,dtype=torch.float32)
pad[0,:len(structure)]=False
y1=torch.tensor(lattice_to_Y(structure.lattice.abc,structure.lattice.angles),dtype=torch.float32)[None]
device=torch.device('cuda:0')
before=torch.cuda.memory_allocated(device)
started=time.time()
ckpt=_load_checkpoint(checkpoint)
model,cfg=_build_model_from_ckpt(ckpt=ckpt,device=device)
_apply_ema_state_dict(model,ckpt['ema_state_dict'])
load_seconds=time.time()-started
types=build_type_encoding(str(ckpt.get('type_encoding',cfg.get('type_encoding','atomic_number'))),vz=VZ).encode_from_A0(a0.to(device),pad.to(device))
lat=y1_to_lattice_latent(y1.to(device),str(cfg['lattice_repr']))
t=torch.tensor([0.5],device=device)
torch.cuda.reset_peak_memory_stats(device)
with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
    out=model(types,frac.to(device),lat,pad.to(device),t,lattice_bias_feats=lat,gem_sigma=1.0-t)
torch.cuda.synchronize()
report={
 'material_id':row['material_id'],'num_atoms':len(structure),'checkpoint_step':int(ckpt['step']),
 'parameters':sum(p.numel() for p in model.parameters()),'lattice_repr':cfg['lattice_repr'],
 'coordinate_repr':cfg.get('coordinate_repr','fractional'),'load_seconds':load_seconds,
 'output_shapes':{k:list(v.shape) for k,v in out.items()},
 'all_outputs_finite':all(torch.isfinite(v).all().item() for v in out.values()),
 'peak_forward_memory_bytes':torch.cuda.max_memory_allocated(device),
 'optimizer_created':False,'backward_called':False,
}
(root/'cont_task/mp_task/validation/checkpoint_forward_dryrun.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
