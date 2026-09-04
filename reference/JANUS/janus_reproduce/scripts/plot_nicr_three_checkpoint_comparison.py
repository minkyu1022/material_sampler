#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
p=Path('outputs/nicr_round24_diagnostics/three_checkpoint_raw_summary.json'); d=json.loads(p.read_text())
out=p.parent/'three_checkpoint_raw_comparison.png'
fig,axs=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
groups=['old','r24','new120']; colors=['.45','#2ca02c','#9467bd']
for row,ph in enumerate(['fcc','bcc']):
 ns=[27,54,81] if ph=='fcc' else [32,64,96]
 for col,(field,title) in enumerate([('energy_per_atom','Energy (eV/atom)'),('volume_per_atom','Volume/atom'),('rms_u','RMS u'),('logw','std(log W)')]):
  ax=axs[row,col]; x=np.arange(3);w=.25
  for j,g in enumerate(groups):
   vals=[d[g][f'{ph}_n{n}'][field]['std' if field=='logw' else 'mean'] for n in ns]
   ax.bar(x+(j-1)*w,vals,w,label=g,color=colors[j])
  ax.set_xticks(x,['.25','.50','.75']);ax.set_title(f'{ph.upper()} {title}');ax.set_xlabel('xCr')
  if row==0 and col==0: ax.legend()
fig.suptitle('1200 K, unweighted terminal samples (256/condition)')
fig.savefig(out,dpi=180)
print(out)
