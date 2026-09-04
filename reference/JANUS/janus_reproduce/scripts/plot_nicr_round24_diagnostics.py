#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
root=Path('outputs/nicr_round24_validation'); out=Path('outputs/nicr_round24_diagnostics');out.mkdir(parents=True,exist_ok=True)
names=['target','negative_prior','negative_discrete','u_path','v_path']
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
summary={}
for row,ph in enumerate(['fcc','bcc']):
 N=108 if ph=='fcc' else 128; ns=[round(N*x) for x in (.25,.5,.75)]
 data=[json.loads((root/ph/f'T1200_n{n}.json').read_text()) for n in ns]
 std=np.array([[d['components'][k]['std'] for k in names] for d in data])
 axes[row,0].bar(np.arange(3)-.15,std[:,3],.3,label='u path');axes[row,0].bar(np.arange(3)+.15,std[:,4],.3,label='v path')
 axes[row,0].set_xticks(range(3),['.25','.50','.75']);axes[row,0].set_ylabel('component std');axes[row,0].set_title(f'{ph.upper()} path components');axes[row,0].legend()
 axes[row,1].plot([.25,.5,.75],[d['std_log_weight'] for d in data],'o-',label='std(logW)')
 axes[row,1].set_title(f'{ph.upper()} total weight');axes[row,1].set_xlabel('xCr');axes[row,1].set_ylabel('std(logW)')
 corr=[]
 for d in data:
  cov=np.array(d['covariance']); total_var=cov.sum(); corr.append(cov.sum(1)/np.sqrt(np.diag(cov)*total_var))
 im=axes[row,2].imshow(np.array(corr).T,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
 axes[row,2].set_xticks(range(3),['.25','.50','.75']);axes[row,2].set_yticks(range(5),names);axes[row,2].set_title(f'{ph.upper()} corr(component,total)');fig.colorbar(im,ax=axes[row,2])
 summary[ph]=[{'x_cr':x,'ess':d['ess'],'std_log_weight':d['std_log_weight'],'component_std':{k:d['components'][k]['std'] for k in names},'component_total_correlation':dict(zip(names,c))} for x,d,c in zip((.25,.5,.75),data,corr)]
fig.savefig(out/'round24_T1200_logw_components.png',dpi=180)
(out/'round24_T1200_logw_components.json').write_text(json.dumps(summary,indent=2))
print(out/'round24_T1200_logw_components.png')
