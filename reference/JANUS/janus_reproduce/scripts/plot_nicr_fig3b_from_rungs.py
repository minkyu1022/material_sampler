#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from janus_reproduce.thermodynamics import KB_EV

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('outputs/nicr_full_rung_eval'))
parser.add_argument('--output',type=Path,default=Path('outputs/nicr_full_rung_eval_fig3b'))
args=parser.parse_args()
ROOT=args.root
OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
Ts=[600,750,900,1050,1200,1350,1500]

def load(ph,T,N):
 rows=[json.loads((ROOT/ph/f'T{T}_n{n}.json').read_text()) for n in range(N+1)]
 x=np.arange(N+1)/N; g=-KB_EV*T*np.array([r['log_xi'] for r in rows])/N
 return x,g,rows

def aligned(T):
 xf,gf,rf=load('fcc',T,108); xb,gb,rb=load('bcc',T,128)
 anchor_ni=gf[0]; anchor_cr=gb[-1]
 gf=(gf-(1-xf)*anchor_ni-xf*anchor_cr)*1000
 gb=(gb-(1-xb)*anchor_ni-xb*anchor_cr)*1000
 return xf,gf,xb,gb,rf,rb

def coexist(xf,gf,xb,gb):
 pts=sorted([(x,g,'fcc') for x,g in zip(xf,gf)]+[(x,g,'bcc') for x,g in zip(xb,gb)])
 hull=[]
 for p in pts:
  while len(hull)>1:
   a,b=hull[-2:]
   if (b[1]-a[1])/(b[0]-a[0]+1e-15) < (p[1]-b[1])/(p[0]-b[0]+1e-15): break
   hull.pop()
  hull.append(p)
 cross=[(a,b) for a,b in zip(hull,hull[1:]) if a[2]!=b[2] and a[0]<b[0]]
 return max(cross,key=lambda z:z[1][0]-z[0][0]) if cross else None

xf,gf,xb,gb,_,_=aligned(1200)
fig=plt.figure(figsize=(7.2,7.0)); gs=fig.add_gridspec(2,2,width_ratios=[1,2.7],height_ratios=[1,1],hspace=.08,wspace=.04)
axes=[[fig.add_subplot(gs[r,c]) for c in range(2)] for r in range(2)]
for ax in axes[0]:
 ax.plot(xf,gf,color='#ed5a1f',lw=3,label='JANUS (fcc)')
 ax.plot(xb,gb,color='#1479bd',lw=3,label='JANUS (bcc)')
 ax.axhline(0,color='.75',lw=.7)
 ax.set_ylim(-30,70)
axes[0][0].set_xlim(0,.2); axes[0][1].set_xlim(.45,1)
axes[0][0].set_ylabel(r'$G_{mix}$ (meV/atom)',fontsize=15)
axes[0][0].text(.02,58,'T = 1200 K',fontsize=18)
axes[0][1].legend(frameon=False,loc='upper right')
left=[];right=[]
for T in Ts:
 a=aligned(T); pair=coexist(a[0],a[1],a[2],a[3])
 if pair: left.append((pair[0][0],T)); right.append((pair[1][0],T))
if left:
 axes[1][0].plot(*zip(*left),'-o',color='#ed5a1f',lw=3,ms=7)
 axes[1][1].plot(*zip(*left),'-o',color='#ed5a1f',lw=3,ms=7)
 axes[1][0].plot(*zip(*right),'-s',color='#1479bd',lw=3,ms=7)
 axes[1][1].plot(*zip(*right),'-s',color='#1479bd',lw=3,ms=7)
for ax in axes[1]: ax.set_ylim(575,1225)
axes[1][0].set_xlim(0,.2); axes[1][1].set_xlim(.45,1)
axes[1][0].set_ylabel('T (K)',fontsize=15); fig.supxlabel(r'$\langle x_{Cr}\rangle$',fontsize=15,y=.025)
for r in range(2):
 axes[r][0].spines['right'].set_visible(False); axes[r][1].spines['left'].set_visible(False)
 axes[r][1].tick_params(labelleft=False,left=False)
 axes[r][0].plot((1-.015,1+.015),(-.015,+.015),transform=axes[r][0].transAxes,color='.5',clip_on=False)
 axes[r][1].plot((-.015,+.015),(-.015,+.015),transform=axes[r][1].transAxes,color='.5',clip_on=False)
for ax in sum(axes,[]): ax.tick_params(labelsize=12); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Existing checkpoints — path-weight estimate (low ESS)',fontsize=12,color='crimson')
fig.savefig(OUT/'fig3b_existing_models.png',dpi=220,bbox_inches='tight')
summary={'temperatures':Ts,'fcc_boundary':left,'bcc_boundary':right,'warning':'Diagnostic: condition-wise ESS is near 1; do not treat as converged free energy.'}
(OUT/'fig3b_existing_models.json').write_text(json.dumps(summary,indent=2))
print(OUT/'fig3b_existing_models.png')
