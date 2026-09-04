
proximal cross entropy (PCE) method for sampling and fine-tuning diffusion models

---

**NOTE** This is an renewed code of PCE for sampling problems.

1) Install dependencies
```
conda env create -f environment.yml
conda activate pce-dm
```

2) training
```
CUDA_VISIBLE_DEVICES=1 python train.py experiment=[experiment_name] eta=[eta] iws=[iws]
```

- mw54
'''
train.py experiment=mw54 ema_decay=0.999 score_matcher.buffer_size=10000 optim.lr=1e-4 epsilon=1 zero_last_step_noise=true sde@ref_sde=vp source=gauss annealed=true model@controller=clipped param_type=adjoint loss_type=control num_epochs_per_stage=100 iws=true exp=mw54_final
'''

'''
train.py experiment=mw54 ema_decay=0.999 score_matcher.buffer_size=10000 optim.lr=1e-4 epsilon=1 zero_last_step_noise=false sde@ref_sde=brownian_motion source=delta annealed=false model@controller=clipped param_type=adjoint loss_type=control num_epochs_per_stage=100 iws=true exp=mw54_final_bm
'''

'''
train.py experiment=mw54 ema_decay=0.999 score_matcher.buffer_size=10000 optim.lr=1e-4 epsilon=1 zero_last_step_noise=true sde@ref_sde=vp source=gauss annealed=true model@controller=clipped param_type=adjoint loss_type=control num_epochs_per_stage=100 iws=true exp=mw54_final seed=2
'''

- student_t
'''
python train.py experiment=mos ema_decay=0.999 score_matcher.buffer_size=100000 optim.lr=1e-4 epsilon=10 zero_last_step_noise=true sde@ref_sde=vp source=gauss annealed=true model@controller=clipped param_type=control loss_type=control num_epochs_per_stage=500 iws=true
'''

- dw4
'''
python train.py experiment=dw4 max_grad_E_norm=100 epsilon=1 num_epochs_per_stage=1000 epsilon=0.1
'''

- lj13
train.py experiment=lj13 max_grad_E_norm=100 epsilon=1 num_epochs_per_stage=200


## Note on GMM40 Sinkhorn evaluation

The GMM40 Sinkhorn-divergence values reported in *Proximal Diffusion Neural Sampler* (arXiv:2510.03824, Table 1) were understated due to an evaluation issue in the Sinkhorn computation.

This mainly came from two implementation details:  
1. the default `geomloss` setting did not fully converge for the GMM40 setup, and  
2. `geomloss` and the SCLD reference use slightly different cost conventions for `p=2`.

We therefore updated the evaluator in `src/eval/distribution_distances.py` to use:
```python
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=1e-3,
                            truncate=None, scaling=0.99)
sinkhorn = 2.0 * sinkhorn_loss(true, pred).detach().cpu().numpy()