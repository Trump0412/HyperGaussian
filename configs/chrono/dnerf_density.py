_base_ = '../../external/4DGaussians/arguments/dnerf/dnerf_default.py'

ModelHiddenParams = dict(
    warp_enabled=True,
    temporal_warp_type='density',
    warp_hidden_dim=32,
    warp_num_layers=2,
    warp_num_bins=128,
    warp_mono_weight=0.05,
    warp_smooth_weight=0.01,
    warp_budget_weight=0.01,
    warp_sample_count=128,
)

