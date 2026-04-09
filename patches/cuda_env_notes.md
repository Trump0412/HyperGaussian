# CUDA Environment Notes

## Observed On This Machine

- Host driver: CUDA 13.0 compatible driver
- Host toolkit: `/usr/local/cuda-12.1`
- Original baseline stack: `python=3.7`, `torch=1.13.1+cu117`
- Pragmatic working stack: `python=3.10`, `torch=2.1.2+cu121`

## Compatibility Issue

Compiling `depth-diff-gaussian-rasterization` against system CUDA 12.1 fails when paired with PyTorch 1.13.1, because that wheel was compiled for CUDA 11.7.

After switching to a dedicated `torch==2.1.2+cu121` environment and pointing `CUDA_HOME` at `/usr/local/cuda-12.1`, both `simple-knn` and `depth-diff-gaussian-rasterization` compile successfully on this machine.

## Current Mitigation

`scripts/setup_baseline_env.sh` now supports:

- `official`: upstream-like PyTorch 1.13.1 / CUDA 11.7 path
- `cuda121`: practical PyTorch 2.1.2 / CUDA 12.1 path
- fast failure when `${CUDA_HOME}/include/cuda_runtime.h` is missing
- toolkit override through `GS4D_CUDA_HOME`
- `open3d` removed from the required CUDA 12.1 path; point-cloud downsampling now uses a vendored `plyfile`-based voxel sampler

## Next Step If You Want A Fully Working Local Build

For the `official` profile, point `GS4D_CUDA_HOME` at a full CUDA 11.7 toolkit that contains:

- `bin/nvcc`
- `include/cuda_runtime.h`
- matching runtime libraries under `lib64/`
