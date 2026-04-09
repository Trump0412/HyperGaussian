# Upstream Snapshot

- Source: `https://github.com/hustvl/4DGaussians`
- Imported commit: `843d5ac636c37e4b611242287754f3d4ed150144`
- Imported as vendored source under `external/4DGaussians/`
- Nested `.git` metadata was removed so the workspace stays a single top-level repository

## Local Integration Changes

- Added chronometric warp args to `external/4DGaussians/arguments/__init__.py`
- Added temporal warp hook in `external/4DGaussians/gaussian_renderer/__init__.py`
- Added separate temporal warp optimizer / save / load flow in `external/4DGaussians/train.py`
- Added temporal warp loading in `external/4DGaussians/render.py`

