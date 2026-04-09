import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPSTREAM_ROOT = os.path.join(REPO_ROOT, "external", "4DGaussians")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, UPSTREAM_ROOT)


def check_dnerf(scene_path):
    required = ["transforms_train.json", "transforms_test.json"]
    missing = [name for name in required if not os.path.exists(os.path.join(scene_path, name))]
    if missing:
        raise FileNotFoundError(f"{scene_path}: missing {missing}")
    with open(os.path.join(scene_path, "transforms_train.json"), "r", encoding="utf-8") as handle:
        train_meta = json.load(handle)
    if not train_meta.get("frames"):
        raise ValueError(f"{scene_path}: transforms_train.json contains no frames")
    return {"frames": len(train_meta["frames"]), "type": "dnerf"}


def check_hypernerf(scene_path):
    required = ["dataset.json", "metadata.json", "scene.json", "points3D_downsample2.ply"]
    missing = [name for name in required if not os.path.exists(os.path.join(scene_path, name))]
    if missing:
        raise FileNotFoundError(f"{scene_path}: missing {missing}")
    rgb_dir = os.path.join(scene_path, "rgb", "2x")
    cam_dir = os.path.join(scene_path, "camera")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"{scene_path}: expected rgb/2x and camera/ directories")
    with open(os.path.join(scene_path, "dataset.json"), "r", encoding="utf-8") as handle:
        dataset_meta = json.load(handle)
    if not dataset_meta.get("ids"):
        raise ValueError(f"{scene_path}: dataset.json contains no ids")
    return {"frames": len(dataset_meta["ids"]), "type": "hypernerf"}


def try_scene_load(scene_path):
    from argparse import Namespace
    from arguments import ModelHiddenParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    parser_stub = argparse.ArgumentParser()
    hyper = ModelHiddenParams(parser_stub)
    hyper_args = hyper.extract(parser_stub.parse_args([]))
    model_args = Namespace(
        sh_degree=3,
        source_path=scene_path,
        model_path=os.path.join(REPO_ROOT, "tmp", "layout_check"),
        images="images",
        resolution=-1,
        white_background=True,
        data_device="cuda",
        eval=True,
        render_process=False,
        add_points=False,
        extension=".png",
        llffhold=8,
    )
    gaussians = GaussianModel(model_args.sh_degree, hyper_args)
    Scene(model_args, gaussians, load_iteration=None, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnerf-scene", default=os.path.join(REPO_ROOT, "data", "dnerf", "bouncingballs"))
    parser.add_argument("--hypernerf-scene", default=os.path.join(REPO_ROOT, "data", "hypernerf", "virg", "broom2"))
    parser.add_argument("--skip-scene-load", action="store_true")
    args = parser.parse_args()

    report = {}
    report["dnerf"] = check_dnerf(args.dnerf_scene)
    report["hypernerf"] = check_hypernerf(args.hypernerf_scene)

    if not args.skip_scene_load:
        try_scene_load(args.dnerf_scene)
        try_scene_load(args.hypernerf_scene)
        report["scene_load"] = "ok"

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

