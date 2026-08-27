import os
import sys
import time
import json

_UNSLOTH_PATCH_PATH_PARTS = ("pip_install_unsloth_flash_patch", "pip-install-unsloth-flash-patch")
_KNOWN_UNSLOTH_PATCH_PATH = "/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch"
_FILTERED_UNSLOTH_PATCH_PATH = "/kaggle/working/unsloth_patch_imports"
_EXCLUDED_UNSLOTH_PATCH_NAMES = ("flash_attn", "flash_attn_2_cuda")
_EXCLUDED_UNSLOTH_PATCH_PREFIXES = ("flash_attn-",)


def _is_unsloth_patch_path(path):
    return any(part in path for part in _UNSLOTH_PATCH_PATH_PARTS)


def _is_excluded_patch_entry(name):
    return name in _EXCLUDED_UNSLOTH_PATCH_NAMES or any(
        name.startswith(prefix) for prefix in _EXCLUDED_UNSLOTH_PATCH_PREFIXES
    )


_UNSLOTH_PATCH_PATHS = [p for p in sys.path if _is_unsloth_patch_path(p)]
if os.path.isdir(_KNOWN_UNSLOTH_PATCH_PATH) and _KNOWN_UNSLOTH_PATCH_PATH not in _UNSLOTH_PATCH_PATHS:
    _UNSLOTH_PATCH_PATHS.append(_KNOWN_UNSLOTH_PATCH_PATH)
sys.path[:] = [p for p in sys.path if not _is_unsloth_patch_path(p)]


def _build_filtered_unsloth_patch_path():
    sources = []
    for path in _UNSLOTH_PATCH_PATHS:
        if os.path.isdir(path) and path not in sources:
            sources.append(path)
    if not sources:
        return None
    os.makedirs(_FILTERED_UNSLOTH_PATCH_PATH, exist_ok=True)
    for src_root in sources:
        for name in os.listdir(src_root):
            if _is_excluded_patch_entry(name):
                continue
            src = os.path.join(src_root, name)
            dst = os.path.join(_FILTERED_UNSLOTH_PATCH_PATH, name)
            if os.path.lexists(dst):
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                pass
    return _FILTERED_UNSLOTH_PATCH_PATH

import torch


def restore_unsloth_patch_path():
    path = _build_filtered_unsloth_patch_path()
    if path and path not in sys.path:
        sys.path.append(path)


restore_unsloth_patch_path()
import argparse
import torch.multiprocessing as mp


DEV_KEYS = ["1ae2feb7", "3e6067c3", "b0039139", "b99e7126", "0934a4d8", "36a08778", "aa4ec2a5"]


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    if not rerun_mode:
        print(f"[Main] Save-version 7-task subset active: {DEV_KEYS}")

    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in DEV_KEYS:
                continue
        queue.put(key)
    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
