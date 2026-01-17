import argparse
import json
import os
import random
import shutil
from typing import Any, Dict, List, Tuple


def _flatten_samples(all_scenes_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    flat: List[Tuple[str, Dict[str, Any]]] = []
    for scene_id, scene_data in all_scenes_data.items():
        subgraphs = scene_data.get("subgraphs", [])
        for subgraph in subgraphs:
            flat.append((scene_id, subgraph))
    return flat


def _select_uniform_per_scene(
    all_scenes_data: Dict[str, Any],
    num_subgraphs: int,
    seed: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    rng = random.Random(seed)
    scene_ids = [sid for sid, sdata in all_scenes_data.items() if len(sdata.get("subgraphs", [])) > 0]
    if not scene_ids:
        return []

    scene_ids_sorted = sorted(scene_ids)
    num_scenes = len(scene_ids_sorted)
    target = num_subgraphs

    base = target // num_scenes
    alloc: Dict[str, int] = {}
    capacity: Dict[str, int] = {}
    for sid in scene_ids_sorted:
        cap = len(all_scenes_data[sid].get("subgraphs", []))
        capacity[sid] = cap
        alloc[sid] = min(base, cap)

    remaining = target - sum(alloc.values())
    if remaining > 0:
        candidates = [sid for sid in scene_ids_sorted if alloc[sid] < capacity[sid]]
        rng.shuffle(candidates)

        idx = 0
        while remaining > 0 and candidates:
            sid = candidates[idx % len(candidates)]
            if alloc[sid] < capacity[sid]:
                alloc[sid] += 1
                remaining -= 1
            idx += 1

            if idx % len(candidates) == 0:
                candidates = [s for s in candidates if alloc[s] < capacity[s]]

    selected: List[Tuple[str, Dict[str, Any]]] = []
    for sid in scene_ids_sorted:
        k = alloc[sid]
        if k <= 0:
            continue
        subgraphs = list(all_scenes_data[sid].get("subgraphs", []))
        idxs = list(range(len(subgraphs)))
        rng.shuffle(idxs)
        for i in idxs[:k]:
            selected.append((sid, subgraphs[i]))

    rng.shuffle(selected)
    if len(selected) > target:
        selected = selected[:target]
    return selected


def _rebuild_json(all_scenes_data: Dict[str, Any], selected: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scene_id, subgraph in selected:
        if scene_id not in out:
            scene_data = all_scenes_data.get(scene_id, {})
            out[scene_id] = dict(scene_data)
            out[scene_id]["scene_id"] = out[scene_id].get("scene_id", scene_id)
            out[scene_id]["subgraphs"] = []
        out[scene_id]["subgraphs"].append(subgraph)
    return out


def _ensure_symlinks(scans_in: str, scans_out: str, scene_ids: List[str]) -> None:
    os.makedirs(scans_out, exist_ok=True)
    for scene_id in scene_ids:
        src = os.path.join(scans_in, scene_id)
        dst = os.path.join(scans_out, scene_id)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Scene directory not found: {src}")
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)


def _copy_minimal_scene(src_scene_dir: str, dst_scene_dir: str) -> None:
    src_sd = os.path.join(src_scene_dir, "sensorsData")
    dst_sd = os.path.join(dst_scene_dir, "sensorsData")
    os.makedirs(dst_sd, exist_ok=True)

    required = [
        os.path.join(src_sd, "points.npy"),
        os.path.join(src_sd, "instance.npy"),
    ]
    optional = [
        os.path.join(src_sd, "object_labels.json"),
    ]

    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file missing: {p}")

    for p in required + optional:
        if not os.path.exists(p):
            continue
        dst_p = os.path.join(dst_sd, os.path.basename(p))
        if os.path.exists(dst_p):
            continue
        shutil.copy2(p, dst_p)


def _copy_scenes(scans_in: str, scans_out: str, scene_ids: List[str], copy_mode: str) -> None:
    os.makedirs(scans_out, exist_ok=True)
    for scene_id in scene_ids:
        src = os.path.join(scans_in, scene_id)
        dst = os.path.join(scans_out, scene_id)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Scene directory not found: {src}")
        if os.path.exists(dst):
            continue

        if copy_mode == "copy_all":
            shutil.copytree(src, dst, symlinks=False)
        elif copy_mode == "copy_min":
            os.makedirs(dst, exist_ok=True)
            _copy_minimal_scene(src_scene_dir=src, dst_scene_dir=dst)
        else:
            raise ValueError(f"Unknown copy mode: {copy_mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=str, required=True)
    parser.add_argument("--in_json", type=str, default="training_samples2.json")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="training_samples2.json")
    parser.add_argument("--num_subgraphs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="uniform_scene",
        choices=["uniform_scene", "global_random"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy_min",
        choices=["copy_min", "copy_all", "symlink", "none"],
    )
    args = parser.parse_args()

    in_json_path = args.in_json if os.path.isabs(args.in_json) else os.path.join(args.in_root, args.in_json)
    if not os.path.exists(in_json_path):
        raise FileNotFoundError(f"Input json not found: {in_json_path}")

    scans_in = os.path.join(args.in_root, "scans")
    if not os.path.isdir(scans_in):
        raise FileNotFoundError(f"Input scans dir not found: {scans_in}")

    os.makedirs(args.out_root, exist_ok=True)

    with open(in_json_path, "r") as f:
        all_scenes_data = json.load(f)

    if args.sample_strategy == "uniform_scene":
        selected = _select_uniform_per_scene(
            all_scenes_data=all_scenes_data,
            num_subgraphs=args.num_subgraphs,
            seed=args.seed,
        )
        total = sum(len(v.get("subgraphs", [])) for v in all_scenes_data.values())
    else:
        flat = _flatten_samples(all_scenes_data)
        total = len(flat)
        if total == 0:
            raise RuntimeError("Input json contains 0 subgraphs")
        k = min(args.num_subgraphs, total)
        rng = random.Random(args.seed)
        indices = list(range(total))
        rng.shuffle(indices)
        selected = [flat[i] for i in indices[:k]]

    if not selected:
        raise RuntimeError("No subgraphs selected. Check input json.")

    k = len(selected)

    out_data = _rebuild_json(all_scenes_data, selected)
    out_json_path = args.out_json if os.path.isabs(args.out_json) else os.path.join(args.out_root, args.out_json)

    with open(out_json_path, "w") as f:
        json.dump(out_data, f)

    scene_ids = sorted(out_data.keys())
    if args.mode == "symlink":
        scans_out = os.path.join(args.out_root, "scans")
        _ensure_symlinks(scans_in=scans_in, scans_out=scans_out, scene_ids=scene_ids)
    elif args.mode in ("copy_min", "copy_all"):
        scans_out = os.path.join(args.out_root, "scans")
        _copy_scenes(scans_in=scans_in, scans_out=scans_out, scene_ids=scene_ids, copy_mode=args.mode)

    print(f"[OK] in_json: {in_json_path}")
    print(f"[OK] total_subgraphs: {total}")
    print(f"[OK] selected_subgraphs: {k}")
    print(f"[OK] selected_scenes: {len(scene_ids)}")
    print(f"[OK] out_root: {args.out_root}")
    print(f"[OK] out_json: {out_json_path}")
    if args.mode != "none":
        print(f"[OK] out_scans: {os.path.join(args.out_root, 'scans')}")


if __name__ == "__main__":
    main()
