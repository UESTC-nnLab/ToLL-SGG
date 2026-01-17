import argparse
import json
import os
import re
import shutil
from pathlib import Path


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _extract_scannetpp_object_labels(segments_anno_path: Path) -> dict:
    data = _load_json(segments_anno_path)
    out = {}
    for g in data.get("segGroups", []):
        obj_id = g.get("objectId", g.get("id"))
        if obj_id is None:
            continue
        try:
            obj_id_int = int(obj_id)
        except Exception:
            continue
        label = g.get("label") or g.get("canonical_label") or g.get("name") or "unknown"
        if label is None:
            label = "unknown"
        out[str(obj_id_int)] = str(label)
    return out


def _safe_copy(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _safe_write_json(obj: dict, dst: Path, overwrite: bool, dry_run: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)
    return True


def _is_aug_scene(scene_id: str) -> bool:
    return re.match(r"^.+__aug\d+$", scene_id) is not None


def _base_scene_id(scene_id: str) -> str:
    m = re.match(r"^(?P<base>.+)__aug\d+$", scene_id)
    if not m:
        return scene_id
    return m.group("base")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_scans_dir",
        type=str,
        default="/data0/jiangxiangwei/Diff-SGG/data/ScanNet_merged_v2_20k_uniform_copy/scans",
    )
    parser.add_argument(
        "--scannet_export_scans_dir",
        type=str,
        default="/data0/jiangxiangwei/Diff-SGG/data/scannet_segments_export/scans",
    )
    parser.add_argument(
        "--scannetpp_segments_export_dir",
        type=str,
        default="/data0/jiangxiangwei/Diff-SGG/data/scannet++_segments_export",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--write_scannetpp_object_labels_in_export",
        action="store_true",
        help="Also write <scene>/object_labels.json into scannet++_segments_export for inspection.",
    )

    args = parser.parse_args()

    target_scans_dir = Path(args.target_scans_dir)
    scannet_export_scans_dir = Path(args.scannet_export_scans_dir)
    scannetpp_segments_export_dir = Path(args.scannetpp_segments_export_dir)

    if not target_scans_dir.is_dir():
        raise FileNotFoundError(f"target_scans_dir not found: {target_scans_dir}")

    all_target_scenes = sorted([p.name for p in target_scans_dir.iterdir() if p.is_dir()])
    base_scenes = [sid for sid in all_target_scenes if not _is_aug_scene(sid)]
    aug_scenes = [sid for sid in all_target_scenes if _is_aug_scene(sid)]

    stats = {
        "scannet_copied": 0,
        "scannet_skipped": 0,
        "scannet_missing": 0,
        "scannetpp_written": 0,
        "scannetpp_skipped": 0,
        "scannetpp_missing": 0,
        "aug_copied": 0,
        "aug_skipped": 0,
        "aug_missing": 0,
    }

    for scene_id in base_scenes:
        dst = target_scans_dir / scene_id / "sensorsData" / "object_labels.json"

        if scene_id.startswith("scene"):
            src = scannet_export_scans_dir / scene_id / "sensorsData" / "object_labels.json"
            if not src.exists():
                stats["scannet_missing"] += 1
                continue
            if _safe_copy(src, dst, overwrite=args.overwrite, dry_run=args.dry_run):
                stats["scannet_copied"] += 1
            else:
                stats["scannet_skipped"] += 1
            continue

        segments_anno = scannetpp_segments_export_dir / scene_id / "segments_anno.json"
        if not segments_anno.exists():
            stats["scannetpp_missing"] += 1
            continue

        labels = _extract_scannetpp_object_labels(segments_anno)
        if _safe_write_json(labels, dst, overwrite=args.overwrite, dry_run=args.dry_run):
            stats["scannetpp_written"] += 1
        else:
            stats["scannetpp_skipped"] += 1

        if args.write_scannetpp_object_labels_in_export:
            export_dst = scannetpp_segments_export_dir / scene_id / "object_labels.json"
            _safe_write_json(labels, export_dst, overwrite=args.overwrite, dry_run=args.dry_run)

    for scene_id in aug_scenes:
        base_id = _base_scene_id(scene_id)
        base_src = target_scans_dir / base_id / "sensorsData" / "object_labels.json"
        dst = target_scans_dir / scene_id / "sensorsData" / "object_labels.json"

        if not base_src.exists():
            stats["aug_missing"] += 1
            continue

        if _safe_copy(base_src, dst, overwrite=args.overwrite, dry_run=args.dry_run):
            stats["aug_copied"] += 1
        else:
            stats["aug_skipped"] += 1

    print("[DONE] build_object_labels_merged")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")


if __name__ == "__main__":
    main()
