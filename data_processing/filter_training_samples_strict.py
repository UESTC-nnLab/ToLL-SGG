import argparse
import json
import os
from collections import Counter

import numpy as np


def filter_scene(scene_id: str, scene_data: dict, scans_root: str):
    sensors_dir = os.path.join(scans_root, scene_id, "sensorsData")
    inst_path = os.path.join(sensors_dir, "instance.npy")

    if not os.path.exists(inst_path):
        return None, {"missing_instance_npy": 1}

    inst = np.load(inst_path)
    present = set(np.unique(inst).tolist())

    subgraphs_in = scene_data.get("subgraphs", [])
    subgraphs_out = []

    stats = Counter()
    stats["subgraphs_in"] += len(subgraphs_in)

    for sg in subgraphs_in:
        nodes = [int(n) for n in sg.get("nodes", [])]
        edges = sg.get("edges", [])
        anchor = sg.get("anchor", None)

        nodes_f = [n for n in nodes if n in present]
        nodes_set = set(nodes_f)

        edges_f = []
        for e in edges:
            if e is None or len(e) < 2:
                continue
            u, v = int(e[0]), int(e[1])
            if u in nodes_set and v in nodes_set:
                edges_f.append([u, v])

        if len(nodes_f) == 0:
            stats["drop_subgraph_empty_nodes"] += 1
            continue

        if len(edges_f) == 0:
            stats["drop_subgraph_empty_edges"] += 1
            continue

        if anchor is None or int(anchor) not in nodes_set:
            anchor_f = nodes_f[0]
            stats["fix_anchor"] += 1
        else:
            anchor_f = int(anchor)

        sg_out = dict(sg)
        sg_out["nodes"] = nodes_f
        sg_out["edges"] = edges_f
        sg_out["anchor"] = anchor_f
        subgraphs_out.append(sg_out)

    stats["subgraphs_out"] += len(subgraphs_out)

    if len(subgraphs_out) == 0:
        stats["drop_scene_no_valid_subgraphs"] += 1
        return None, stats

    scene_out = dict(scene_data)
    scene_out["scene_id"] = scene_id
    scene_out["subgraphs"] = subgraphs_out
    return scene_out, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/ScanNet_merged/training_samples2.json",
    )
    parser.add_argument(
        "--scans_root",
        type=str,
        default="data/ScanNet_merged/scans",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/ScanNet_merged/training_samples2_strict.json",
    )
    args = parser.parse_args()

    input_json = os.path.abspath(args.input_json)
    scans_root = os.path.abspath(args.scans_root)
    output_json = os.path.abspath(args.output_json)

    with open(input_json, "r") as f:
        all_scenes = json.load(f)

    out = {}
    agg = Counter()

    for scene_id, scene_data in all_scenes.items():
        scene_out, stats = filter_scene(scene_id, scene_data, scans_root)
        agg.update(stats)
        if scene_out is not None:
            out[scene_id] = scene_out

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print("input_json:", input_json)
    print("scans_root:", scans_root)
    print("output_json:", output_json)
    print("scenes_in:", len(all_scenes))
    print("scenes_out:", len(out))
    for k in sorted(agg.keys()):
        print(f"{k}: {agg[k]}")


if __name__ == "__main__":
    main()
