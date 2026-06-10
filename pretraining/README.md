# Pretraining

This folder contains the pretraining code used by ToLL-SGG.

## Branches

| Component | Entry | Source folder | Paper module |
| --- | --- | --- | --- |
| ACTGR + ToLL++ | `main_diff.py` | `src_diff/` | anchor-conditioned topological geometric reasoning and decoupled layout restoration |
| SMA | `main_swav.py` | `src_swav/` | structural multi-view augmentation and SwAV-style cross-view distillation |

## ACTGR + ToLL++

Run:

```bash
bash pretraining/scripts/train_tollpp_scannet.sh \
  pretraining/configs/tollpp_scannet.json
```

Main implementation points:

- Single anchor input is passed from the dataset to `PointDif`.
- The anchor node receives the 11D absolute descriptor: centroid, standard deviation, box dimensions, volume, and maximum length.
- Relative edge descriptors encode geometric offsets and size ratios.
- The recurrent graph reasoning module uses a fixed-depth GNN with GRU updates.
- ToLL++ decouples canonical shape diffusion from center and scale regression.

Important config fields:

- `root_ScanNet`: preprocessed ScanNet scene directory.
- `json_path`: subgraph training JSON produced by the preprocessing code.
- `MASK_ENCODER_INIT_PATH`: optional point encoder initialization checkpoint.
- `SCANNET_TEXT_EMB_PATH`: optional text embedding file for the extra text contrastive branch.
- `PATH` / `analysis_save_dir`: checkpoint and analysis output directories.

## SMA Structural Distillation

Run with distributed training:

```bash
NPROC_PER_NODE=2 MASTER_PORT=29501 \
bash pretraining/scripts/train_toll_sma_3dssg.sh \
  pretraining/configs/toll_sma_3dssg.json
```

Main implementation points:

- Multi-view object inputs are produced by the 3DSSG pretraining dataset.
- Student views use point and edge masking.
- Teacher views are maintained by target/EMA-style modules.
- Object, edge, and triplet prototypes are trained with SwAV/Sinkhorn-style assignments.
- The queue-based DDP trainer is implemented in `src_swav/model/ddptrain_que.py`.

Important config fields:

- `PRETRAIN_DATASET`: normally `3dssg` for this branch.
- `PRETRAIN_SPLITS`: train/validation scan split names.
- `dataset.root`: 3DSSG metadata root.
- `dataset.root_3rscan`: raw 3RScan mesh root.
- `MASK_ENCODER_INIT_PATH`: optional point encoder initialization checkpoint.
- `DIFFUSION_ENABLED`: keep the generative branch active together with SMA.
- `OBJ_LABEL_CONTRASTIVE_ENABLED`: optional auxiliary object label contrastive branch.

## Evaluation / Clustering

Both pretraining branches include feature clustering visualization utilities. Outputs are written under `analysis_save_dir`, usually:

```text
outputs/pretraining/<experiment>/analysis/
```

Use `--eval_only --eval_ckpt <checkpoint>` with the corresponding entry script to run offline clustering evaluation when supported by the branch.
