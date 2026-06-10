# Fine-tuning

This folder contains the downstream 3D Scene Graph Generation modules used after ToLL pretraining.

## Structure

```text
finetuning/
  main.py
  configs/mmgnet.json
  scripts/train_3dssg.sh
  src/
    dataset/
    model/SGFN_MMG/
    utils/
```

The main downstream model family lives in `src/model/SGFN_MMG/`. The config file keeps the common SGFN-MMG style settings for object and predicate prediction.

## Configure

Edit `configs/mmgnet.json` before training:

- `MODEL.use_pretrain`: path to a ToLL pretrained checkpoint.
- `dataset.root`: 3DSSG metadata root. The repository includes lightweight metadata under `../data/3DSSG_subset`.
- `dataset.root_3rscan`: path to the raw 3RScan scenes.
- `MODEL.obj_label_path`: object class list, normally `data/3DSSG_subset/classes.txt` from the repository root.
- `MODEL.rel_label_path`: relation class list, normally `data/3DSSG_subset/relations.txt` from the repository root.
- `MODEL.adapter_path`: optional CLIP adapter checkpoint.

## Run

From the repository root:

```bash
bash finetuning/scripts/train_3dssg.sh finetuning/configs/mmgnet.json
```

## Notes

The fine-tuning code is kept as the downstream module stack for 3DSSG experiments. If you plug this into another 3DSGG training framework, load ToLL weights into the object encoder, relation encoder, and graph reasoning modules, then train the object and predicate heads with the standard 3DSSG supervision.

For release, replace the checkpoint placeholder in `MODEL.use_pretrain` with the public ToLL checkpoint link or the local downloaded path.
