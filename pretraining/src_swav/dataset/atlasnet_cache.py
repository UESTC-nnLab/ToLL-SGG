import os

import torch


def make_object_key(scan_id, instance_id):
    return f"{str(scan_id)}::{int(instance_id)}"


def make_sample_object_key(sample_id, instance_id):
    return f"{str(sample_id)}::{int(instance_id)}"


def load_embedding_cache(cache_path):
    if not cache_path:
        return None

    cache_path = os.path.expanduser(str(cache_path))
    if not os.path.exists(cache_path):
        return None

    cache = torch.load(cache_path, map_location="cpu")
    if "embeddings" not in cache or "keys" not in cache:
        raise ValueError(
            f"Invalid AtlasNet cache at {cache_path}: expected 'embeddings' and 'keys'."
        )

    embeddings = cache["embeddings"]
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.float().contiguous()

    keys = [str(key) for key in cache["keys"]]
    if embeddings.ndim != 2:
        raise ValueError(
            f"Invalid AtlasNet embeddings at {cache_path}: expected 2D tensor, got {tuple(embeddings.shape)}."
        )
    if len(keys) != embeddings.shape[0]:
        raise ValueError(
            f"AtlasNet cache size mismatch at {cache_path}: {len(keys)} keys vs {embeddings.shape[0]} embeddings."
        )

    key_to_index = {key: idx for idx, key in enumerate(keys)}
    latent_dim = int(cache.get("latent_dim", embeddings.shape[1]))

    return {
        "path": cache_path,
        "embeddings": embeddings,
        "keys": keys,
        "key_to_index": key_to_index,
        "latent_dim": latent_dim,
        "meta": cache.get("meta", {}),
    }


def fetch_object_embeddings(cache_bundle, scan_id, instance_ids, sample_id=None):
    if cache_bundle is None:
        return None, None

    latent_dim = int(cache_bundle["latent_dim"])
    embeddings = torch.zeros((len(instance_ids), latent_dim), dtype=torch.float32)
    valid_mask = torch.zeros((len(instance_ids),), dtype=torch.bool)

    cache_tensor = cache_bundle["embeddings"]
    key_to_index = cache_bundle["key_to_index"]

    for row_idx, instance_id in enumerate(instance_ids):
        cache_idx = None
        if sample_id is not None:
            sample_object_key = make_sample_object_key(sample_id, instance_id)
            cache_idx = key_to_index.get(sample_object_key)
        if cache_idx is None:
            object_key = make_object_key(scan_id, instance_id)
            cache_idx = key_to_index.get(object_key)
        if cache_idx is None:
            continue
        embeddings[row_idx] = cache_tensor[cache_idx]
        valid_mask[row_idx] = True

    return embeddings, valid_mask
