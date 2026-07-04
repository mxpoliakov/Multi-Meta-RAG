"""Cache per-layer hidden states (last-token and mean pooled) for every query.

    python extract_hidden_states.py [--model Qwen/Qwen2.5-0.5B]
"""

import os

# Fall back to CPU for ops unsupported on MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import probe_artifacts

DATASET_FILE = "MultiHop-RAG/dataset/MultiHopRAG.json"
BATCH_SIZE = 16


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    model_name = probe_artifacts.parse_model()
    device = get_device()
    # Half precision only on CUDA; fp32 on MPS/CPU.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Model: {model_name}, device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding keeps the last real token recoverable via the mask.
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        model_name, dtype=dtype, output_hidden_states=True
    )
    model.to(device).eval()

    with open(DATASET_FILE) as f:
        queries = [q["query"] for q in json.load(f)]

    last_pooled, mean_pooled = [], []
    for start in tqdm(
        range(0, len(queries), BATCH_SIZE), desc="Extracting", unit="batch"
    ):
        batch = queries[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            hidden_states = model(**inputs).hidden_states  # tuple of [B, T, H]

        # [B, L, T, H] in fp32 for numerically stable pooling.
        hs = torch.stack(hidden_states, dim=1).float()
        mask = inputs["attention_mask"]  # [B, T]

        # Last real token per sequence (right padding -> sum(mask) - 1).
        last_idx = mask.sum(dim=1) - 1
        batch_idx = torch.arange(hs.size(0), device=device)
        last = hs[batch_idx, :, last_idx]  # [B, L, H]

        m = mask[:, None, :, None].float()  # [B, 1, T, 1]
        mean = (hs * m).sum(dim=2) / m.sum(dim=2)  # [B, L, H]

        last_pooled.append(last.cpu().numpy())
        mean_pooled.append(mean.cpu().numpy())

    last_array = np.concatenate(last_pooled)
    mean_array = np.concatenate(mean_pooled)

    last_file = probe_artifacts.hidden_states_path("last", model_name)
    mean_file = probe_artifacts.hidden_states_path("mean", model_name)
    np.save(last_file, last_array)
    np.save(mean_file, mean_array)
    print(f"Saved {last_array.shape} -> {last_file}")
    print(f"Saved {mean_array.shape} -> {mean_file}")


if __name__ == "__main__":
    main()
