import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizerFast,
    Wav2Vec2FeatureExtractor,
)

import config

ds = load_dataset("ajyy/MELD_audio")

# build label mapping from dataset
labels = sorted(list(set(ds["train"]["emotion"])))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_LABELS = len(labels)

text_tokenizer = RobertaTokenizerFast.from_pretrained(config.BERT_MODEL)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.W2V_MODEL)


class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # sanity checks
        audio_array = item["audio"]["array"]
        assert not np.isnan(audio_array).any(), "NaN values in audio"
        assert np.isfinite(audio_array).all(), "infinite values in audio"
        return {
            "text": item["text"],
            "audio": (item["audio"]["array"], item["audio"]["sampling_rate"]),
            "label": label2id[item["emotion"]],
        }


# collate function that tokenizes and pads batch
def collate_fn(batch):
    texts = [b["text"] for b in batch]
    audios = [b["audio"][0] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sampling_rates = [item["audio"][1] for item in batch]
    # audios = [np.nan_to_num(a / np.max(np.abs(a), initial=1.0)) for a in audios]
    # tokenize texts
    tokenized = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.MAX_TEXT_LEN,
        return_tensors="pt",
    )

    # feature-extract audios
    audio_inputs = feature_extractor(
        audios,
        sampling_rate=sampling_rates[0],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    #  audio_inputs["input_values"] = torch.tanh(audio_inputs["input_values"] / 3) * 3

    batch_out = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "audio_inputs": audio_inputs["input_values"],
        "audio_attention_mask": audio_inputs.get("attention_mask", None),
        "labels": labels,
    }
    return batch_out


train_ds = MELDDataset(ds["train"])
val_ds = MELDDataset(ds["validation"])
test_ds = MELDDataset(ds["test"])

train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
)
val_loader = DataLoader(
    val_ds,
    batch_size=config.VAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
)
test_loader = DataLoader(
    test_ds,
    batch_size=config.VAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
)


if __name__ == "__main__":
    print(next(iter(val_loader)))
