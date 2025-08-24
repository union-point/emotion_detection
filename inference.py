import random

import numpy as np
import torch
import torch.nn as nn

import config
from data_loader import test_loader
from models.fine_grained_fusion import MultiModalEmotionClassifier

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiModalEmotionClassifier().to(device)

# TODO: pass from arguments
checkpoint = torch.load("best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
# TODO make possible inferance on a new data
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_inputs = batch["audio_inputs"].to(device)
        audio_attention_mask = batch["audio_attention_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_inputs=audio_inputs,
            audio_attention_mask=audio_attention_mask,
        )
        # TODO return emotion, no class number
        print("out", logits.cpu().argmax(dim=1))
