import torch
import torch.nn as nn
from transformers import (
    RobertaModel,
    Wav2Vec2Model,
)

import config


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        weights = torch.softmax(self.attn(hidden_states), dim=1)
        return torch.sum(weights * hidden_states, dim=1)


class AttentionPoolingModel(nn.Module):
    def __init__(self, num_labels=7, dropout=0.1):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained(config.BERT_MODEL)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(config.W2V_MODEL)

        self.text_pool = AttentionPooling(self.text_encoder.config.hidden_size)
        self.audio_pool = AttentionPooling(self.audio_encoder.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size
            + self.audio_encoder.config.hidden_size,
            num_labels,
        )

    def forward(
        self, input_ids, attention_mask, audio_inputs, audio_attention_mask=None
    ):
        text_hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_emb = self.text_pool(text_hidden)

        audio_hidden = self.audio_encoder(
            audio_inputs, attention_mask=audio_attention_mask
        ).last_hidden_state
        audio_emb = self.audio_pool(audio_hidden)
        combined = torch.cat([text_emb, audio_emb], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)


if __name__ == "__main__":
    from data_loader import val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionPoolingModel().to(device)
    batch = next(iter(val_loader))
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

    print("logits", logits)
