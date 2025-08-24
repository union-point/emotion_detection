import torch
import torch.nn as nn
from transformers import RobertaModel, Wav2Vec2Model

import config


class EarlyFusion(nn.Module):
    def __init__(self, num_labels=7):
        super().__init__()
        # text encoder
        self.text_encoder = RobertaModel.from_pretrained(config.BERT_MODEL)
        # audio encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained(config.W2V_MODEL)

        # classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.text_encoder.config.hidden_size
            + self.audio_encoder.config.hidden_size,
            num_labels,
        )

    def forward(
        self, input_ids, attention_mask, audio_inputs, audio_attention_mask=None
    ):
        # text forward
        bert_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.pooler_output
        # for models wich dont have pooler_output
        if text_emb is None:
            text_emb = bert_out.last_hidden_state.mean(dim=1)

        # audio forward
        wav_out = self.audio_encoder(audio_inputs, attention_mask=audio_attention_mask)
        audio_emb = wav_out.last_hidden_state.mean(dim=1)

        # concat
        combined = torch.cat([text_emb, audio_emb], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits


if __name__ == "__main__":
    from data_loader import val_loader

    model = EarlyFusion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print("train mode logits", logits)
