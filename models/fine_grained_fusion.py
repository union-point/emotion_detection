import torch
import torch.nn as nn
from transformers import RobertaModel, Wav2Vec2Model

import config


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        weights = torch.softmax(self.attn(hidden_states), dim=1)
        return torch.sum(weights * hidden_states, dim=1)


class CrossAttentionFusion(nn.Module):
    # Fine-Grained Interaction Fusion with cross-attention layers

    def __init__(self, text_dim, audio_dim, hidden_dim):
        super().__init__()
        # project both into same hidden dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # cross-attention: text attends to audio and vice versa
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1
        )

    def forward(self, text_hidden, audio_hidden, text_mask=None, audio_mask=None):
        # project to same space
        text_h = self.text_proj(text_hidden)
        audio_h = self.audio_proj(audio_hidden)
        print("text_h", text_h.shape)
        print("audio_h", audio_h.shape)
        # text queries audio
        text_cross, _ = self.text_to_audio(
            query=text_h,
            key=audio_h,
            value=audio_h,
            key_padding_mask=(~audio_mask.bool()) if audio_mask is not None else None,
        )
        # audio queries text
        audio_cross, _ = self.audio_to_text(
            query=audio_h,
            key=text_h,
            value=text_h,
            key_padding_mask=(~text_mask.bool()) if text_mask is not None else None,
        )

        # fuse original + cross signals
        fused_text = text_h + text_cross
        fused_audio = audio_h + audio_cross
        return fused_text, fused_audio


class FineGrainedFusion(nn.Module):
    def __init__(self, num_labels=7, dropout=0.1, hidden_dim=256):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(config.BERT_MODEL)
        self.wav2vec = Wav2Vec2Model.from_pretrained(config.W2V_MODEL)

        self.fusion = CrossAttentionFusion(
            text_dim=self.bert.config.hidden_size,
            audio_dim=self.wav2vec.config.hidden_size,
            hidden_dim=hidden_dim,
        )

        self.text_pool = AttentionPooling(hidden_dim)
        self.audio_pool = AttentionPooling(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def freeze_encoder(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = True

    def forward(
        self, input_ids, attention_mask, audio_inputs, audio_attention_mask=None
    ):
        # text encoding
        text_hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        # audio encoding
        audio_hidden = self.wav2vec(
            audio_inputs, attention_mask=audio_attention_mask
        ).last_hidden_state
        print("audio_hidden", audio_hidden.shape)
        print("text_hidden", text_hidden.shape)
        if audio_attention_mask is not None:
            # downsample raw mask to match audio_hidden length
            new_len = audio_hidden.shape[1]
            audio_mask = (
                torch.nn.functional.interpolate(
                    audio_attention_mask[:, None, :].float(),
                    size=new_len,
                    mode="nearest",
                )
                .squeeze(1)
                .bool()
            )
        else:
            audio_mask = None
        # fine-grained fusion
        fused_text, fused_audio = self.fusion(
            text_hidden,
            audio_hidden,
            text_mask=attention_mask,
            audio_mask=audio_mask,
        )

        # pool fused representations
        text_emb = self.text_pool(fused_text)
        audio_emb = self.audio_pool(fused_audio)

        # final classification
        combined = torch.cat([text_emb, audio_emb], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)


if __name__ == "__main__":
    from data_loader import test_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FineGrainedFusion().to(device)
    batch = next(iter(test_loader))
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
    print("labels", batch["labels"])
