import torch
import torch.nn as nn
from pyparsing import C
from transformers import RobertaModel, Wav2Vec2Model

import config


class CrossAttentionFusion(nn.Module):
    # Fine-Grained Interaction Fusion with cross-attention layers

    def __init__(
        self,
        text_encoder_dim,
        audio_encoder_dim,
        fusion_dim,
        num_attention_head,
        dropout,
    ):
        super().__init__()
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_encoder_dim,
            num_heads=num_attention_head,
            dropout=dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(text_encoder_dim, fusion_dim)
        self.text_layer_norm = nn.LayerNorm(fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim,
            num_heads=num_attention_head,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(audio_encoder_dim, fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(fusion_dim)

        self.fusion_linear = nn.Linear(fusion_dim, fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(fusion_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_embeddings, text_embeddings):
        # Text cross attenttion text Q audio , K and V text
        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)
        text_norm = self.dropout(text_norm)

        # Audio cross attetntion Q text, K and V audio
        audio_attention, _ = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            average_attn_weights=False,
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)
        audio_norm = self.dropout(audio_norm)

        # Concatenate the text and audio embeddings
        fusion_norm = torch.cat((text_norm, audio_norm), 1)
        fusion_norm = self.dropout(fusion_norm)
        return fusion_norm


class Classifier(nn.Module):
    def __init__(self, linear_layer_output, num_classes=7):
        super().__init__()
        previous_dim = config.MemoCMT.fusion_dim
        if len(linear_layer_output) > 0:
            for i, linear_layer in enumerate(linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        return x


class MemoCMT(nn.Module):
    def __init__(self, num_classes=7):
        super(MemoCMT, self).__init__()
        self.text_encoder_dim = config.MemoCMT.text_encoder_dim
        self.audio_encoder_dim = config.MemoCMT.audio_encoder_dim
        self.fusion_dim = config.MemoCMT.fusion_dim
        self.num_attention_head = config.MemoCMT.num_attention_head
        self.linear_layer_output = config.MemoCMT.linear_layer_output
        self.fusion_head_output_type = config.MemoCMT.fusion_head_output_type
        self.dropout = config.MemoCMT.dropout
        self.num_classes = num_classes

        # Text module
        self.text_encoder = RobertaModel.from_pretrained(config.BERT_MODEL)
        # Audio module
        self.audio_encoder = Wav2Vec2Model.from_pretrained(config.W2V_MODEL)

        # Fusion module

        self.fusion = CrossAttentionFusion(
            self.text_encoder_dim,
            self.audio_encoder_dim,
            self.fusion_dim,
            self.num_attention_head,
            self.dropout,
        )
        # Classifier
        self.classifier = Classifier(
            self.linear_layer_output, num_classes=self.num_classes
        )

    def freeze_encoders(self):
        for name, param in self.text_encoder.named_parameters():
            if name not in ["pooler.dense.bias", "pooler.dense.weight"]:
                param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.audio_encoder.parameters():
            param.requires_grad = True

    def forward(
        self, input_ids, attention_mask, audio_inputs, audio_attention_mask=None
    ):
        text_embeddings = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        audio_embeddings = self.audio_encoder(
            audio_inputs, audio_attention_mask
        ).last_hidden_state

        ## Fusion Module
        fusion_norm = self.fusion(audio_embeddings, text_embeddings)
        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head

        return self.classifier(cls_token_final_fusion_norm)


if __name__ == "__main__":
    from data_loader import val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MemoCMT().to(device)
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
    print("labels", batch["labels"])
