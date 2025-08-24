import math
import os
import random

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
)

import config
import wandb
from data_loader import train_loader, val_loader
from models import get_model

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_class = get_model(config.MODEL_NAME)
model = model_class(**config.MODEL_PARAMS).to(device)
print("Number of model parameters:", sum(p.numel() for p in model.parameters()))
print(
    "Number of trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

if config.CHECKPOINT is not None:
    checkpoint = torch.load(
        os.path.join(config.OUTPUT_DIR, config.CHECKPOINT), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

if config.FREEZE_ENCODER:
    model.freeze_encoders()

criterion = nn.CrossEntropyLoss()

param_groups = [
    {"params": model.text_encoder.parameters(), "lr": config.LR / 10},
    {"params": model.audio_encoder.parameters(), "lr": config.LR / 10},
    {
        "params": (
            list(model.fusion.parameters()) + list(model.classifier.parameters())
        ),
        "lr": config.LR,
    },
]

optimizer = torch.optim.AdamW(param_groups, lr=config.LR)

# LR scheduler
num_training_steps = len(train_loader) * config.MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=math.ceil(config.WARMUP_FACTOR * num_training_steps),
    num_training_steps=num_training_steps,
)


def train():
    wandb.init(
        project=config.project_name, name=config.run_name, mode=config.wandb_mode
    )
    wandb.require("core")

    #   torch.autograd.set_detect_anomaly(True)
    best_val_f1 = 0.0
    for epoch in range(1, config.MAX_EPOCHS + 1):
        #  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_loop = tqdm(train_loader, leave=False)
        model.train()
        running_loss = 0.0
        true_answer = 0
        for step, batch in enumerate(train_loop, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_inputs=audio_inputs,
                audio_attention_mask=audio_attention_mask,
            )
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input_ids.size(0)
            true_answer += (
                (
                    logits.cpu().argmax(dim=1)
                    == torch.eye(10)[labels.cpu()].argmax(dim=1)
                )
                .sum()
                .item()
            )
            train_loop.set_description(
                f"Epoch [{epoch}/{config.MAX_EPOCHS}], Train loss {(running_loss / (step * input_ids.size(0))):.4f}, Train Accuricy {(true_answer / (step * input_ids.size(0))):.4f}"
            )
            if step % config.LOG_INTERVAL == 0:
                step_loss = running_loss / (step * input_ids.size(0))
                step_acc = true_answer / (step * input_ids.size(0))

                wandb.log(
                    {
                        "step": step,
                        "train/loss": step_loss,
                        "train/accuracy": step_acc,
                    },
                    step=step,
                )

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = true_answer / len(train_loader.dataset)

        # evolution
        model.eval()
        preds, trues, val_loss = [], [], 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=False)
            for batch in val_loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                audio_attention_mask = batch["audio_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_inputs=audio_inputs,
                    audio_attention_mask=audio_attention_mask,
                )
                loss = criterion(logits, labels)
                val_loss += loss.item() * input_ids.size(0)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())
                val_loop.set_description(f"Val loss {loss.item():.4f}")

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="macro")
        print(
            f"Epoch {epoch}/{config.MAX_EPOCHS} Train loss: {epoch_loss:.4f} Val loss: {val_loss:.4f} val_acc: {val_acc:.4f} train_acc:{train_acc:.4f} val_f1: {val_f1:.4f}"
        )
        # log to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
            }
        )
        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(
                    config.OUTPUT_DIR,
                    f"{config.FUSION_METHOD}_best_E" + str(epoch) + ".pt",
                ),
            )
            print("saved best model")

    # Clean up
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
    # gc.collect()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupted. saving model")
        # torch.save(
        #     {"model_state_dict": model.state_dict()},
        #     os.path.join(config.OUTPUT_DIR, "interrupt_multimodal_model.pt"),
        # )
