"""
Training Pipeline for MultiModal Disaster Prediction
======================================================
End-to-end training script with MLflow experiment tracking,
early stopping, and learning rate scheduling.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from pathlib import Path

from src.models.fusion_model import MultiModalFusionNet


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Run one training epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        satellite_img = batch["image"].to(device)
        sensor_data   = batch["sensor"].to(device)
        labels        = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(satellite_img, sensor_data)
            loss = criterion(output["logits"], labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = output["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Validation", leave=False):
        satellite_img = batch["image"].to(device)
        sensor_data   = batch["sensor"].to(device)
        labels        = batch["label"].to(device)

        output = model(satellite_img, sensor_data)
        loss = criterion(output["logits"], labels)

        total_loss += loss.item()
        probs = output["probs"].cpu().numpy()
        preds = output["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


def train(config: dict):
    """
    Main training loop.
    
    Args:
        config: Training configuration dictionary from YAML config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize model
    model = MultiModalFusionNet(
        image_channels=config["model"]["image_channels"],
        sensor_input_size=config["model"]["sensor_input_size"],
        feature_dim=config["model"]["feature_dim"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Checkpointing
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    # MLflow tracking
    mlflow.set_experiment(config.get("experiment_name", "disaster-prediction"))

    with mlflow.start_run():
        mlflow.log_params({
            "lr": config["training"]["lr"],
            "epochs": config["training"]["epochs"],
            "feature_dim": config["model"]["feature_dim"],
            "num_classes": config["model"]["num_classes"]
        })

        for epoch in range(1, config["training"]["epochs"] + 1):
            print(f"\nEpoch {epoch}/{config['training']['epochs']}")

            # NOTE: Replace with actual DataLoaders
            # train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            # val_loss, val_f1 = validate(model, val_loader, criterion, device)

            # Log metrics
            # mlflow.log_metrics({"train_loss": train_loss, "train_f1": train_f1,
            #                     "val_loss": val_loss, "val_f1": val_f1}, step=epoch)

            scheduler.step()

            # Early stopping & checkpointing
            # if val_f1 > best_val_f1:
            #     best_val_f1 = val_f1
            #     patience_counter = 0
            #     torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            #     mlflow.pytorch.log_model(model, "best_model")
            #     print(f"  New best model saved (val_f1={val_f1:.4f})")
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print("Early stopping triggered.")
            #         break

        print("Training complete.")


if __name__ == "__main__":
    # Default config for quick test
    config = {
        "experiment_name": "disaster-prediction",
        "model": {
            "image_channels": 13,
            "sensor_input_size": 32,
            "feature_dim": 512,
            "num_classes": 4
        },
        "training": {
            "epochs": 50,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "checkpoint_dir": "checkpoints",
            "early_stopping_patience": 10
        }
    }
    train(config)
