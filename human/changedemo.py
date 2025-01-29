#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose, Resize, RandomRotation, RandomHorizontalFlip,
    ToTensor, Normalize
)
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Hugging Face-specific imports
from datasets import load_dataset, concatenate_datasets, ClassLabel
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ------------------------------------------------------------------------------
# 1) HELPER FUNCTION FOR CONFUSION MATRIX PLOTTING
# ------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Visualization helper for a confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ------------------------------------------------------------------------------
# 2) DATASET CLASS (FORCES GRAYSCALE -> RGB)
# ------------------------------------------------------------------------------
class HFDataset(Dataset):
    """
    Wrapper for Hugging Face datasets returning:
      - images (PIL => transform => Tensor)
      - one-hot labels
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = sample['image']
        label_one_hot_list = sample['label']  # e.g. [1.0, 0.0, 0.0, ...]

        # Convert to 3-channel RGB if not already
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Convert label list -> FloatTensor
        label_one_hot = torch.tensor(label_one_hot_list, dtype=torch.float)
        return img, label_one_hot


# ------------------------------------------------------------------------------
# 3) MAIN FUNCTION
# ------------------------------------------------------------------------------
def main():
    # ---------------------------
    # CONFIGURATION
    # ---------------------------
    DATA_DIRS = [
        # "human/archive",        # Removed as requested
        "fer2013",
        "mma-facial-expression",
        "testingemotion"
    ]
    MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

    BATCH_SIZE = 32
    NUM_EPOCHS = 3
    LR = 3e-5
    SEED = 42
    WEIGHT_DECAY = 1e-5

    # Final 7 classes to unify
    FINAL_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    final_label_feature = ClassLabel(names=FINAL_LABELS)

    # ----------------------------------------------------------------
    # unify_labels function
    # ----------------------------------------------------------------
    def unify_labels(ds):
        """
        ds is a split from load_dataset("imagefolder", data_dir=...).

        Steps:
         1) If 'label' not in ds.features => no labels => skip
         2) Remove 'contempt' if present
         3) Rename 'anger' -> 'angry'
         4) Keep only final 7
         5) Cast to ClassLabel(7)
        """
        if "label" not in ds.features:
            print(" -> No 'label' column => skipping this dataset.")
            return None

        old_label_names = ds.features["label"].names

        # remove 'contempt'
        def keep_not_contempt(example):
            lbl_str = old_label_names[example["label"]]
            return (lbl_str != "contempt")
        ds = ds.filter(keep_not_contempt)

        # rename 'anger'->'angry'
        def rename_anger(example):
            idx = example["label"]
            old_str = old_label_names[idx]
            if old_str == "anger":
                example["new_label_str"] = "angry"
            else:
                example["new_label_str"] = old_str
            return example
        ds = ds.map(rename_anger)

        # keep only final 7
        def keep_in_7(example):
            return (example["new_label_str"] in FINAL_LABELS)
        ds = ds.filter(keep_in_7)

        if len(ds) == 0:
            print(" -> No images left after unify => skipping.")
            return None

        def map_to_final(example):
            lbl = example["new_label_str"]
            new_id = final_label_feature.str2int(lbl)
            example["label"] = new_id
            return example
        ds = ds.map(map_to_final, remove_columns=["new_label_str"])

        ds = ds.cast_column("label", final_label_feature)
        return ds

    # --------------------------
    # 1. LOAD & CONCAT DATASETS
    # --------------------------
    from datasets import load_dataset, concatenate_datasets
    all_splits = []
    for path in DATA_DIRS:
        print(f"Loading dataset from: {path}")
        ds = load_dataset("imagefolder", data_dir=path)
        ds_keys = list(ds.keys())
        if "train" in ds_keys:
            single_split = ds["train"]
        else:
            single_split = ds[ds_keys[0]]

        unified_split = unify_labels(single_split)
        if unified_split is None or len(unified_split) == 0:
            print(f" -> Skipping {path} because unify gave empty/no label data.")
            continue
        all_splits.append(unified_split)

    if not all_splits:
        raise ValueError("No valid labeled data found in the given directories!")

    full_dataset = concatenate_datasets(all_splits)

    label_names = FINAL_LABELS
    num_classes = len(label_names)

    print(f"\nCombined dataset has {len(full_dataset)} samples total.")
    print("Label names:", label_names)

    # One-hot encode
    def one_hot_encode(example):
        lab_int = example["label"]
        vec = [0.0]*num_classes
        vec[lab_int] = 1.0
        example["label"] = vec
        return example

    full_dataset = full_dataset.shuffle(seed=SEED)
    full_dataset = full_dataset.map(one_hot_encode)

    # Check sample
    sample_item = full_dataset[0]
    print("\nAfter unify + one-hot encoding:")
    print("Sample label (one-hot):", sample_item["label"])
    print("Sample image size:", sample_item["image"].size)

    # --------------------------
    # 2. SPLIT 60%/20%/20%
    # --------------------------
    split_1 = full_dataset.train_test_split(test_size=0.4, seed=SEED)
    train_data = split_1["train"]
    remain_data = split_1["test"]

    split_2 = remain_data.train_test_split(test_size=0.5, seed=SEED)
    val_data = split_2["train"]
    test_data = split_2["test"]

    print(f"Train size={len(train_data)}, Val size={len(val_data)}, Test size={len(test_data)}")

    # --------------------------
    # 3. CREATE TRANSFORMS
    # --------------------------
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    image_mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    image_std  = getattr(processor, "image_std",  [0.229, 0.224, 0.225])

    # For training: random augment
    train_transform = Compose([
        Resize((224, 224)),
        RandomRotation(30),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(image_mean, image_std)
    ])

    # For val/test: minimal
    eval_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(image_mean, image_std)
    ])

    # --------------------------
    # 4. CREATE DATALOADERS
    # --------------------------
    train_dataset = HFDataset(train_data, transform=train_transform)
    val_dataset   = HFDataset(val_data,   transform=eval_transform)
    test_dataset  = HFDataset(test_data,  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --------------------------
    # 5. MODEL & OPTIMIZER
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------
    # 6. TRAINING LOOP (One-Hot)
    # --------------------------
    print("\n=== Starting Training ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, label_one_hot) in enumerate(train_loader):
            imgs = imgs.to(device)
            label_one_hot = label_one_hot.to(device)

            outputs = model(imgs)  # [B, 7]
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=1)

            # cross-entropy: -sum(one_hot * log_probs)/bs
            loss = -(label_one_hot * log_probs).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)}]"
                      f" Loss={loss.item():.4f}")

        avg_loss = total_loss/len(train_loader)
        print(f"\n==> Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Train Loss: {avg_loss:.4f}")

        # VALIDATION
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for i, (imgs, label_one_hot) in enumerate(val_loader):
                print(f"Val batch {i}")
                imgs = imgs.to(device)
                label_one_hot = label_one_hot.to(device)

                out = model(imgs)
                preds = torch.argmax(out.logits, dim=1)
                true_cls = torch.argmax(label_one_hot, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(true_cls.cpu().numpy())

        # classification report
        from sklearn.metrics import classification_report
        val_report = classification_report(
            val_labels_list, val_preds,
            target_names=label_names, zero_division=0
        )
        print(f"\n[Epoch {epoch+1} Validation]\n{val_report}")

    # --------------------------
    # 7. TEST EVALUATION
    # --------------------------
    print("\n=== Final Test Evaluation ===")
    model.eval()
    test_preds, test_labels_list = [], []

    with torch.no_grad():
        for imgs, label_one_hot in test_loader:
            imgs = imgs.to(device)
            label_one_hot = label_one_hot.to(device)

            out = model(imgs)
            preds = torch.argmax(out.logits, dim=1)
            true_cls = torch.argmax(label_one_hot, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(true_cls.cpu().numpy())

    from sklearn.metrics import confusion_matrix
    test_cm = confusion_matrix(test_labels_list, test_preds)
    print("\nTest Classification Report:\n", classification_report(
        test_labels_list, test_preds,
        target_names=label_names, zero_division=0
    ))
    plt.figure(figsize=(10,8))
    plot_confusion_matrix(test_cm, label_names, title="Test Confusion Matrix")
    plt.show()


# ------------------------------------------------------------------------------
# 8) RUN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
