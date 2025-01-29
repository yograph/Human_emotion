from datasets import ClassLabel
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification
from torchvision.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, ToTensor, Normalize
from sklearn.metrics import classification_report

import numpy as np
import itertools
from PIL import Image
from torch.utils.data import Dataset

import torch
import matplotlib.pyplot as plt

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


class HFDataset(Dataset):
    """
    Wrapper for Hugging Face datasets returning:
      - images (PIL => transform => Tensor)
      - class indices (not one-hot encoded)
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

        # Convert one-hot label to class index
        label_idx = torch.tensor(label_one_hot_list.index(1.0), dtype=torch.long)

        # Return a dictionary instead of a tuple
        return {
            "pixel_values": img,
            "labels": label_idx
        }

def main():
    # ---------------------------
    # CONFIGURATION
    # ---------------------------
    DATA_DIRS = [
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
        example["label"] = lab_int  # Directly use the class index
        return example

    full_dataset = full_dataset.shuffle(seed=SEED)
    full_dataset = full_dataset.map(one_hot_encode)

    # Check sample
    sample_item = full_dataset[0]
    print("\nAfter unify + one-hot encoding:")
    print("Sample label (class index):", sample_item["label"])
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

    # Define TrainingArguments
    model_name = MODEL_CHECKPOINT.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",  # Align with evaluation_strategy
        learning_rate=LR,
        lr_scheduler_type="cosine",
        auto_find_batch_size=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=1000,
        logging_steps=50,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        report_to="none"
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # --------------------------
    # 6. TRAINING
    # --------------------------
    print("\n=== Starting Training ===")
    trainer.train()

    # --------------------------
    # 7. TEST EVALUATION
    # --------------------------
    print("\n=== Final Test Evaluation ===")
    model.eval()
    test_preds, test_labels_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)  # Class indices, not one-hot

            out = model(imgs)
            preds = torch.argmax(out.logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    from sklearn.metrics import confusion_matrix
    test_cm = confusion_matrix(test_labels_list, test_preds)
    print("\nTest Classification Report:\n", classification_report(
        test_labels_list, test_preds,
        target_names=label_names, zero_division=0
    ))
    plt.figure(figsize=(10,8))
    plot_confusion_matrix(test_cm, label_names, title="Test Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()