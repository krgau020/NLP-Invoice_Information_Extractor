#training script for LayoutLM on FUNSD dataset



import os
import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification, Trainer, TrainingArguments
from PIL import Image
import glob

# -------------------------------
# 1️⃣ CONFIG
# -------------------------------

MODEL_NAME = "microsoft/layoutlm-base-uncased"

label2id = {
    "O": 0,
    "B-QUESTION": 1,
    "I-QUESTION": 2,
    "B-ANSWER": 3,
    "I-ANSWER": 4,
    "B-HEADER": 5,
    "I-HEADER": 6
}
id2label = {v: k for k, v in label2id.items()}

# -------------------------------
# 2️⃣ NORMALIZE BOX
# -------------------------------

def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height)
    ]

# -------------------------------
# 3️⃣ LOAD FUNSD ANNOTATIONS
# -------------------------------

# ⚠️ NEW: Pass in the images folder path
def load_funsd(json_path, images_folder):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 📸 Get image filename based on JSON filename
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    image_path = os.path.join(images_folder, base_name + ".png")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    words, boxes, labels = [], [], []

    for form in data["form"]:
        label = form["label"]
        bio_tag = "O"
        if label == "question":
            bio_tag = "B-QUESTION"
        elif label == "answer":
            bio_tag = "B-ANSWER"
        elif label == "header":
            bio_tag = "B-HEADER"

        for i, word in enumerate(form["words"]):
            w = word["text"]
            box = word["box"]
            box = normalize_box(box, width, height)
            words.append(w)
            boxes.append(box)

            if bio_tag == "O":
                labels.append("O")
            elif i == 0:
                labels.append(bio_tag)
            else:
                labels.append("I" + bio_tag[1:])

    return {"words": words, "boxes": boxes, "labels": labels}

# -------------------------------
# 4️⃣ TOKENIZE & ALIGN
# -------------------------------

def encode_data(sample, tokenizer, max_length=512):
    words = sample["words"]
    boxes = sample["boxes"]
    labels = sample["labels"]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    aligned_labels = []
    aligned_boxes = []

    word_ids = encoding.word_ids()
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_labels.append(label2id[labels[word_idx]])
            aligned_boxes.append(boxes[word_idx])

    encoding.pop("offset_mapping")
    encoding["bbox"] = aligned_boxes
    encoding["labels"] = aligned_labels

    return encoding

# -------------------------------
# 5️⃣ DATASET
# -------------------------------

class FunsdDataset(Dataset):
    def __init__(self, file_list, tokenizer, images_folder):
        self.encodings = []
        for f in file_list:
            sample = load_funsd(f, images_folder)
            enc = encode_data(sample, tokenizer)
            self.encodings.append(enc)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        return item

# -------------------------------
# 6️⃣ MAIN
# -------------------------------

if __name__ == "__main__":
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)

    json_folder = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\annotations"
    images_folder = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\images"

    files = glob.glob(f"{json_folder}/*.json")
    print(f"Found {len(files)} JSON files.")

    dataset = FunsdDataset(files, tokenizer, images_folder)

    model = LayoutLMForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./layoutlm-funsd-checkpoints",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_steps=10,
        learning_rate=5e-5,
        save_steps=50,
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()





