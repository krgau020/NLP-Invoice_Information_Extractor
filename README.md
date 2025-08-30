
## 📖 Overview

This project implements **information extraction from documents** using **LayoutLM** on the **FUNSD dataset**.
The system learns to classify tokens (words) into categories like **Question, Answer, and Header** by leveraging both text and layout (bounding box) information.

---

## 🎯 Objectives

* Extract structured key-value pairs from scanned documents.
* Use **LayoutLM (document-aware transformer)** to understand both text and spatial layout.
* Train, validate, and run inference on custom document datasets (with OCR + bounding boxes).

---

## 🔑 Key Features

* Preprocessing that normalizes bounding boxes to LayoutLM’s expected format.
* BIO tagging scheme for labeling tokens (`B-QUESTION`, `I-ANSWER`, etc.).
* Trainer API from HuggingFace for **easy training and evaluation**.
* Validation script to compute metrics like **accuracy, precision, recall, and F1-score**.
* Inference pipeline for predicting labels on unseen documents (only requires images).
* Support for multiple checkpoints (training automatically saves progress).

---

## 📊 Dataset (FUNSD)

* The [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/) contains **scanned forms** with annotations in JSON format.
* Each annotation file contains:

  * `text`: the OCR’d word.
  * `box`: bounding box coordinates.
  * `label`: entity type (question, answer, header, or other).

Your data directory should look like this:

```
train/
  ├── images/
  └── annotations/
val/
  ├── images/
  └── annotations/
test/
  └── images/
```

---

## 🏋️ Training Workflow

1. Load JSON annotations and corresponding images.
2. Normalize bounding boxes → scale to `[0, 1000]` as required by LayoutLM.
3. Tokenize OCR’d words and align labels with subwords.
4. Train a LayoutLM model for token classification using HuggingFace’s `Trainer`.

---

## 📈 Evaluation

During validation:

* Images are passed through the model.
* Predictions are compared against ground-truth annotations.
* Metrics:

  * **Accuracy** → Overall token correctness.
  * **Precision, Recall, F1** → Per-label performance.

---

## 🔮 Inference

* On the **test set**, only document images are required.
* The trained model predicts labels for each token (e.g., Question/Answer/Header).
* Outputs can be post-processed into structured **key-value pairs**.

---

## 📦 Checkpoints

* Training automatically saves checkpoints in `./layoutlm-funsd-checkpoints/`.
* Multiple checkpoints (`checkpoint-100`, `checkpoint-200`, …) are created.
* You can resume training or run validation/inference from any checkpoint.


---

## 🙌 Acknowledgements

* [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/)
* [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
* HuggingFace Transformers library

---

