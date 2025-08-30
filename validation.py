import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ CONFIG (Same as training)
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
# 2️⃣ NORMALIZE BOX (Same as training)
# -------------------------------

def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height)
    ]

# -------------------------------
# 3️⃣ LOAD FUNSD ANNOTATIONS (Same as training)
# -------------------------------

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
# 4️⃣ TOKENIZE & ALIGN (Same as training)
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
# 5️⃣ VALIDATION DATASET
# -------------------------------

class FunsdValidationDataset(Dataset):
    def __init__(self, file_list, tokenizer, images_folder):
        self.encodings = []
        self.original_labels = []  # Store original labels for evaluation
        
        for f in file_list:
            sample = load_funsd(f, images_folder)
            enc = encode_data(sample, tokenizer)
            self.encodings.append(enc)
            self.original_labels.append(sample["labels"])

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        return item

# -------------------------------
# 6️⃣ EVALUATION FUNCTIONS
# -------------------------------

def align_predictions(predictions, label_ids, tokenizer):
    """
    Align predictions with original word-level labels
    """
    preds = np.argmax(predictions, axis=2)
    
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:
                out_label_list[i].append(id2label[label_ids[i][j]])
                preds_list[i].append(id2label[preds[i][j]])
    
    return preds_list, out_label_list

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model and return predictions and true labels
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          bbox=bbox, 
                          labels=labels)
            
            predictions.extend(outputs.logits.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

def calculate_metrics(predictions, true_labels, tokenizer):
    """
    Calculate various evaluation metrics
    """
    pred_list, true_list = align_predictions(predictions, true_labels, tokenizer)
    
    # Flatten for token-level metrics
    flat_pred = [item for sublist in pred_list for item in sublist]
    flat_true = [item for sublist in true_list for item in sublist]
    
    # Entity-level metrics using seqeval
    entity_f1 = f1_score(true_list, pred_list)
    entity_precision = precision_score(true_list, pred_list)
    entity_recall = recall_score(true_list, pred_list)
    entity_accuracy = accuracy_score(true_list, pred_list)
    
    # Token-level classification report
    token_report = classification_report(flat_true, flat_pred, 
                                       target_names=list(label2id.keys()),
                                       output_dict=True)
    
    return {
        'entity_metrics': {
            'f1': entity_f1,
            'precision': entity_precision,
            'recall': entity_recall,
            'accuracy': entity_accuracy
        },
        'token_metrics': token_report,
        'predictions': pred_list,
        'true_labels': true_list,
        'flat_predictions': flat_pred,
        'flat_true': flat_true
    }

def plot_confusion_matrix(true_labels, predictions, labels):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(metrics):
    """
    Print detailed evaluation results
    """
    print("="*60)
    print("📊 VALIDATION RESULTS")
    print("="*60)
    
    print("\n🎯 Entity-Level Metrics (seqeval):")
    print(f"  F1-Score:  {metrics['entity_metrics']['f1']:.4f}")
    print(f"  Precision: {metrics['entity_metrics']['precision']:.4f}")
    print(f"  Recall:    {metrics['entity_metrics']['recall']:.4f}")
    print(f"  Accuracy:  {metrics['entity_metrics']['accuracy']:.4f}")
    
    print("\n🔤 Token-Level Metrics:")
    token_metrics = metrics['token_metrics']
    
    print(f"  Overall Accuracy: {token_metrics['accuracy']:.4f}")
    print(f"  Macro Avg F1:     {token_metrics['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg F1:  {token_metrics['weighted avg']['f1-score']:.4f}")
    
    print("\n📝 Per-Class Performance:")
    for label_name, label_metrics in token_metrics.items():
        if label_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"  {label_name:12} | P: {label_metrics['precision']:.3f} | "
                  f"R: {label_metrics['recall']:.3f} | F1: {label_metrics['f1-score']:.3f} | "
                  f"Support: {label_metrics['support']}")

def save_results(metrics, output_file="validation_results.json"):
    """
    Save validation results to JSON file
    """
    # Remove non-serializable items for saving
    save_metrics = {
        'entity_metrics': metrics['entity_metrics'],
        'token_metrics': metrics['token_metrics']
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")

# -------------------------------
# 7️⃣ MAIN VALIDATION SCRIPT
# -------------------------------

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Load validation data
    json_folder = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\validation_data\annotations"
    images_folder = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\validation_data\images"
    
    val_files = glob.glob(f"{json_folder}/*.json")
    print(f"📂 Found {len(val_files)} validation JSON files.")
    
    if len(val_files) == 0:
        print("❌ No validation files found! Please check the path.")
        return
    
    # Create validation dataset
    val_dataset = FunsdValidationDataset(val_files, tokenizer, images_folder)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load trained model
    model_path = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\layoutlm-funsd-checkpoints\checkpoint-447"  # Path to your trained model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please make sure you have trained the model first!")
        return
    
    print(f"🔄 Loading model from {model_path}")
    model = LayoutLMForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    model.to(device)
    
    # Run evaluation
    print("🔍 Running validation...")
    predictions, true_labels = evaluate_model(model, val_dataloader, device)
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    metrics = calculate_metrics(predictions, true_labels, tokenizer)
    
    # Print results
    print_detailed_results(metrics)
    
    # Plot confusion matrix
    unique_labels = list(set(metrics['flat_true'] + metrics['flat_predictions']))
    unique_labels.sort()
    plot_confusion_matrix(metrics['flat_true'], metrics['flat_predictions'], unique_labels)
    
    # Save results
    save_results(metrics)
    
    print("\n✅ Validation completed successfully!")

if __name__ == "__main__":
    main()