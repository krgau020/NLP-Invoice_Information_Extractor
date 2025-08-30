import os
import json
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Set Tesseract path (add this at the top)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

# Color mapping for visualization
LABEL_COLORS = {
    "O": "#CCCCCC",          # Gray
    "B-QUESTION": "#FF6B6B", # Red
    "I-QUESTION": "#FF8E8E", # Light Red
    "B-ANSWER": "#4ECDC4",   # Teal
    "I-ANSWER": "#7ED7D1",   # Light Teal
    "B-HEADER": "#45B7D1",   # Blue
    "I-HEADER": "#6BC5E0"    # Light Blue
}

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

def unnormalize_box(normalized_box, width, height):
    """Convert normalized box back to original coordinates"""
    return [
        int(normalized_box[0] * width / 1000),
        int(normalized_box[1] * height / 1000),
        int(normalized_box[2] * width / 1000),
        int(normalized_box[3] * height / 1000)
    ]

# -------------------------------
# 3️⃣ OCR FUNCTIONS
# -------------------------------

def extract_text_and_boxes(image_path):
    """
    Extract text and bounding boxes from image using Tesseract OCR
    """
    try:
        image = Image.open(image_path)
        width, height = image.size
        
        # Use Tesseract to get detailed information
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        boxes = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            confidence = float(ocr_data['conf'][i])
            
            # Filter out empty texts and low confidence detections
            if text and confidence > 30:  # Confidence threshold
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Create bounding box [x0, y0, x1, y1]
                box = [x, y, x + w, y + h]
                normalized_box = normalize_box(box, width, height)
                
                words.append(text)
                boxes.append(normalized_box)
        
        return {
            "words": words,
            "boxes": boxes,
            "image_size": (width, height)
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

# -------------------------------
# 4️⃣ TOKENIZE & ENCODE FOR INFERENCE
# -------------------------------

def encode_for_inference(words, boxes, tokenizer, max_length=512):
    """
    Encode words and boxes for inference (no labels)
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    aligned_boxes = []
    word_ids = encoding.word_ids()
    
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            if word_idx < len(boxes):
                aligned_boxes.append(boxes[word_idx])
            else:
                aligned_boxes.append([0, 0, 0, 0])
    
    encoding.pop("offset_mapping")
    encoding["bbox"] = aligned_boxes
    encoding["word_ids"] = word_ids  # Store word_ids in encoding
    
    return encoding

# -------------------------------
# 5️⃣ INFERENCE FUNCTIONS
# -------------------------------

def predict_entities(model, tokenizer, words, boxes, device):
    """
    Predict entities for given words and boxes
    """
    # Encode the input
    encoding = encode_for_inference(words, boxes, tokenizer)
    
    # Convert to tensors
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"]).unsqueeze(0).to(device)
    bbox = torch.tensor(encoding["bbox"]).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_ids = predictions.argmax(-1).squeeze().tolist()
        confidence_scores = predictions.max(-1)[0].squeeze().tolist()
    
    # Get word_ids from encoding
    word_id_list = encoding["word_ids"]
    
    # Extract word-level predictions
    word_predictions = []
    word_confidences = []
    
    processed_words = set()
    
    for i, (word_idx, pred_id, conf) in enumerate(zip(word_id_list, predicted_class_ids, confidence_scores)):
        if word_idx is not None and word_idx not in processed_words and word_idx < len(words):
            word_predictions.append(id2label[pred_id])
            word_confidences.append(conf)
            processed_words.add(word_idx)
    
    # Ensure we have predictions for all words
    while len(word_predictions) < len(words):
        word_predictions.append("O")
        word_confidences.append(0.0)
    
    return word_predictions[:len(words)], word_confidences[:len(words)]

def group_entities(words, boxes, predictions, confidences, image_size):
    """
    Group consecutive tokens into entities
    """
    entities = []
    current_entity = None
    
    for i, (word, box, pred, conf) in enumerate(zip(words, boxes, predictions, confidences)):
        if pred.startswith("B-"):  # Beginning of entity
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            
            # Start new entity
            entity_type = pred[2:]  # Remove "B-"
            current_entity = {
                "type": entity_type,
                "text": word,
                "words": [word],
                "boxes": [unnormalize_box(box, image_size[0], image_size[1])],
                "normalized_boxes": [box],
                "confidence": conf,
                "start_index": i,
                "end_index": i
            }
            
        elif pred.startswith("I-") and current_entity:  # Inside entity
            entity_type = pred[2:]  # Remove "I-"
            if entity_type == current_entity["type"]:
                current_entity["text"] += " " + word
                current_entity["words"].append(word)
                current_entity["boxes"].append(unnormalize_box(box, image_size[0], image_size[1]))
                current_entity["normalized_boxes"].append(box)
                current_entity["confidence"] = (current_entity["confidence"] + conf) / 2  # Average confidence
                current_entity["end_index"] = i
            else:
                # Inconsistent entity type, save current and start new
                entities.append(current_entity)
                current_entity = {
                    "type": entity_type,
                    "text": word,
                    "words": [word],
                    "boxes": [unnormalize_box(box, image_size[0], image_size[1])],
                    "normalized_boxes": [box],
                    "confidence": conf,
                    "start_index": i,
                    "end_index": i
                }
        else:
            # "O" label or inconsistent sequence
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget the last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities

# -------------------------------
# 6️⃣ VISUALIZATION FUNCTIONS
# -------------------------------

def visualize_predictions(image_path, words, boxes, predictions, image_size, save_path=None):
    """
    Visualize predictions on the original image
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a better font
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for word, box, pred in zip(words, boxes, predictions):
        if pred != "O":
            # Convert normalized box back to image coordinates
            actual_box = unnormalize_box(box, image_size[0], image_size[1])
            color = LABEL_COLORS.get(pred, "#000000")
            
            # Draw bounding box
            draw.rectangle(actual_box, outline=color, width=2)
            
            # Draw label
            label_y = max(0, actual_box[1] - 15)
            draw.text((actual_box[0], label_y), pred, fill=color, font=font)
    
    if save_path:
        image.save(save_path)
        print(f"📸 Visualization saved to {save_path}")
    
    return image

def create_detailed_visualization(image_path, entities, save_path=None):
    """
    Create a detailed visualization with matplotlib
    """
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    for entity in entities:
        # Get the bounding box that covers all words in the entity
        all_boxes = entity["boxes"]
        if all_boxes:
            min_x = min(box[0] for box in all_boxes)
            min_y = min(box[1] for box in all_boxes)
            max_x = max(box[2] for box in all_boxes)
            max_y = max(box[3] for box in all_boxes)
            
            color = LABEL_COLORS.get(f"B-{entity['type']}", "#000000")
            
            # Draw rectangle
            rect = patches.Rectangle(
                (min_x, min_y), max_x - min_x, max_y - min_y,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(min_x, min_y - 5, f"{entity['type']}: {entity['text'][:30]}...",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                   fontsize=8, color='white', weight='bold')
    
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis('off')
    plt.title(f"Entity Extraction Results - {os.path.basename(image_path)}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed visualization saved to {save_path}")
    
    plt.show()

# -------------------------------
# 7️⃣ RESULT PROCESSING
# -------------------------------

def save_results(image_path, words, boxes, predictions, confidences, entities, output_dir):
    """
    Save inference results to JSON file
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    results = {
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(),
        "word_level_predictions": [
            {
                "word": word,
                "box": box,
                "prediction": pred,
                "confidence": conf
            }
            for word, box, pred, conf in zip(words, boxes, predictions, confidences)
        ],
        "entities": entities,
        "summary": {
            "total_words": len(words),
            "total_entities": len(entities),
            "entity_counts": {}
        }
    }
    
    # Count entities by type
    for entity in entities:
        entity_type = entity["type"]
        results["summary"]["entity_counts"][entity_type] = results["summary"]["entity_counts"].get(entity_type, 0) + 1
    
    # Save to file
    output_file = os.path.join(output_dir, f"{base_name}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_file

def print_results(image_path, entities):
    """
    Print extraction results in a readable format
    """
    print(f"\n{'='*60}")
    print(f"🖼️  RESULTS FOR: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if not entities:
        print("❌ No entities found!")
        return
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        entity_type = entity["type"]
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)
    
    for entity_type, group_entities in entity_groups.items():
        print(f"\n📝 {entity_type.upper()}S ({len(group_entities)} found):")
        print("-" * 40)
        for i, entity in enumerate(group_entities, 1):
            print(f"  {i}. {entity['text']}")
            print(f"     Confidence: {entity['confidence']:.3f}")

# -------------------------------
# 8️⃣ MAIN INFERENCE FUNCTION
# -------------------------------

def process_single_image(image_path, model, tokenizer, device, output_dir):
    """
    Process a single image and extract entities
    """
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    # Extract text and boxes using OCR
    ocr_data = extract_text_and_boxes(image_path)
    if not ocr_data:
        print(f"❌ Failed to extract text from {image_path}")
        return None
    
    words = ocr_data["words"]
    boxes = ocr_data["boxes"]
    image_size = ocr_data["image_size"]
    
    if not words:
        print(f"⚠️  No text found in {image_path}")
        return None
    
    print(f"📝 Found {len(words)} words")
    
    # Get predictions
    predictions, confidences = predict_entities(model, tokenizer, words, boxes, device)
    
    # Group into entities
    entities = group_entities(words, boxes, predictions, confidences, image_size)
    
    # Print results
    print_results(image_path, entities)
    
    # Save results
    results_file = save_results(image_path, words, boxes, predictions, confidences, entities, output_dir)
    print(f"💾 Results saved to: {results_file}")
    
    # Create visualizations
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Simple visualization
    simple_viz_path = os.path.join(output_dir, f"{base_name}_simple_viz.png")
    visualize_predictions(image_path, words, boxes, predictions, image_size, simple_viz_path)
    
    # Detailed visualization
    detailed_viz_path = os.path.join(output_dir, f"{base_name}_detailed_viz.png")
    create_detailed_visualization(image_path, entities, detailed_viz_path)
    
    return {
        "image_path": image_path,
        "words": words,
        "boxes": boxes,
        "predictions": predictions,
        "confidences": confidences,
        "entities": entities,
        "results_file": results_file
    }

# -------------------------------
# 9️⃣ MAIN SCRIPT
# -------------------------------

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Load trained model
    model_path = "./layoutlm-funsd-checkpoints"
    
    # Check for checkpoint subdirectory
    if os.path.exists("./layoutlm-funsd-checkpoints/checkpoint-447"):
        model_path = "./layoutlm-funsd-checkpoints/checkpoint-447"
    elif not os.path.exists(model_path):
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
    
    # Set up paths
    test_images_dir = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\test_data"
    output_dir = "./inference_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext.upper())))
    
    if not image_files:
        print(f"❌ No image files found in {test_images_dir}")
        print("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return
    
    print(f"📂 Found {len(image_files)} image(s) to process")
    
    # Process all images
    all_results = []
    for image_path in image_files:
        try:
            result = process_single_image(image_path, model, tokenizer, device, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
    
    # Create summary
    if all_results:
        summary = {
            "total_images_processed": len(all_results),
            "timestamp": datetime.now().isoformat(),
            "overall_statistics": {
                "total_entities": sum(len(r["entities"]) for r in all_results),
                "total_words": sum(len(r["words"]) for r in all_results)
            }
        }
        
        summary_file = os.path.join(output_dir, "inference_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ INFERENCE COMPLETED!")
        print(f"{'='*60}")
        print(f"📊 Total images processed: {len(all_results)}")
        print(f"📝 Total entities extracted: {summary['overall_statistics']['total_entities']}")
        print(f"💾 Results saved in: {output_dir}")
        print(f"📋 Summary saved to: {summary_file}")
    
    else:
        print("❌ No images were successfully processed!")

if __name__ == "__main__":
    # Install required packages message
    print("📦 Make sure you have installed required packages:")
    print("pip install pytesseract pillow matplotlib")
    print("And install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
    print("\n" + "="*60)
    
    main()