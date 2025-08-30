import os
import json
import torch
import re
from datetime import datetime
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import glob
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# 1️⃣ CONFIG (Same as before)
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
# 2️⃣ UTILITY FUNCTIONS (Same as before)
# -------------------------------

def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height)
    ]

def extract_text_and_boxes(image_path):
    """Extract text and bounding boxes from image using Tesseract OCR"""
    try:
        image = Image.open(image_path)
        width, height = image.size
        
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        boxes = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            confidence = float(ocr_data['conf'][i])
            
            if text and confidence > 30:
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
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

def encode_for_inference(words, boxes, tokenizer, max_length=512):
    """Encode words and boxes for inference"""
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
    encoding["word_ids"] = word_ids
    
    return encoding

def predict_entities(model, tokenizer, words, boxes, device):
    """Predict entities for given words and boxes"""
    encoding = encode_for_inference(words, boxes, tokenizer)
    
    input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"]).unsqueeze(0).to(device)
    bbox = torch.tensor(encoding["bbox"]).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_ids = predictions.argmax(-1).squeeze().tolist()
        confidence_scores = predictions.max(-1)[0].squeeze().tolist()
    
    word_id_list = encoding["word_ids"]
    
    word_predictions = []
    word_confidences = []
    processed_words = set()
    
    for i, (word_idx, pred_id, conf) in enumerate(zip(word_id_list, predicted_class_ids, confidence_scores)):
        if word_idx is not None and word_idx not in processed_words and word_idx < len(words):
            word_predictions.append(id2label[pred_id])
            word_confidences.append(conf)
            processed_words.add(word_idx)
    
    while len(word_predictions) < len(words):
        word_predictions.append("O")
        word_confidences.append(0.0)
    
    return word_predictions[:len(words)], word_confidences[:len(words)]

def group_entities(words, boxes, predictions, confidences, image_size):
    """Group consecutive tokens into entities"""
    entities = []
    current_entity = None
    
    for i, (word, box, pred, conf) in enumerate(zip(words, boxes, predictions, confidences)):
        if pred.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            
            entity_type = pred[2:]
            current_entity = {
                "type": entity_type,
                "text": word,
                "words": [word],
                "boxes": [box],
                "confidence": conf,
                "start_index": i,
                "end_index": i
            }
            
        elif pred.startswith("I-") and current_entity:
            entity_type = pred[2:]
            if entity_type == current_entity["type"]:
                current_entity["text"] += " " + word
                current_entity["words"].append(word)
                current_entity["boxes"].append(box)
                current_entity["confidence"] = (current_entity["confidence"] + conf) / 2
                current_entity["end_index"] = i
            else:
                entities.append(current_entity)
                current_entity = {
                    "type": entity_type,
                    "text": word,
                    "words": [word],
                    "boxes": [box],
                    "confidence": conf,
                    "start_index": i,
                    "end_index": i
                }
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# -------------------------------
# 3️⃣ ENHANCED INFORMATION EXTRACTION
# -------------------------------

class DocumentIntelligence:
    def __init__(self):
        self.patterns = {
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
            ],
            'amount': [
                r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|\$)\b',
                r'\b(?:total|amount|price|cost|fee):\s*\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?\b'
            ],
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
                r'\+\d{1,3}\s*\d{3,4}\s*\d{3,4}\s*\d{3,4}'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'invoice_number': [
                r'(?:invoice|inv|bill)[\s#:]*([A-Z0-9\-]+)',
                r'(?:number|no|#)[\s:]*([A-Z0-9\-]+)'
            ],
            'address': [
                r'\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln)',
                r'\b\d{5}(?:-\d{4})?\b'  # ZIP codes
            ]
        }
    
    def extract_structured_info(self, text):
        """Extract structured information from text"""
        info = {}
        
        for category, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                info[category] = list(set(matches))  # Remove duplicates
        
        return info
    
    def create_qa_pairs(self, entities):
        """Create question-answer pairs from entities"""
        qa_pairs = []
        
        questions = [entity for entity in entities if entity['type'] == 'QUESTION']
        answers = [entity for entity in entities if entity['type'] == 'ANSWER']
        headers = [entity for entity in entities if entity['type'] == 'HEADER']
        
        # Match questions with their closest answers
        for question in questions:
            closest_answer = None
            min_distance = float('inf')
            
            for answer in answers:
                # Calculate distance between question and answer
                q_end = question['end_index']
                a_start = answer['start_index']
                distance = abs(a_start - q_end)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_answer = answer
            
            if closest_answer:
                qa_pairs.append({
                    'question': question['text'],
                    'answer': closest_answer['text'],
                    'confidence': (question['confidence'] + closest_answer['confidence']) / 2,
                    'question_box': question['boxes'],
                    'answer_box': closest_answer['boxes']
                })
        
        return qa_pairs, headers

class SmartDocumentQA:
    def __init__(self, entities, structured_info, qa_pairs, headers, full_text):
        self.entities = entities
        self.structured_info = structured_info
        self.qa_pairs = qa_pairs
        self.headers = headers
        self.full_text = full_text
        
        # Create searchable content
        self.searchable_content = {
            'questions': [qa['question'] for qa in qa_pairs],
            'answers': [qa['answer'] for qa in qa_pairs],
            'headers': [h['text'] for h in headers],
            'all_text': full_text
        }
    
    def find_similar(self, query, text_list, threshold=0.3):
        """Find similar text using string matching"""
        matches = []
        query_lower = query.lower()
        
        for text in text_list:
            similarity = SequenceMatcher(None, query_lower, text.lower()).ratio()
            if similarity > threshold:
                matches.append((text, similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def answer_question(self, question):
        """Answer questions about the document"""
        question_lower = question.lower()
        
        # Check for specific information types
        if any(word in question_lower for word in ['date', 'when']):
            if 'date' in self.structured_info:
                return f"Dates found in document: {', '.join(self.structured_info['date'])}"
        
        if any(word in question_lower for word in ['amount', 'price', 'cost', 'total', 'money', 'pay']):
            if 'amount' in self.structured_info:
                return f"Amounts found: {', '.join(self.structured_info['amount'])}"
        
        if any(word in question_lower for word in ['phone', 'contact', 'number']):
            if 'phone' in self.structured_info:
                return f"Phone numbers: {', '.join(self.structured_info['phone'])}"
        
        if any(word in question_lower for word in ['email', 'mail']):
            if 'email' in self.structured_info:
                return f"Email addresses: {', '.join(self.structured_info['email'])}"
        
        if any(word in question_lower for word in ['invoice', 'bill', 'number']):
            if 'invoice_number' in self.structured_info:
                return f"Invoice numbers: {', '.join(self.structured_info['invoice_number'])}"
        
        # Search in QA pairs
        similar_questions = self.find_similar(question, self.searchable_content['questions'])
        if similar_questions:
            best_match = similar_questions[0][0]
            for qa in self.qa_pairs:
                if qa['question'] == best_match:
                    return f"Based on document: {qa['answer']}"
        
        # Search in headers
        similar_headers = self.find_similar(question, self.searchable_content['headers'])
        if similar_headers:
            return f"Related section found: {similar_headers[0][0]}"
        
        # General text search
        if any(word in self.full_text.lower() for word in question_lower.split()):
            relevant_sentences = []
            sentences = self.full_text.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_lower.split()):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return f"Relevant information: {'. '.join(relevant_sentences[:2])}"
        
        return "I couldn't find specific information about that in the document. Try asking about dates, amounts, phone numbers, emails, or invoice details."

# -------------------------------
# 4️⃣ MAIN ENHANCED PROCESSING
# -------------------------------

def process_document_with_qa(image_path, model, tokenizer, device, output_dir):
    """Process document and create QA system"""
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    # Extract text and boxes
    ocr_data = extract_text_and_boxes(image_path)
    if not ocr_data:
        return None
    
    words = ocr_data["words"]
    boxes = ocr_data["boxes"]
    image_size = ocr_data["image_size"]
    full_text = " ".join(words)
    
    print(f"📝 Found {len(words)} words")
    
    # Get entity predictions
    predictions, confidences = predict_entities(model, tokenizer, words, boxes, device)
    entities = group_entities(words, boxes, predictions, confidences, image_size)
    
    # Extract structured information
    doc_intel = DocumentIntelligence()
    structured_info = doc_intel.extract_structured_info(full_text)
    qa_pairs, headers = doc_intel.create_qa_pairs(entities)
    
    # Create QA system
    qa_system = SmartDocumentQA(entities, structured_info, qa_pairs, headers, full_text)
    
    # Print results
    print(f"✅ Extracted {len(entities)} entities")
    print(f"📊 Found {len(qa_pairs)} question-answer pairs")
    print(f"🔍 Detected structured info: {list(structured_info.keys())}")
    
    return {
        'image_path': image_path,
        'qa_system': qa_system,
        'entities': entities,
        'structured_info': structured_info,
        'qa_pairs': qa_pairs,
        'headers': headers
    }

def interactive_qa_session(results):
    """Interactive Q&A session"""
    if not results:
        print("❌ No processed documents available!")
        return
    
    print(f"\n{'='*60}")
    print(f"🤖 SMART DOCUMENT Q&A SYSTEM")
    print(f"{'='*60}")
    print(f"📄 Loaded: {os.path.basename(results['image_path'])}")
    
    # Show available information
    print(f"\n📊 Available Information:")
    for info_type, values in results['structured_info'].items():
        print(f"  • {info_type.title()}: {len(values)} found")
    
    print(f"  • Questions: {len(results['qa_pairs'])} pairs")
    print(f"  • Headers: {len(results['headers'])} found")
    
    print(f"\n💡 Try asking about:")
    print(f"  • 'What is the date?'")
    print(f"  • 'How much is the total amount?'")
    print(f"  • 'What is the phone number?'")
    print(f"  • 'Show me the invoice number'")
    print(f"  • Any specific question from the document")
    
    qa_system = results['qa_system']
    
    while True:
        try:
            question = input(f"\n❓ Ask a question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answer = qa_system.answer_question(question)
            print(f"🤖 Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

# -------------------------------
# 5️⃣ MAIN SCRIPT
# -------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Load model
    model_path = "./layoutlm-funsd-checkpoints"
    if os.path.exists("./layoutlm-funsd-checkpoints/checkpoint-447"):
        model_path = "./layoutlm-funsd-checkpoints/checkpoint-447"
    
    model = LayoutLMForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Process images
    test_images_dir = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\test_data"
    output_dir = "./enhanced_inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext.upper())))
    
    if not image_files:
        print(f"❌ No images found in {test_images_dir}")
        return
    
    print(f"📂 Found {len(image_files)} image(s)")
    
    # Process first image for demo
    if image_files:
        result = process_document_with_qa(image_files[0], model, tokenizer, device, output_dir)
        if result:
            # Start interactive session
            interactive_qa_session(result)

if __name__ == "__main__":
    main()