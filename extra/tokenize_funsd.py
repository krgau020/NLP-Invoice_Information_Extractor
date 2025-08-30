# 

import os
import json
from transformers import LayoutLMTokenizerFast

# 📌 Load LayoutLM tokenizer
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

# ✅ Define BIO label list
labels = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("Label2ID:", label2id)

# ✅ Load JSON
json_path = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\annotations\00040534.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

# ✅ FUNSD images are 1000x1000 → normalize boxes to 0–1000
def normalize_bbox(box):
    width, height = 1000, 1000
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

# ✅ Build word list, box list, BIO labels
words = []
boxes = []
labels_list = []

for block in data["form"]:
    block_label = block["label"].upper()
    if block_label == "OTHER":
        continue  # Skip 'other'

    word_texts = [w["text"] for w in block["words"]]
    block_box = normalize_bbox(block["box"])
    word_boxes = [block_box] * len(word_texts)

    bio_tags = []
    for i in range(len(word_texts)):
        bio_tag = f"B-{block_label}" if i == 0 else f"I-{block_label}"
        bio_tags.append(bio_tag)

    words.extend(word_texts)
    boxes.extend(word_boxes)
    labels_list.extend(bio_tags)

print("\nSample words:", words[:10])
print("Sample boxes:", boxes[:10])
print("Sample BIO labels:", labels_list[:10])

# ✅ Tokenize with alignment
encoding = tokenizer(
    words,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding="max_length",
    truncation=True,
    max_length=512,
)

word_ids = encoding.word_ids()

# ✅ Map word IDs → BIO tag IDs → boxes
token_labels = []
token_boxes = []

for word_id in word_ids:
    if word_id is None:
        token_labels.append(-100)  # Special tokens, ignore in loss
        token_boxes.append([0, 0, 0, 0])
    else:
        token_labels.append(label2id[labels_list[word_id]])
        token_boxes.append(boxes[word_id])

# ✅ Add boxes back to encoding
encoding["bbox"] = token_boxes
encoding["labels"] = token_labels

# ✅ Check output shapes
print("\nEncoding keys:", encoding.keys())
print("Input IDs:", encoding["input_ids"][:20])
print("BBoxes:", encoding["bbox"][:5])
print("Labels:", encoding["labels"][:20])


















'''✅ This script:
Loads the JSON

Normalizes bounding boxes

Creates BIO labels for each word

Tokenizes words → keeps word-to-token alignment

Expands word-level boxes/labels to token-level

Adds bbox + labels to the encoding → exactly what LayoutLM expects

📌 ✅ Next, you have:
input_ids → the text tokens

attention_mask → from tokenizer

bbox → normalized boxes

labels → token labels

'''




















'''📌 What tokenization is used here?
We’re using: LayoutLMTokenizerFast → which is basically a BertTokenizerFast.

So the type is:
Subword tokenization → specifically WordPiece.

📂 Details
🔑 1️⃣ WordPiece tokenizer
Used by BERT, LayoutLM, RoBERTa (with slight variations like BPE for RoBERTa).

Splits words into subword units.

Example:


"unhappiness"
→ ["un", "##happiness"]
Here, ## shows it’s a continuation of a word.



🔑 2️⃣ Why subword tokenization?

Handles unknown or rare words.

Handles typos or morphology.

Keeps vocab size small (30k–50k subword units instead of millions of whole words).



✅ How this affects alignment
Your JSON words are whole words.
But the model processes subword tokens.
So you must:

Know which tokens come from which word.

Expand one word’s label/box to all its tokens.

📎 Example
Block:


{"words": [{"text": "Internationalization"}]}
Tokenized:


["international", "##ization"]
So:

Original word: Internationalization

Label: B-HEADER

BBox: [x0, y0, x1, y1]

👉 Both international and ##ization get:

Label: B-HEADER for the first, I-HEADER for the rest (or same if word is single unit)

BBox: same box



✅ How the code handles it

encoding = tokenizer(
    words,
    is_split_into_words=True,
    return_offsets_mapping=True,
)
word_ids = encoding.word_ids()
word_ids → map each token back to its word index.

You loop over word_ids → assign correct label + box for each token.



⚡ Key takeaway
Tokenizer type: WordPiece
Model type: BERT-style (LayoutLM)
Alignment logic: Expand word-level gold labels + boxes to each token produced by the wordpiece split.



✔️ That’s why you see:
is_split_into_words=True → disables normal sentence tokenization.

word_ids → built-in helper for fast token → word mapping.'''























'''✅ Key concepts shown here
Part	       Meaning
Label2ID	   Maps text labels → numeric class IDs
Offset mapping   	Connects token back to character span
Bounding box (bbox)	  Tells where on the page each token appears
BIO labels   	For segmenting spans (headers, questions, answers)
Token IDs   	Input for the transformer model
Attention mask	   Ignore padding tokens during computation

'''