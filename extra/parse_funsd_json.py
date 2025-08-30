### reading json data (annotation)

import os
import json

# Pick a file
json_path = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\annotations\00040534.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total blocks: {len(data['form'])}")
print("-" * 50)

for block in data["form"]:
    label = block["label"]
    box = block["box"]
    words = [w["text"] for w in block["words"]]

    text = " ".join(words)
    print(f"Label: {label}")
    print(f"Box: {box}")
    print(f"Text: {text}")
    print("-" * 50)
