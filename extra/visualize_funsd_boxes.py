
#### form image with colored boxes by label


import os
import json
import cv2
import matplotlib.pyplot as plt

json_path = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\annotations\00040534.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

image_path = r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\images\00040534.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for block in data["form"]:
    box = block["box"]
    label = block["label"]
    color = (0, 255, 0)

    if label == "question":
        color = (255, 0, 0)
    elif label == "answer":
        color = (0, 0, 255)
    elif label == "header":
        color = (255, 255, 0)

    x0, y0, x1, y1 = box
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.title("FUNSD Boxes Visualization")
plt.axis("off")
plt.show()
