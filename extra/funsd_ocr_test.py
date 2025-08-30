## testing ocr on funsd data using tesseract

from PIL import Image
import pytesseract

# If not added to PATH, point directly:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test image (make your own or download a small test.png with text)
image = Image.open(r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\dataset\training_data\images\00040534.png")



#Optional: convert to grayscale
image = image.convert("L")

# Run OCR
text = pytesseract.image_to_string(image)

print("FUNSD OCR Result:")
print("-" * 40)
print(text)

