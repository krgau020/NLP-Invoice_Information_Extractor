## testing OCR --> image to text

from PIL import Image
import pytesseract

# If not added to PATH, point directly:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test image (make your own or download a small test.png with text)
image = Image.open(r"C:\Users\admin\Desktop\NLP\Information-Extraction-System\ocr_test_image.png")

text = pytesseract.image_to_string(image)

print("OCR Result:")
print(text)
