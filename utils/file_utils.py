import base64
from PIL import Image
from io import BytesIO
import pytesseract


def extract_text_from_image(image_base64: str):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    text = pytesseract.image_to_string(image)
    return text

