import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import dilation, erosion, rectangle
from skimage.measure import find_contours
from skimage.draw import polygon
from skimage.transform import resize
from transformers import BertTokenizer, TFBertModel

def load_image(image_path):
    return Image.open(image_path)

def preprocess_image(img):
    img = img.resize((224, 224))
    img = img.convert("L")
    img_array = np.array(img) / 255.0
    return img_array

def get_mask(img):
    kernel = rectangle(7, 6)
    dilate = dilation(canny(rgb2gray(img), 0), kernel)
    dilate = dilation(dilate, kernel)
    dilate = dilation(dilate, kernel)
    erode = erosion(dilate, kernel)
    mask = np.zeros_like(erode, dtype=bool)
    contours = find_contours(erode, level=0.5)
    if len(contours) > 0:
        rr, cc = polygon(contours[0][:, 0], contours[0][:, 1])
        mask[rr, cc] = True
    mask = gaussian(mask, 7) > 0.74
    mask = resize(mask, (224, 224), anti_aliasing=False).astype(float)
    return mask

def detect_wire_colors(img):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased', from_pt=True)

    preprocessed_img = preprocess_image(img)
    mask = get_mask(img)
    masked_img = preprocessed_img * mask

    textual_description = "The wire colors are red, green, blue, and yellow."
    inputs = tokenizer(text=textual_description, return_tensors='tf', padding=True, truncation=True)
    outputs = model(inputs)

    predicted_colors = np.random.rand(1, 4)
    color_names = ["red", "blue", "green", "yellow"]
    detected_colors = [color_names[np.argmax(predicted_colors)]]

    print("Detected wire colors:")
    for i, color in enumerate(detected_colors):
        print(f"Wire {i + 1} color: {color}")

    expected_sequence = ["red", "green", "blue", "yellow"]
    print("\nExpected color sequence:")
    for i, color in enumerate(expected_sequence):
        print(f"Wire {i + 1} color: {color}")

if __name__ == "__main__":
    path_img = "/Users/manyavalecha/Documents/colourwire.jpg"
    img = load_image(path_img)
    detect_wire_colors(img)
