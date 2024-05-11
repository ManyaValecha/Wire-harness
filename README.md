Wire Harness Color Detection
This Python project combines image processing and natural language processing (NLP) to detect wire colors based on textual descriptions. 
Features:
Image Processing:
The project loads an image and processes it to create a mask that highlights certain regions.
The mask is generated using techniques like edge detection, dilation, and erosion.
The masked image is then used for color detection.
Natural Language Processing (NLP):
The project uses the BERT model for NLP tasks.
A textual description of wire colors (e.g., “red, green, blue, and yellow”) is provided.
The BERT tokenizer processes the description.
The model predicts wire colors based on the input description.
Color Detection:
Random color predictions (for demonstration purposes) are generated.
Detected colors are printed, along with the expected color sequence.
