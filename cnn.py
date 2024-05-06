import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_image(image):
    # Convert PhotoImage to PIL Image
    pil_image = Image.open(image)
    # Convert to grayscale
    pil_image = pil_image.convert('L')
    # Resize to 28x28 pixels
    pil_image = pil_image.resize((28, 28))
    # Convert image to array and scale pixel values
    image_array = np.array(pil_image) / 255.0
    # Reshape to match model input shape
    image_array = image_array.reshape(1, 28, 28)
    return image_array

def predict_digit(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Make prediction
    predictions = model.predict(preprocessed_image)
    # Get the predicted digit
    predicted_digit = np.argmax(predictions)
    return predicted_digit

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Predict the digit
        predicted_digit = predict_digit(file_path)
        result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Tkinter GUI
root = tk.Tk()
root.title("Handwritten Digit Recognition")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

open_button = tk.Button(frame, text="Open Image", command=open_image)
open_button.pack(side=tk.LEFT)

result_label = tk.Label(frame, text="Predicted Digit: ")
result_label.pack(side=tk.LEFT, padx=10)

root.mainloop()
