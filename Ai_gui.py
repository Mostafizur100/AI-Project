import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('poster_model.h5')

# Define class labels
classes = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
           'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance',
           'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

# Create the main application window
app = tk.Tk()
app.title("Movie Genre Classification")

# Create a frame for image input on the left
image_frame = tk.Frame(app)
image_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label for displaying the selected image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a frame for displaying predicted labels on the right
labels_frame = tk.Frame(app)
labels_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create a label for displaying the predicted labels
output_label = tk.Label(labels_frame, text="Predicted Labels:")
output_label.pack()

def load_and_predict(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((150, 150))  # Resize the image to match the model input size
        image = np.array(image)
        image = image / 255.0  # Normalize the pixel values

        # Make predictions
        proba = model.predict(image.reshape(1, 150, 150, 3))
        top_3 = np.argsort(proba[0])[:-4:-1]

        # Display the image
        photo = ImageTk.PhotoImage(Image.open(file_path))
        image_label.config(image=photo)
        image_label.image = photo

        # Display the predicted labels
        predictions_text = "Predicted Labels:\n"
        for i in range(3):
            predictions_text += "{} ({:.3f})\n".format(classes[top_3[i]], proba[0][top_3[i]])

        output_label.config(text=predictions_text)
    except Exception as e:
        messagebox.showerror("Error", "An error occurred while processing the image.")

# Create a button to select an image
def select_image():
    file_path = filedialog.askopenfilename()
    load_and_predict(file_path)

browse_button = tk.Button(image_frame, text="Browse", command=select_image)
browse_button.pack()

# Start the GUI application
app.mainloop()
