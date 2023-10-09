import cv2
import os
import pandas as pd
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import PhotoImage
from tkinter import filedialog
from PIL import Image, ImageTk
import shutil


# Create the main application window
app = tk.Tk()
app.title("Shape Matching Application")
# Get the screen width and height
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
# Set the window size to fit the screen
app.geometry(f"{screen_width}x{screen_height}")

# Create a Text widget for displaying text
text_widget = tk.Text(app, wrap=tk.WORD)
text_widget.pack(fill=tk.BOTH, expand=True)

# Insert text into the Text widget
text_widget.insert(tk.END, "Shape Matching with Dynamic Time Warping\n", ("main_heading",))

text_widget.insert(tk.END, "Project Overview:\n", ("sub_heading",))
text_widget.insert(tk.END, "The Dynamic Time Warping-Based Shape Matching for Computer Vision project is a significant exploration into the realm of computer vision and pattern recognition. This research focuses on the development of an intelligent system capable of identifying similarities and correspondences between two shapes within digital images. Shape matching is a fundamental task with widespread applications, including image recognition, object tracking, and gesture recognition. To address this challenge, the project leverages the Dynamic Time Warping (DTW) algorithm, a powerful technique that enables precise matching even when shapes are subject to variations in speed or scale.\n\n")

text_widget.insert(tk.END, "Key Contributions:\n", ("sub_heading",))
text_widget.insert(tk.END, "This project contributes to the academic community by demonstrating the practical application of DTW in real-world scenarios. It showcases the importance of shape matching in computer vision and highlights the versatility of DTW in handling complex shape variations. By providing a user-friendly graphical interface, the project bridges the gap between sophisticated algorithms and end-users, making shape matching accessible to a broader audience. Additionally, the system's ability to generate visual representations of matching results and store data in Excel files enhances its utility for research and analysis in computer vision studies.\n\n")

text_widget.insert(tk.END, "Significance:\n", ("sub_heading",))
text_widget.insert(tk.END, "The project's significance lies in its potential to advance various domains within computer vision. Researchers, practitioners, and students can benefit from the project's insights into shape matching methodologies and its user-friendly interface for experimentation. Moreover, the academic community can explore and extend the project's capabilities, fostering innovation in computer vision applications. Ultimately, the project contributes to the broader goal of harnessing advanced algorithms like DTW for practical, real-world solutions in the field of computer vision")


# Configure text tags for formatting
text_widget.tag_configure("main_heading", font=("Helvetica", 20, "bold"))
text_widget.tag_configure("sub_heading", font=("Helvetica", 16, "underline"))

# Create a label for displaying the matched image with a placeholder text
image_label = tk.Label(app, text="press the below Upload Image button to upload the input image")
image_label.pack()

# Create an "Upload Image" button
upload_button = tk.Button(app, text="Upload Image")
upload_button.pack()

# Create a folder to store the output images
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# Create a folder to store the output graphs
output_graph_folder = 'output_Graph'
os.makedirs(output_graph_folder, exist_ok=True)


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
photo_name = "Sundar Pichai"

# Create an empty list to store the data frames
data_frames = []

# Function to perform DTW shape matching and visualize the result
def shape_matching_with_visualization(template_sequence, target_sequence, threshold):
    # Compute the DTW distance and the optimal path
    distance, path = fastdtw(template_sequence, target_sequence)

    if distance < threshold:
        return True, distance, path
    else:
        return False, distance, path

# Function to save the binary data to an Excel file
def save_data_to_excel(data_frames, output_file):
    data = pd.concat(data_frames, ignore_index=True)
    data.to_excel(output_file, index=False)

# Function to perform DTW shape matching
def shape_matching(template_sequence, target_sequence, threshold):
    # Compute the DTW distance between the sequences
    distance, _ = fastdtw(template_sequence, target_sequence)

    if distance < threshold:
        return True, distance
    else:
        return False, distance

# Load and preprocess the template sequences for shape matching
template_sequences = {}

# Iterate over the images in the 'images' folder
for filename in os.listdir('images'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join('images', filename)
        template_image = cv2.imread(image_path)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Extract the contour of the template image
        contour, _ = cv2.findContours(template_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        template_sequence = np.squeeze(contour)
        template_sequences[filename] = template_sequence


# Function to handle shape matching
def match_shape(input_image_path):
    matching_results_text.delete(1.0, tk.END)  # Clear previous matching results

    # Load the given image for shape matching
    given_image = cv2.imread(input_image_path)
    given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

    matching_results = []

    # Extract contour from the given grayscale image
    given_contour, _ = cv2.findContours(given_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    given_sequence = np.squeeze(given_contour)

    # Perform shape matching with template sequences
    for template_filename, template_sequence in template_sequences.items():
        match, distance, path = shape_matching_with_visualization(template_sequence, given_sequence, threshold=200)

        if match:
            #print(f"Given image matched with {template_filename} (DTW Distance: {distance})")

            # Find peak positions
            max_template_position = np.argmax(template_sequence, axis=0)
            max_given_position = np.argmax(given_sequence, axis=0)

            #print("Peak position in the template sequence:", max_template_position)
            #print("Peak position in the given sequence:", max_given_position)

            # Extract the x and y coordinates of the matched points
            matched_x = [template_sequence[i, 0] for i, _ in path]
            matched_y = [template_sequence[i, 1] for i, _ in path]

            # Plot the sequences and highlight the matched points
            plt.figure(figsize=(8, 4))
            plt.plot(template_sequence[:, 0], template_sequence[:, 1], label='Template Sequence', marker='o', markersize=5)
            plt.plot(given_sequence[:, 0], given_sequence[:, 1], label='Given Sequence', marker='x', markersize=5)
            plt.scatter(matched_x, matched_y, c='r', label='Matched Points')
            plt.legend()
            plt.title(f"Matching Result with {template_filename} (DTW Distance: {distance})")
            plt.xlabel("X")
            plt.ylabel("Y")


            matching_results_text.insert(tk.END, f"Peak position in the template sequence: {max_template_position}\n")
            matching_results_text.insert(tk.END, f"Peak position in the given sequence: {max_given_position}\n")

            # Save the graph image in the "output_Graph" folder
            graph_image_path = os.path.join(output_graph_folder, f'graph.png')
            plt.savefig(graph_image_path, bbox_inches='tight')  # Save the graph image
            plt.close()  # Close the plot to release resources

            # Display the graph image in the GUI
            graph = Image.open(graph_image_path)
            graph.thumbnail((400, 400))
            graph_photo = ImageTk.PhotoImage(graph)
            graph_label.config(image=graph_photo)
            graph_label.image = graph_photo  # Keep a reference to prevent it from being garbage collected

        matching_results.append((template_filename, match, distance))

    # Sort matching results by distance
    matching_results.sort(key=lambda x: x[2])

    # Save the matching results to an Excel file
    output_file = 'matching_results ' + photo_name + '.xlsx'
    matching_data = pd.DataFrame({'Template': [result[0] for result in matching_results], 'DTW Distance': [result[2] for result in matching_results]})
    save_data_to_excel([matching_data], output_file)

    # Display matching results in the GUI
    matching_results_text.insert(tk.END, "Matching Results:\n")
    for result in matching_results:
        if result[1]:
            matching_results_text.insert(tk.END, f"Given image matched with {result[0]} (DTW Distance: {result[2]})\n")

            

            # Load and display the matched image
            matched_image_path = os.path.join(output_graph_folder, f'graph.png')
            matched_image = Image.open(matched_image_path)
            matched_image.thumbnail((400, 400))  # Resize the image if needed
            matched_image = ImageTk.PhotoImage(matched_image)
            image_label.config(image=matched_image, text="")  # Update the image label with the matched image
            image_label.image = matched_image  # Keep a reference to prevent image from being garbage collected

        #else:
            #matching_results_text.insert(tk.END, f"No match found with {result[0]} (DTW Distance: {result[2]})\n")
    matching_results_text.insert(tk.END, "\nResults saved to Excel.")

# Function to handle image upload
def upload_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if file_path:
        # Define the destination path and file name
        destination_path = 'output_Graph'
        destination_file = 'graph.jpg'

        # Copy the selected image to the destination
        shutil.copy(file_path, os.path.join(destination_path, destination_file))

        # Display a confirmation message in the text widget
        confirmation_text = f"Image '{destination_file}' uploaded successfully to '{destination_path}'."
        text_widget.insert(tk.END, "\n\n" + confirmation_text)
        text_widget.see(tk.END)

        # Perform shape matching with the uploaded image
        match_shape(os.path.join(destination_path, destination_file))

# Create an "Upload Image" button
upload_button.config(command=upload_image)

# Create a label for displaying the matching results
matching_results_text = tk.Text(app, wrap=tk.WORD)
matching_results_text.pack(fill=tk.BOTH, expand=True)

# Create a label for displaying the graph image
graph_label = tk.Label(app, text="Matching Graph")
graph_label.pack()

# Start the main tkinter event loop
app.mainloop()
