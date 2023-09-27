import cv2
import os
import pandas as pd
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('D:\sem 2\Computational Intelligence\Project\DTW\haarcascade_frontalface_default.xml')
photo_name = "Sundar Pichai"

# Create a folder to store the output images
output_folder = 'output_images ' + photo_name
os.makedirs(output_folder, exist_ok=True)

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
template_sequences = {}  # Store template sequences here

# Iterate over the images in the 'images' folder
for filename in os.listdir('images'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join('images', filename)
        template_image = cv2.imread(image_path)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Extract the contour of the template image
        contour, _ = cv2.findContours(template_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        template_sequence = np.squeeze(contour)  # Flatten to a 2D array
        template_sequences[filename] = template_sequence

# Load the given image for shape matching
given_image_path = 'D:\sem 2\Computational Intelligence\Project\DTW\download.jpeg'  # Replace with the path to your given image
given_image = cv2.imread(given_image_path)
given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

matching_results = []

# Extract contour from the given grayscale image
given_contour, _ = cv2.findContours(given_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
given_sequence = np.squeeze(given_contour)

# Perform shape matching with template sequences
for template_filename, template_sequence in template_sequences.items():
    match, distance, path = shape_matching_with_visualization(template_sequence, given_sequence, threshold=200)

    if match:
        print(f"Given image matched with {template_filename} (DTW Distance: {distance})")

        # Plot the sequences and highlight the matched points
        plt.figure(figsize=(8, 4))
        plt.plot(template_sequence[:, 0], template_sequence[:, 1], label='Template Sequence', marker='o', markersize=5)
        plt.plot(given_sequence[:, 0], given_sequence[:, 1], label='Given Sequence', marker='x', markersize=5)
        for (i, j) in path:
            plt.plot([template_sequence[i, 0], given_sequence[j, 0]], [template_sequence[i, 1], given_sequence[j, 1]], 'k--')
        plt.legend()
        plt.title(f"Matching Result with {template_filename} (DTW Distance: {distance})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    matching_results.append((template_filename, match, distance))


# Sort matching results by distance
matching_results.sort(key=lambda x: x[2])

# Display and save the matched images along with matching values
for result in matching_results:
    template_filename, match, distance = result
    if match:
        print(f"Given image matched with {template_filename} (DTW Distance: {distance})")
        # Save the matched image with a filename indicating the match
        matched_image_filename = f'{given_image_path}_matched_with_{template_filename}.jpg'
        cv2.imwrite(os.path.join(output_folder, matched_image_filename), given_image)

# Save the matching results to an Excel file
output_file = 'matching_results ' + photo_name + '.xlsx'
matching_data = pd.DataFrame({'Template': [result[0] for result in matching_results], 'DTW Distance': [result[2] for result in matching_results]})
save_data_to_excel([matching_data], output_file)

print("Matching results saved to Excel.")
