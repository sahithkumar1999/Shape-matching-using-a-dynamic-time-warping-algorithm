import cv2
import os
import pandas as pd
from fastdtw import fastdtw
import numpy as np


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('D:\sem 2\Computational Intelligence\Project\Shape-matching-using-a-dynamic-time-warping-algorithm\Shape-matching-using-a-dynamic-time-warping-algorithm\haarcascade_frontalface_default.xml')
photo_name= "Sundar Pichai"
# Create a folder to store the output images
output_folder = 'output_images ' + photo_name
os.makedirs(output_folder, exist_ok=True)

#templates = 'D:\sem 2\Computational Intelligence\Project\Shape-matching-using-a-dynamic-time-warping-algorithm\Shape-matching-using-a-dynamic-time-warping-algorithm\output_images Sundar Pichai'

# Create an empty list to store the data frames
data_frames = []

# Function to save the binary data to an Excel file
def save_data_to_excel(data_frames, output_file):
    #data= photo_name
    data = pd.concat(data_frames, ignore_index=True)
    data.to_excel(output_file, index=False)


images= 'D:\sem 2\Computational Intelligence\Project\Shape-matching-using-a-dynamic-time-warping-algorithm\Shape-matching-using-a-dynamic-time-warping-algorithm\images'
templates= output_folder


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



# Iterate over the images in the 'templates' folder
for template_filename in os.listdir(templates):
    if template_filename.endswith('.jpg') or template_filename.endswith('.jpeg') or template_filename.endswith('.png'):
        template_image_path = os.path.join(templates, template_filename)
        template_image = cv2.imread(template_image_path)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        
        # Extract the contour of the facial component (e.g., eye, nose, mouth)
        contour, _ = cv2.findContours(template_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        template_sequence = np.squeeze(contour)  # Flatten to a 2D array
        template_sequences[template_filename] = template_sequence





# Iterate over the images in the 'images' folder
for filename in os.listdir(images):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(images, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the face region
            face = gray[y:y+h, x:x+w]

            # Define the eye regions
            eye_region_1 = gray[y:y+h//2, x:x+w//2]
            eye_region_2 = gray[y:y+h//2, x+w//2:x+w]

            # Define the nose and mouth regions
            nose_region = gray[y+h//2:y+3*h//4, x:x+w]
            mouth_region = gray[y+2*h//3:y+h, x:x+w]

            # Convert the extracted regions to sequences of contour points
            regions = {
                'face': face,
                'left_eye': eye_region_1,
                'right_eye': eye_region_2,
                'nose': nose_region,
                'mouth': mouth_region,
            }

            matching_results = []

            for region_name, region in regions.items():
                region_contour, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                region_sequence = np.squeeze(region_contour)  # Flatten to a 2D array

                # Perform shape matching with template sequences
                for template_filename, template_sequence in template_sequences.items():
                    match, distance = shape_matching(template_sequence, region_sequence, threshold=200)  # Adjust the threshold as needed
                    matching_results.append((template_filename, region_name, match, distance))

            # Sort matching results by distance
            matching_results.sort(key=lambda x: x[3])

            # Display and save the results
            for result in matching_results:
                template_filename, region_name, match, distance = result
                if match:
                    print(f"Image {filename}: {region_name} matched with {template_filename} (DTW Distance: {distance})")
                    # Save the matched image with a filename indicating the match
                    matched_image_filename = f'{filename}_{photo_name}_{region_name}_matched.jpg'
                    cv2.imwrite(os.path.join(output_folder, matched_image_filename), image)

# Save the binary data to an Excel file
output_file = 'binary_data ' + photo_name + '.xlsx'
save_data_to_excel(data_frames, output_file)

# Close all windows
cv2.destroyAllWindows()