import cv2
import os
import pandas as pd


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
photo_name= "Sundar Pichai"
# Create a folder to store the output images
output_folder = 'output_images ' + photo_name
os.makedirs(output_folder, exist_ok=True)

# Create an empty list to store the data frames
data_frames = []

# Function to save the binary data to an Excel file
def save_data_to_excel(data_frames, output_file):
    #data= photo_name
    data = pd.concat(data_frames, ignore_index=True)
    data.to_excel(output_file, index=False)

# Iterate over the images in the 'images' folder
for filename in os.listdir('images'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join('images', filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
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

            # Display and save the extracted regions
            #cv2.imshow('Face', face)
            #cv2.imshow('Left Eye', eye_region_1)
            #cv2.imshow('Right Eye', eye_region_2)
            #cv2.imshow('Nose', nose_region)
            #cv2.imshow('Mouth', mouth_region)

            # Save the extracted regions as images
            cv2.imwrite(os.path.join(output_folder, f'{filename + photo_name}_face{i+1}.jpg'), face)
            cv2.imwrite(os.path.join(output_folder, f'{filename + photo_name}_left_eye{i+1}.jpg'), eye_region_1)
            cv2.imwrite(os.path.join(output_folder, f'{filename + photo_name}_right_eye{i+1}.jpg'), eye_region_2)
            cv2.imwrite(os.path.join(output_folder, f'{filename + photo_name}_nose{i+1}.jpg'), nose_region)
            cv2.imwrite(os.path.join(output_folder, f'{filename + photo_name}_mouth{i+1}.jpg'), mouth_region)

            # Convert the extracted regions to binary data
            face_binary = cv2.imencode('.jpg', face)[1].tobytes()
            eye1_binary = cv2.imencode('.jpg', eye_region_1)[1].tobytes()
            eye2_binary = cv2.imencode('.jpg', eye_region_2)[1].tobytes()
            nose_binary = cv2.imencode('.jpg', nose_region)[1].tobytes()
            mouth_binary = cv2.imencode('.jpg', mouth_region)[1].tobytes()

            # Create a data frame for each region and append to the list
            face_data = pd.DataFrame({'Image': [filename + photo_name], 'Binary': [face_binary]})
            eye1_data = pd.DataFrame({'Image': [filename + photo_name], 'Binary': [eye1_binary]})
            eye2_data = pd.DataFrame({'Image': [filename + photo_name], 'Binary': [eye2_binary]})
            nose_data = pd.DataFrame({'Image': [filename + photo_name], 'Binary': [nose_binary]})
            mouth_data = pd.DataFrame({'Image': [filename + photo_name], 'Binary': [mouth_binary]})

            data_frames.append(face_data)
            data_frames.append(eye1_data)
            data_frames.append(eye2_data)
            data_frames.append(nose_data)
            data_frames.append(mouth_data)

# Save the binary data to an Excel file
output_file = 'binary_data ' + photo_name +'.xlsx'
save_data_to_excel(data_frames, output_file)

# Close all windows
cv2.destroyAllWindows()

