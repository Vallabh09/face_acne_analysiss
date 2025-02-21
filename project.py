import cv2
import tensorflow as tf
import numpy as np
from ultralytics import YOLO


classification_model = tf.keras.models.load_model('acne_classifier.h5')  
yolo_model = YOLO('best.pt') # Load pre-trained models

# Function to capture an image
def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Press 's' to Save", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('captured_image.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg' 

# Function to classify image (Acne/No Acne)
def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize for the model
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    prediction = classification_model.predict(image)
    return 'Acne' if prediction[0][0] > 0.5 else 'No Acne'

# Function to detect acne using YOLO
def detect_acne(image_path):
    results = yolo_model(image_path)
    detected_image = results[0].plot()  # Visualize detections
    cv2.imshow('YOLO Detection', detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return results[0].boxes.xyxy.numpy()  # Return bounding box coordinates

# Function to segment face regions manually
def segment_face(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    regions = {
        "forehead": [(w//3, h//8), (2*w//3, h//4)],  # Approximate forehead region
        "left_cheek": [(w//8, h//3), (w//3, 2*h//3)],  # Approximate left cheek region
        "right_cheek": [(2*w//3, h//3), (7*w//8, 2*h//3)],  # Approximate right cheek region
        "chin": [(w//3, 3*h//4), (2*w//3, 7*h//8)]  # Approximate chin region
    }
    return regions

# Function to calculate acne density by region
def calculate_acne_density(acne_positions, regions):
    density = {region: 0 for region in regions}
    for box in acne_positions:
        x, y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        for region, (top_left, bottom_right) in regions.items():
            if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                density[region] += 1
    return density

# Function to find the region with the highest acne concentration
def find_highest_concentration_region(density):
    max_region = max(density, key=density.get)  # Find the region with the maximum count
    max_count = density[max_region]
    return max_region, max_count

# Function to display insights based on acne density
def display_insights(density):
    insights = {
        "forehead": "Stomach or small intestine issues",
        "left_cheek": "Respiratory problems",
        "right_cheek": "Liver issues",
        "chin": "Hormonal imbalance or pelvic organ issues"
    }
    for region, count in density.items():
        if count > 0:
            print(f"{region.capitalize()}: {insights[region]} ({count} spots detected)")

    # Highlight the region with the highest acne concentration
    if any(count > 0 for count in density.values()):
        max_region, max_count = find_highest_concentration_region(density)
        print(f"\nRegion with highest acne concentration: {max_region.capitalize()} ({max_count} spots)")
    else:
        print("No significant acne concentration detected.")

# Main function to execute the workflow
def main():
    print("Capturing image...")
    image_path = capture_image()

    print("Classifying image...")
    classification = classify_image(image_path)
    print("Classification Result:", classification)

    if classification == 'Acne':
        print("Detecting acne spots...")
        acne_positions = detect_acne(image_path)
        print("Acne Positions Detected:", acne_positions)

        print("Segmenting face regions...")
        regions = segment_face(image_path)
        print("Calculating acne density by region...")
        density = calculate_acne_density(acne_positions, regions)
        print("Acne Density:", density)

        print("Displaying insights...")
        display_insights(density)
    else:
        print("No acne detected.")

if __name__ == "__main__":
    main()
