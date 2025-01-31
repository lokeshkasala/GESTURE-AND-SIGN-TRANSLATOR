import os
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)
directory = 'Image/'

# Ensure directories exist
for letter in 'abcdefghijklmnopqrstuvwxyz':
    os.makedirs(os.path.join(directory, letter.upper()), exist_ok=True)

# Function to get the count of images in a given directory
def get_image_count(letter):
    return len(os.listdir(os.path.join(directory, letter.upper())))

# Dictionary to store counts of images per letter
counts = {letter: get_image_count(letter) for letter in 'abcdefghijklmnopqrstuvwxyz'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Display full frame and region of interest (ROI)
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)

    # Capture key press
    interrupt = cv2.waitKey(10)
    key = chr(interrupt & 0xFF).lower()  # Convert key code to character

    if key in counts:
        # Save the image
        path = os.path.join(directory, key.upper(), f"{counts[key]}.png")
        cv2.imwrite(path, roi)
        # Increment count
        counts[key] += 1

    # Exit condition
    if interrupt & 0xFF == ord('a'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
