import cv2
import os
import time
import uuid

image_path = '.data/'  # Make sure to provide the correct path
total_image = 10
labels = ["Hello", "Yes", "No", "Thanks", "IloveYou", "Please"]

# Create the image directory if it doesn't exist
os.makedirs(image_path, exist_ok=True)

video = cv2.VideoCapture(0)

for label in labels:
    print(f"Collecting images for {label}")
    time.sleep(5)
    for j in range(total_image):
        image_name = f'{label}_{j}_{str(uuid.uuid4())}.jpg'
        ret, frame = video.read()
        cv2.imshow('Frame', frame)
        cv2.imwrite(os.path.join(image_path, image_name), frame)
        time.sleep(2)
        if (cv2.waitKey(1) & 0xFF == ord('q')):  # Fixed the condition here
            break

video.release()
cv2.destroyAllWindows()
