from ultralytics import YOLO
import cv2
import numpy as np
import mss

# Load the model
model = YOLO('best.pt')

# Create an mss instance to capture the screen
sct = mss.mss()

# Define the region of the screen to capture (e.g., full screen)
monitor = sct.monitors[1]  # monitor[1] is usually the primary screen

# Create a named window for OpenCV
cv2.namedWindow('YOLOv8 Screen Capture', cv2.WINDOW_NORMAL)

# Move the window to the second monitor (adjust coordinates accordingly)
cv2.moveWindow('YOLOv8 Screen Capture', 0, -1080)  # Move it directly above the primary monitor

# Capture the screen and run detection in a loop
while True:
    # Capture a screenshot
    screenshot = sct.grab(monitor)
    
    # Convert the screenshot to a NumPy array (OpenCV format)
    img = np.array(screenshot)

    # Convert from BGRA (mss default) to BGR (OpenCV uses BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Run YOLO model on the captured frame
    results = model.predict(source=img_bgr, conf=0.25)

    # Annotate the frame with bounding boxes
    annotated_frame = results[0].plot()  # Use the first result per frame

    # Display the annotated frame
    cv2.imshow('YOLOv8 Screen Capture', annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
