from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('best.pt')

# Run inference using the webcam
results = model.predict(source=0, conf=0.25, stream=True)

# Loop through the streamed results
for result in results:
    frame = result.orig_img  # Get the original frame
    annotated_frame = result.plot()  # Draw bounding boxes on the frame
    
    # Display the annotated frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
