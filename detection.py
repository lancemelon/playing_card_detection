from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict(source='testVid.mp4', conf=0.25, save=True, stream=True)