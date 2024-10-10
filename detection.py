from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict(source='testScreenRecord.mp4', conf=0.25, save=True, stream=True)

for result in results:
    print(result)