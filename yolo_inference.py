from ultralytics import YOLO 

model = YOLO('yolov8x')

result = model.track('input_videos/input_video.mp4',conf=0.2, save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)