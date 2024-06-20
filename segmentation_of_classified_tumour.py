from ultralytics import YOLO
import os

folder_path = r"C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\Datasets\Dataset 5\Testing\glioma"

# List all files in the folder
files = os.listdir(folder_path)

# List containing paths to all image files
data = [os.path.join(folder_path, file) for file in files]

model = YOLO(r'C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\models\yolov8n-seg-d3.pt')

for image_path in data:
    image_name = os.path.basename(image_path)
    results=model(image_path, save=True, project=r'C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\Datasets\Dataset 6\Testing\glioma', name=image_name)
    # print("Image Path", image_path)
    # print("Image Name", image_name)