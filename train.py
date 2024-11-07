from google.colab import drive
drive.mount('/content/gdrive')

#data: https://www.kaggle.com/datasets/omidsakaki1370/fire-dataset-for-detection-with-yolov8

#pip install ultralytics
import os
from ultralytics import YOLO

ROOT_DIR = '/content/gdrive/MyDrive/fire_data'


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data=os.path.join(ROOT_DIR, "google_colab_config.yaml"), epochs=100)  # train the model