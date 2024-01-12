 # YOLOv5 Object Detection with Roboflow and Flask

This project demonstrates how to use YOLOv5 for object detection, with data ingestion and preprocessing handled using the Roboflow library. The project also includes a Flask API for live object detection using a webcam.

## Prerequisites

- Python 3.8 or higher
- Roboflow account
- YOLOv5 installed
- Flask installed

## Project Structure

The project is structured as follows:

```
├── app.py
├── requirements.txt
├── src
    ├── components
        ├── data_ingestion.py
        ├── data_preprocessing.py
        ├── model_trainer.py
    ├── exception.py
    ├── logger.py
    ├── utils
        ├── main_utils.py
```

## Data Ingestion

The `data_ingestion.py` script handles downloading the dataset from Roboflow. It uses the Roboflow Python library to download the dataset and extract it to a specified folder.

```python
from roboflow import Roboflow 
from src.logger import logging  
from src.exception import CustomException
from dataclasses import dataclass
import os


@dataclass
class DataIngestion:
    folder: str = os.path.abspath('data')

    def __post_init__(self):
        os.makedirs(self.folder, exist_ok=True)

    def dataset_download(self):
        try:
            rf = Roboflow(api_key="   ")
            project = rf.workspace("exploring-roboflow-and-yolo").project("sign-detection-78n0m") 
            dataset = project.version(1).download("yolov5", location=self.folder)
            
            # Make sure to return the absolute path to the downloaded data
            dataset_path = os.path.abspath(self.folder)
            
            logging.info(f'Dataset Downloaded Successfully. Path: {dataset_path}')
            return dataset_path

        except Exception as e:
            logging.error(f'Error in dataset_download: {str(e)}')
            raise CustomException(f'Dataset download failed: {str(e)}')
```

## Data Preprocessing

The `data_preprocessing.py` script is responsible for preprocessing the downloaded dataset. This typically involves extracting number of class values and creating a custom_yaml.yaml
file

```python
@dataclass
class Data_preprocessing():
    model_yaml_file: str = 'yolov5/models/yolov5s.yaml'
    custom_yaml_file: str = 'artifacts/model/'

    def __post_init__(self):
        os.makedirs(self.custom_yaml_file, exist_ok=True)

    def reading_nc(self, path):
        #code

    def custom_yaml_model(self, number_of_classes):
        #code
```

## Model Trainer

The `model_trainer.py` script is responsible for training the YOLOv5 model on the preprocessed dataset. It involves setting up the configuration for training, specifying the model architecture, and initiating the training process

```python
@dataclass
class TrainModel:

    def train_model(self, trainer_yaml_path, data_yaml_path):
````

## Flask App
The app.py script is a Flask application that serves as an API for live object detection using a webcam. It uses the trained YOLOv5 model to perform real-time object detection on video frames.

## Getting Started
Install required dependencies: pip install -r requirements.txt
Run the Flask app: python app.py
Access the live object detection API at http://localhost:8080
Additional Components
exception.py: Contains custom exception classes for better error handling.
logger.py: Configures logging for the project.
utils/main_utils.py: Utility functions used across different components.
Feel free to explore the project and adapt it to your specific use case. If you encounter any issues or have questions, please reach out for assistance.

For more information on YOLOv5, Roboflow, and Flask, refer to their respective documentation.

YOLOv5 GitHub
Roboflow Documentation
Flask Documentation

## Steps to Follow 
1) git clone https://github.com/Mandalor-09/Sign_Detection_yolo.git
2) cd /Sign_Detection_yolo/
3) git clone https://github.com/ultralytics/yolov5
4) pip install -r requirements.txt
5) python src/components/model_trainer.py
6) python app.py

[![Watch the video](https://github.com/Mandalor-09/Sign_Detection_yolo/blob/main/sign_detection.png)](https://github.com/Mandalor-09/Sign_Detection_yolo/blob/main/working.mp4)

