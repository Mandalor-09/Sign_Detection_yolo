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
            rf = Roboflow(api_key=" ")
            project = rf.workspace("exploring-roboflow-and-yolo").project("sign-detection-78n0m") 
            dataset = project.version(1).download("yolov5", location=self.folder)
            
            # Make sure to return the absolute path to the downloaded data
            dataset_path = os.path.abspath(self.folder)
            
            logging.info(f'Dataset Downloaded Successfully. Path: {dataset_path}')
            return dataset_path

        except Exception as e:
            logging.error(f'Error in dataset_download: {str(e)}')
            raise CustomException(f'Dataset download failed: {str(e)}')

if __name__ == '__main__':
    ingestion = DataIngestion()
    dataset_path = ingestion.dataset_download()
    print(f"Dataset path: {dataset_path}")
