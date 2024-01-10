from roboflow import Roboflow 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
from src.components.data_ingestion import DataIngestion

@dataclass
class Data_preprocessing():
    model_yaml_file: str = 'yolov5/models/yolov5s.yaml'
    custom_yaml_file: str = 'artifacts/model/'

    def __post_init__(self):
        os.makedirs(self.custom_yaml_file, exist_ok=True)

    def reading_nc(self, path):
        main_dict = {}
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if ':' in line:
                    index_data, main_data = line.strip().split(':', 1)
                    main_dict[index_data.strip()] = main_data.strip()
            number_of_classes = int(main_dict['nc'])
            logging.info(f'Number of classes in our custom Dataset is {number_of_classes}')
            return number_of_classes
        


    def custom_yaml_model(self, number_of_classes):
        with open(self.model_yaml_file, 'r') as model_file:
            yaml_content = model_file.read()
            new_yaml_content = yaml_content.replace('nc: 80', f'nc : {number_of_classes}')

        custom_file_final = os.path.join(self.custom_yaml_file, 'custom_yaml.yaml')
        
        # Check if the file exists
        file_exists = os.path.exists(custom_file_final)
        
        with open(custom_file_final, 'r+' if file_exists else 'w') as custom_file:
            custom_file.write(new_yaml_content)
        
        if file_exists:
            logging.info(f'Appended content to existing Custom YAML File at {self.custom_yaml_file}')
        else:
            logging.info(f'Created Custom YAML File with nc = {number_of_classes} at {self.custom_yaml_file}')
        
        return custom_file_final


if __name__ == '__main__':
    try:
        logging.info("Start dataset download...")
        ingestion = DataIngestion()
        dataset_path = ingestion.dataset_download()
        logging.info(f"Dataset downloaded successfully. Path: {dataset_path}")

        preprocess = Data_preprocessing()
        logging.info("Start reading number of classes...")
        num_classes = preprocess.reading_nc(os.path.join(dataset_path, 'data.yaml'))

        logging.info("Start creating custom YAML file...")
        final_yaml_file = preprocess.custom_yaml_model(num_classes)
        logging.info(f'The Custom MOdel file is at {final_yaml_file}')

    except Exception as e:
        logging.error(f"Error during data ingestion and preprocessing: {str(e)}")
