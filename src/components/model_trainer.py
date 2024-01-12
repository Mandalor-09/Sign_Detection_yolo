from dataclasses import dataclass
import os
import yaml
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import Data_preprocessing

@dataclass
class TrainModel:

    def train_model(self, trainer_yaml_path, data_yaml_path):
        #trainer_yaml_path = os.path.join('./', trainer_yaml_path)
        #data_yaml_path = os.path.join('./', data_yaml_path)  # Specify the data folder here
        try:
            logging.info("Loading trainer configuration from YAML file...")
            with open(trainer_yaml_path, 'r') as trainer_file:
                trainer_config = yaml.safe_load(trainer_file)

                if not isinstance(trainer_config, dict):
                    raise CustomException("Invalid YAML file")

            logging.info("Constructing and executing the training command...")

            # Adjusting paths to absolute paths 
            # --save-period 5
            training_command = f"python yolov5/train.py --img 640 --batch 16 --epochs 51 --data {os.path.abspath(data_yaml_path)} --cfg {os.path.abspath(trainer_yaml_path)} --weights yolov5s.pt --name yolov5s_results --cache"
            logging.info(training_command)

            # Check if the training command was successful
            if os.system(training_command) == 0:
                logging.info("Model training completed.")
            else:
                logging.error("Model training failed.")

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")

if __name__ == '__main__':
    try:
        logging.info("Start dataset download...")
        ingestion = DataIngestion()
        dataset_path = ingestion.dataset_download()
        logging.info(f"Dataset downloaded successfully. Path: {dataset_path}")

        preprocess = Data_preprocessing()
        logging.info("Start reading the number of classes...")
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        num_classes = preprocess.reading_nc(data_yaml_path)

        logging.info("Start creating a custom YAML file...")
        final_yaml_file = preprocess.custom_yaml_model(num_classes)
        logging.info(f'The Custom Model file is at {final_yaml_file}')

        trainer = TrainModel()
        logging.info("Start training the model...")
        logging.info(f'<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>> {final_yaml_file} {data_yaml_path}')

        trainer.train_model(final_yaml_file, data_yaml_path)

    except Exception as e:
        logging.error(f"Error during data ingestion and preprocessing: {str(e)}")
