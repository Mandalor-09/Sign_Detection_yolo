from dataclasses import dataclass
import os
import yaml
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import Data_preprocessing
from src.components.model_trainer import TrainModel


def start_training():
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
        #logging.info(f'<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>> {final_yaml_file} {data_yaml_path}')

        trainer.train_model(final_yaml_file, data_yaml_path)
        return 'Training Completed'

    except Exception as e:
        logging.error(f"Error during data ingestion and preprocessing: {str(e)}")
