import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeException
from dataclasses  import  dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass

class DataIngestionConfig:  #  fetch data from database
    train_file_path=os.path.join("artifacts/data_ingestion","train.csv") # put train file in artifacts
    test_file_path=os.path.join("artifacts/data_ingestion","test.csv")
    raw_data_path=os.path.join("artifacts/data_ingestion","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            logging.info("Data reading from local system")
            data=pd.read_csv(os.path.join("notebook/data","income_cleandata.csv"))  # data read 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)  # make directory 
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            train_set,test_set=train_test_split(data,test_size=0.30,random_state=42)
            logging.info("Data splited into train test")
            train_set.to_csv(self.ingestion_config.train_file_path,index=False)
            test_set.to_csv(self.ingestion_config.test_file_path,index=False)
            logging.info("Data Ingestion completed")
            return (self.ingestion_config.train_file_path,
            self.ingestion_config.test_file_path)

        except Exception as e:
            logging.info("Error occured in data ingestoion stage")
            raise CustomeException(sys,e)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_tranformation(train_data_path,test_data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))