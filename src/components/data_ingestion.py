import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeException
from dataclasses  import  dataclass
from sklearn.model_selection import train_test_split

@dataclass

class DataIngestionConfig:  #  fetch data from database
    train_file_path=os.path.join("artifacts","train.csv") # put train file in artifacts
    test_file_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

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
            self.ingestion_config.train_file_path)

        except Exception as e:
            logging.info("Error occured in data ingestoion stage")
            raise CustomeException(sys,e)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()