# handle missing value 
# outliers treatment
# handle imbalanced data
#convert categorical columns into numerical columns

import os,sys
import  pandas as pd
import numpy as np
from src.exception import CustomeException
from src.logger import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join("artifacts/data_transformation","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Started")
            numerical_features=['age','workclass','education_num','marital_status',
                                'occupation','relationship','race','sex','capital_gain',
                                'capital_loss','hours_per_week','native_country']
            
            num_pipeline= Pipeline(steps=[("imputer",SimpleImputer(strategy='median')),
                                          ("scalar",StandardScaler())
                                          ])
            
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_features)
            ])
            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)
        
    def remote_outliers_IQR(self,col,df):
        try:
            Q1=df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)
            iqr=Q3-Q1

            upper_limit=Q3 +1.5*iqr
            lower_limit=Q1-1.5*iqr

            df.loc[(df[col]>upper_limit),col]=upper_limit
            df.loc[(df[col]>lower_limit),col]=lower_limit

            return df
        except Exception as e:
            logging.info("Outliers handling code")
            raise CustomeException(e,sys)

    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            numerical_features=['age','workclass','education_num','marital_status',
                                'occupation','relationship','race','sex','capital_gain',
                                'capital_loss','hours_per_week','native_country']
            
            for col in numerical_features:
                self.remote_outliers_IQR(col=col,df=train_data)
            logging.info("outliers capped on train data")

            for col in numerical_features:
                self.remote_outliers_IQR(col=col,df=test_data)

            preprocessor_obj=self.get_data_transformation_obj()
            target_columns="income"
            drop_columns=[target_columns]

            logging.info("splitting train data into dependent and independent features")

            input_feature_train_data=train_data.drop(drop_columns,axis=1)
            target_feature_train_data=train_data[target_columns]

            logging.info("splitting test data into dependent and independent features")

            input_feature_test_data=test_data.drop(drop_columns,axis=1)
            target_feature_test_data=test_data[target_columns]
            # apply transformstion on our train data and test data

            input_train_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr=preprocessor_obj.transform(input_feature_test_data)

            # applying preprocessor

            train_array=np.c_[input_train_arr,np.array(target_feature_train_data)]
            test_array=np.c_[input_test_arr,np.array(target_feature_test_data)]

            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path,obj=preprocessor_obj
                        )

            return (train_array,test_array,self.data_transformation_config.preprocess_obj_file_path)
        except Exception as e:
            raise CustomeException(e,sys)

