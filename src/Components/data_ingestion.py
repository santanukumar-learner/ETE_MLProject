import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import Datatransformation
from data_transformation import Datatransformationconfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            df = pd.read_csv("notebook/stud.csv")
            logging.info("Read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=101)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = Datatransformation()
    data_transformation.initiate_datatransformation(train_data, test_data)
