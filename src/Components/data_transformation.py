import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from logger import logging
from exception import CustomException
from src.utils import save_object


@dataclass
class Datatransformationconfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


def get_data_transformer_object():
    try:
        numerical_columns = ['writing_score', 'reading_score']
        categorical_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                               'test_preparation_course']
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mode")),
                ("onehotencoder", OneHotEncoder())
            ]
        )
        logging.info("numerical encoding completed")
        logging.info("categorical encoding completed")
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", cat_pipeline, categorical_feature)
            ]
        )
        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)


class Datatransformation:
    def __init__(self):
        self.data_transformation_config = Datatransformationconfig()

    def initiate_datatransformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading training and test dat complete")

            logging.info("Obtaining preprocessor object")
            preprocessing_obj = get_data_transformer_object()

            # numerical_column = ['writing_score', 'reading_score']
            target_column = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info(
                f"Applying preprocessing object on train and test dataset"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            CustomException(e, sys)
