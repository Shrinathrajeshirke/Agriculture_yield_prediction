import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Soil_Quality: float,
        Seed_Variety: int,
        Fertilizer_Amount_kg_per_hectare: float,
        Sunny_Days: float,
        Rainfall_mm: float,
        Irrigation_Schedule: int,
        ):

        self.Soil_Quality = Soil_Quality

        self.Seed_Variety = Seed_Variety

        self.Fertilizer_Amount_kg_per_hectare = Fertilizer_Amount_kg_per_hectare

        self.Sunny_Days = Sunny_Days

        self.Rainfall_mm = Rainfall_mm

        self.Irrigation_Schedule = Irrigation_Schedule


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Soil_Quality": [self.Soil_Quality],
                "Seed_Variety": [self.Seed_Variety],
                "Fertilizer_Amount_kg_per_hectare": [self.Fertilizer_Amount_kg_per_hectare],
                "Sunny_Days": [self.Sunny_Days],
                "Rainfall_mm": [self.Rainfall_mm],
                "Irrigation_Schedule": [self.Irrigation_Schedule],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)