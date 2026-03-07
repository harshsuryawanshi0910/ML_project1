import pandas as pd
import numpy as np
import pickle
import os

class CustomData:
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score]
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise e

class PredictPipeline:
    def __init__(self):
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join(current_dir, 'artifacts', 'preprocessor.pkl')
        
    def load_model(self):
        try:
            print(f"Loading model from: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}")
            print("Please train the model first and save it to artifacts/model.pkl")
            raise
        except Exception as e:
            raise e
    
    def load_preprocessor(self):
        try:
            print(f"Loading preprocessor from: {self.preprocessor_path}")
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            return preprocessor
        except FileNotFoundError:
            print(f"Preprocessor file not found at {self.preprocessor_path}")
            print("Please train the model first and save it to artifacts/preprocessor.pkl")
            raise
        except Exception as e:
            raise e
    
    def predict(self, features):
        try:
            model = self.load_model()
            preprocessor = self.load_preprocessor()
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise e