import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder

import joblib 


# pre-processing and training functions

def load_data(df_path):
    return pd.read_excel(df_path)


def divide_train_test(df,target):
    X_train, X_test ,y_train, y_test = train_test_split(df, df[target], train_size=0.8, random_state=42)

    return X_train, X_test, y_train, y_test 

def preprocessing(df, var = 'Engine'):
    df[var] = df[var].str.replace("Â", "", regex=True)

    return df 

def train_encode_categories(df,output_path):
    encoder = LabelEncoder()
    encoder.fit(df)
    joblib.dump(encoder,output_path)

    return encoder 

def encode_feature(df, encoder):
    encoder = joblib.load(encoder)

    return encoder.transform(df)

def train_scaler(df, output_path):
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler,output_path)

    return scaler 


def train_model(df,target,output_path):
    rf_model = RandomForestClassifier()

    rf_model.fit(df,target)

    joblib.dump(rf_model,output_path)

    return None 

def predict(df, model):
    model = joblib.load(model)
    
    return model.predict(df)