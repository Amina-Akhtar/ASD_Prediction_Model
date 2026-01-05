from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

l_encoders=joblib.load(r"Model\Label_Encoders.pkl")
scaler=joblib.load(r"Model\Standard_Scaler.pkl")
model=joblib.load(r"Model\Stacked_Model.pkl")
app=FastAPI()
model_cols=["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","Age_Mons","Sex","Jaundice","Family_mem_with_ASD","Ethnicity_Hispanic","Ethnicity_Latino","Ethnicity_Native Indian","Ethnicity_Others","Ethnicity_Pacifica","Ethnicity_White European","Ethnicity_asian","Ethnicity_black","Ethnicity_middle eastern","Ethnicity_mixed","Ethnicity_south asian","Who_completed_the_test_family member","Who_completed_the_test_health care professional","Who_completed_the_test_others","Who_completed_the_test_self"]

class UserInput(BaseModel):
    A1:int
    A2:int
    A3:int
    A4:int
    A5:int
    A6:int
    A7:int
    A8:int
    A9:int
    A10:int
    Age_Mons:int
    Sex:str
    Ethnicity:str
    Jaundice:str
    Family_mem_with_ASD:str
    Who_completed_the_test:str

def preprocess(data:dict):
    df=pd.DataFrame([data])
    # label encoding
    df["Sex"]=l_encoders["Sex"].transform(df["Sex"])
    df["Jaundice"]=l_encoders["Jaundice"].transform(df["Jaundice"])
    df["Family_mem_with_ASD"]=l_encoders["Family_mem_with_ASD"].transform(df["Family_mem_with_ASD"])
    # for col,le in l_encoders.items():
    #     df[col]=le.transform(df[col])
    # avoid applying label encoding on target variable
    # one hot encoding
    df=pd.get_dummies(df,columns=["Ethnicity","Who_completed_the_test"],drop_first=False)
    #standard scaling
    for c in model_cols:
        if c not in df.columns:
            df[c]=0
    df=df[model_cols]
    df["Age_Mons"]=scaler.transform(df[["Age_Mons"]])
    return df

@app.post("/predict")
def predict_autism(input:UserInput):
    data=preprocess(input.dict())
    prediction= model.predict(data)[0]
    # probability per class
    prediction_proba = model.predict_proba(data)[0]  
    confidence = max(prediction_proba)  
    return {
        "prediction":int(prediction),
        "confidence":float(confidence),
        "features":data.to_dict(orient="records")[0]
    }  

# No->0 and Yes->1 in Class/ASD Traits  
# uvicorn app:app --reload   
# 1. app file name 2. app fastapi instance
