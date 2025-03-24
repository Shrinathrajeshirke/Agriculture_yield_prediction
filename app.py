from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
   
app=application

## Route for a home page


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Soil_Quality=request.form.get('Soil_Quality'), 
            Seed_Variety=request.form.get('Seed_Variety'),
            Fertilizer_Amount_kg_per_hectare=request.form.get('Fertilizer_Amount_kg_per_hectare'),
            Sunny_Days=request.form.get('Sunny_Days'),
            Rainfall_mm=request.form.get('Rainfall_mm'),
            Irrigation_Schedule=float(request.form.get('Irrigation_Schedule')),
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        print("Before calling predict pipeline")
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print("After calling predict pipeline")
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)