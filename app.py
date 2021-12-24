from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

pr_model = pickle.load(open("karworth_lr_model.pkl", "rb"))
df = pd.read_csv("Karworth_Clean_Dataframe.csv")
# Initalise the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    brand = sorted(df['brand'].unique())
    year = sorted(df['year'].unique(), reverse=True)
    transmission = sorted(df['transmission'].unique())    
    fuelType = sorted(df['fuelType'].unique())    
    engineSize = sorted(df['engineSize'].unique())
    return render_template('index.html',brand = brand, year = year, transmission = transmission, fuelType = fuelType, engineSize = engineSize)

@app.route('/predict', methods = ['POST'])
def predict():
    brand = request.form.get('brand')
    model = request.form.get('model')
    year = request.form.get('year')
    fuelType = request.form.get('fuelType')
    transmission = request.form.get('transmission')
    engineSize = request.form.get('engineSize')
    mpg = request.form.get('mpg')
    mileage = request.form.get('mileage')

    # ohe = OneHotEncoder()
    # cols = ['brand','model','year', 'mileage', 'mpg', 'engineSize', 'transmission','fuelType']
    # data_unseen = pd.DataFrame([[brand,model,year,fuelType,transmission,engineSize,mpg,mileage]], columns = ['brand','model','year', 'mileage', 'mpg', 'engineSize', 'transmission','fuelType'])
    # oheFit = ohe.fit(data_unseen[cols])
    # oheFit = oheFit.transform(data_unseen[cols])

    # print("oheFit")

    # print(oheFit)
    fixed_model = model.lstrip()
    predTest = pd.DataFrame(data=np.array([brand, fixed_model, year, transmission, mileage, fuelType, mpg, engineSize]).reshape(1,8),
                        columns = ['brand','model','year','transmission','mileage','fuelType', 'mpg', 'engineSize' ])

    pr_result = pr_model.predict(predTest)
    pr_result = "{:,}".format(int(pr_result[0]))

    print(pr_result)





    
    # final = np.array([brand,model,year,fuelType,transmission,engineSize,mpg,mileage])
    # data_unseen = pd.DataFrame([final], columns = ['brand','model','year', 'mileage', 'mpg', 'engineSize', 'transmission','fuelType'])                         
    
    # prediction = pr_model.predict(data_unseen)
    return pr_result

if __name__ == "__main__":
    app.run(debug=True)


# Loads pre-trained model
#model = load_model('deployment_28042020')