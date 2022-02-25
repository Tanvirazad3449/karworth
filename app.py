from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

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
    fixed_model = model.lstrip()
    predTest = pd.DataFrame(data=np.array([brand, fixed_model, year, transmission, mileage, fuelType, mpg, engineSize]).reshape(1,8),
                        columns = ['brand','model','year','transmission','mileage','fuelType', 'mpg', 'engineSize' ])
    pr_result = pr_model.predict(predTest)
    pr_result = "{:,}".format(int(pr_result[0]))
    print(pr_result)
    return pr_result

if __name__ == "__main__":
    app.run(debug=True)


# Loads pre-trained model
#model = load_model('deployment_28042020')