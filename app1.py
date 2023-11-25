import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = joblib.load(r"C:\Users\KIRAN\Desktop\Dynamic\Fifth SGPA Prdictor Deployment\fifth_sgpa_predictor.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global df

    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)
    print(input_features)
    # Validate input
    # if features_value < 0 or features_value > 10:
    for i in features_value:
        if i < 1 or i > 10:
            return render_template('index.html', prediction_text='Please enter valid input values.')

    output = model.predict([features_value])[0].round(2)
    if output > 10:
        output = 10

    sgpa1 = input_features[0]
    sgpa2 = input_features[1]
    sgpa3 = input_features[2]
    sgpa4 = input_features[3]
    sgpa5 = output

    # Input and predicted value store in df then save in a CSV file
    df2 = pd.DataFrame({'sgpa1': [sgpa1], 'sgpa2': [sgpa2], 'sgpa3': [sgpa3], 'sgpa4': [sgpa4], 'sgpa5': [output]})
    df = pd.concat([df, df2], ignore_index=True)
    csv_file = 'predicted.csv'
    header = not os.path.isfile(csv_file)
    df.to_csv(csv_file, mode='a', header=header, index=False)

    # plt pie for innput values
    plt.pie(features_value, labels = ['SGPA1', 'SGPA2', 'SGPA3', 'SGPA4'], explode = [0, 0.1, 0, 0], autopct = "%0.2f%%", shadow = True)
    os.remove('static/pie.png')
    plt.savefig('static/pie.png')

    return render_template('output.html', sgpa1=sgpa1, sgpa2=sgpa2, sgpa3=sgpa3, sgpa4=sgpa4, sgpa5=sgpa5, pie_path="static/pie.png")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
    # from werkzeug.serving import run_simple
    # run_simple('localhost', 5000, app)