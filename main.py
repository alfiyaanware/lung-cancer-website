from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
# import cv2
import sklearn
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

app = Flask(__name__)


@app.route("/")
def renders():
  return render_template('home.html')


@app.route('/about')
def about():
  return render_template('about.html')


@app.route("/treat")
def treat():
  return render_template('treat.html')


@app.route("/predict")
def predict():
  return render_template('predict.html')


@app.route("/detect")
def detect():
  return render_template('detect.html')


@app.route('/results', methods=['POST'])
def result():
  to_predict_list = request.form.to_dict()
  copy = request.form.to_dict()
  # to_predict_list = list(to_predict_list)
  temp = to_predict_list['obesity']
  to_predict_list['obesity'] = round((int(temp) - 6) * (10 - 0) / (105 - 7))
  print(to_predict_list)

  new_input_df = pd.DataFrame([to_predict_list])

  scaler = pickle.load(open('scaler.pkl', 'rb'))
  new_input_df[:] = scaler.transform(new_input_df[:])

  lr = pickle.load(open('model.pkl', 'rb'))

  result = lr.predict(new_input_df)[0]

  return render_template('results.html', prediction=result, dict=copy)


@app.route("/imageresult", methods=['POST'])
def imageresult():
  file = request.files['scan']
  img = Image.open(file.stream)
  img = img.resize((256, 256))
  img_array = np.array(img)
  img_array = img_array / 255.0
  knn = pickle.load(open('knn.pkl', 'rb'))
  prediction = knn.predict(img_array)

  return render_template('imageresult.html', prediction=prediction)


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
