from keras.models import load_model
from flask import Flask, jsonify, request
import pickle
import numpy as np

model = load_model('model.h5')
scaler_target = pickle.load(open('scaler_target.pkl', 'rb'))
scaler_data = pickle.load(open('scaler_data.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['data1']
    data2 = request.form['data2']
    data3 = request.form['data3']
    data4 = request.form['data4']
    data = [data1, data2, data3, data4]
    predict_input = np.array(data).reshape(1, -1)
    # predict_input = scaler_data.transform(predict_input)
    # prediction = scaler_target.inverse_transform(model.predict(predict_input))
    # output = {'prediction': prediction.tolist()[0]}
    # return jsonify(output)

@app.route('/')
def index():
    return '''
        <form method="POST" action="/predict">
            <label>data1</label><br>
            <input type="text" name="data1"><br>
            <label>data2</label><br>
            <input type="text" name="data2"><br>
            <label>data3</label><br>
            <input type="text" name="data3"><br>
            <label>data4</label><br>
            <input type="text" name="data4"><br>
            <input type="submit" value="predict">
        </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
