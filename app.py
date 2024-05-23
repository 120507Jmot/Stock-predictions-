from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    data = yf.download(ticker, period='5y')
    
    data['Adj Close'] = data['Adj Close'].fillna(method='ffill')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

    prediction_days = 60
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    test_data = yf.download(ticker, period='60d')
    actual_prices = test_data['Adj Close'].values
    total_dataset = np.concatenate((data['Adj Close'].values, actual_prices))

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    signals = []
    for i in range(1, len(predicted_prices)):
        if predicted_prices[i] > predicted_prices[i - 1]:
            signals.append("Buy")
        else:
            signals.append("Sell")

    return jsonify({
        'predicted_prices': predicted_prices.tolist(),
        'actual_prices': actual_prices.tolist(),
        'signals': signals
    })

if __name__ == '__main__':
    app.run(debug=True)