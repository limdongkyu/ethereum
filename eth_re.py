from sklearn.externals import joblib
from tensorflow.keras.models import load_model


transformer = joblib.load('model/eth_Investing.com.pkl')

model = load_model('model/eth_Investing.com.h5')

##########모델 예측

x_test = transformer.transform([
    [132.93, 133.90, 127.28, 127.90],
    [122.06, 133.93, 117.14, 132.92],
    [132.78, 132.96, 120.63, 122.06],
    [142.55, 142.75, 131.00, 132.78],
    [142.03, 143.87, 140.30, 142.53],
    [144.95, 145.17, 141.40, 142.03],
    [145.04, 145.43, 143.51, 144.96]
])

x_test = x_test.reshape((1, 7, 4))
print(x_test)

y_predict = model.predict(x_test)
print(y_predict)
print(y_predict.flatten()[0])

inverse = transformer.inverse_transform([[0, 0, 0, y_predict.flatten()[0]]])
print(inverse.flatten()[-1])