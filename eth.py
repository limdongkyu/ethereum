import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import utils
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.externals import joblib
import matplotlib.pyplot as plt
# import pickle
from datetime import datetime
import time

# tf.random.set_seed(777)
"""
디렉토리 구조

├── data
│   └── eth_Investing.com.csv # 사용되어질 데이터
├── eth.py # 현재 파일
├── eth_re.py # 데이터 + 모델 로드 후 검증 파일
├── model
    ├── eth_Investing.com.h5 - 모델 정보 저장될 파일 자동 생김
    ├── eth_Investing.com.pkl - 데이터 저장될 파일 자동 생김

"""
# csv 파일로 원하는 만큼 데이터 다운로드
# https://kr.investing.com/crypto/ethereum/historical-data
# 해당 사이트에 이더리움 가격정보 제공 날짜 부터 현재 날짜까지 가져온 후 일주일치만 실제 테스트를 위해 삭제 후 사용 함
# "날짜","현재가","오픈","고가","저가","거래량","변동 %"
df = pd.read_csv('./data/eth_Investing.com.csv')

df.rename(columns={
    '날짜': 'Date',
    '현재가': 'Close',
    '오픈': 'Open',
    '고가': 'High',
    '저가': 'Low',
    '거래량': 'Trade',
    '변동 %': 'Per'
}, inplace=True)


def converter_date(data_string):
    return time.mktime(datetime.strptime(data_string, '%Y년 %m월 %d일').timetuple())


def converter_trade(data_string):
    if data_string[-1] == 'K':
        return float(data_string[:-1]) * 1024
    elif data_string[-1] == 'M':
        return float(data_string[:-1]) * 1024 * 1024
    else:
        return 0  # '-' 버려야 되나? 일단 0 으로 처리


def converter_per(data_string):
    return float(data_string[:-1]) * 0.01


# 사용 할 의미가 있는지 감이 안옴
df['Date'] = df.apply(lambda row: converter_date(row['Date']), axis=1)
df['Trade'] = df.apply(lambda row: converter_trade(row['Trade']), axis=1)
df['Per'] = df.apply(lambda row: converter_per(row['Per']), axis=1)

# print(df['Trade'][752])

print(df.head())

"""
 Date   Close    Open    High     Low        Trade     Per
0  1.576076e+09  145.03  143.51  145.82  140.00  12824084.48  0.0106
1  1.575990e+09  143.51  145.88  146.57  142.53  11460935.68 -0.0161
2  1.575904e+09  145.86  147.88  148.48  144.49  11660165.12 -0.0137
3  1.575817e+09  147.88  151.17  151.75  147.02  10139729.92 -0.0218
4  1.575731e+09  151.17  148.25  152.22  147.09   9730785.28  0.0195

"""

print(df.info())

"""
RangeIndex: 754 entries, 0 to 753
Data columns (total 7 columns):
Date     754 non-null float64
Close    754 non-null object
Open     754 non-null object
High     754 non-null object
Low      754 non-null object
Trade    754 non-null float64
Per      754 non-null float64
dtypes: float64(3), object(4)
memory usage: 41.4+ KB
None
"""

print(df.describe())

"""
 Date         Trade         Per
count  7.540000e+02  7.540000e+02  754.000000
mean   1.543547e+09  7.137677e+06    0.000222
std    1.881838e+07  7.772213e+06    0.052350
min    1.511017e+09  0.000000e+00   -0.201800
25%    1.527282e+09  1.326449e+06   -0.023400
50%    1.543547e+09  5.793382e+06   -0.001050
75%    1.559812e+09  1.019216e+07    0.024900
max    1.576076e+09  8.123318e+07    0.232200

"""
# print(df.values)

df = df[['Open', 'High', 'Low', 'Close']]


data = df.values
train = data[:(len(data) - int(len(data) * 0.3))]
test = data[:int(len(data) * 0.3)]

transformer = MinMaxScaler()
train = transformer.fit_transform(train)
test = transformer.transform(test)

print('train length = ', len(train))
# train length =  528
print('test length = ', len(test))
# test length =  226

sequence_length = 7
window_length = sequence_length + 1

x_train = []
y_train = []
for i in range(0, len(train) - window_length + 1):
    window = train[i:i + window_length, :]
    x_train.append(window[:-1])
    y_train.append(window[-1])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for i in range(0, len(test) - window_length + 1):
    window = test[i:i + window_length, :]
    x_test.append(window[:-1])
    y_test.append(window[-1])
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

utils.shuffle(x_train, y_train)

joblib.dump(transformer, 'model/eth_Investing.com.pkl')


# 모델 학습
# 모델 검증

input_param = Input(shape=(sequence_length, 4))

net = LSTM(units=10)(input_param)
net = Dense(units=4)(net)
model = Model(inputs=input_param, outputs=net)

model.summary()
model.compile(loss=tf.compat.v1.keras.losses.MSE, optimizer=Adam(lr=0.01))
history = model.fit(x_train, y_train, epochs=60, validation_data=(x_test, y_test),
                    callbacks=[ModelCheckpoint(filepath='model/eth_Investing.com.h5', save_best_only=True, verbose=1)])

# 모델 예측

print(transformer)
print(y_test)
print(y_test.shape)

y_test_inverse = []
for y in y_test:
    inverse = transformer.inverse_transform([[0, 0, 0, y[0]]])
    y_inverse = inverse.flatten()[-1]
    print(y_inverse)
    y_test_inverse.append(y_inverse)

y_predict = model.predict(x_test)
y_predict_inverse = []
for y in y_predict:
    inverse = transformer.inverse_transform([[0, 0, 0, y[0]]])
    y_inverse = inverse.flatten()[-1]
    print(y_inverse)
    y_predict_inverse.append(y_inverse)

plt.plot(y_test_inverse)
plt.plot(y_predict_inverse)
plt.xlabel('Time Period')
plt.ylabel('Close')
plt.show()
"""
['Open', 'High', 'Low', 'Close']
Close    Open    High     Low        Trade     Per
"127.90","132.93","133.90","127.28","20.11M","-3.78%"
"132.92","122.06","133.93","117.14","20.23M","8.89%"
"122.06","132.78","132.96","120.63","17.04M","-8.07%"
"132.78","142.55","142.75","131.00","17.60M","-6.84%"
"142.53","142.03","143.87","140.30","11.31M","0.35%"
"142.03","144.95","145.17","141.40","11.15M","-2.02%"
"144.96","145.04","145.43","143.51","10.95M","-0.05%"
"""
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
print('x_test = ', x_test)

y_predict = model.predict(x_test)
print('y_predict = ', y_predict)
print(y_predict.flatten()[0])

inverse = transformer.inverse_transform([[0, 0, 0, y_predict.flatten()[0]]])
print(inverse.flatten()[-1])
