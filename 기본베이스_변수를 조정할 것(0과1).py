# ==================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
import scipy as cp
# ======================================================================================================================
result=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/AllMergeJamDonny.csv')
result
result.loc[result["survival_time"] == 64 , "survived"] = 1
result.loc[result["survival_time"] < 64 , "survived"] = 0
a=result.drop(columns=['Unnamed: 0','acc_id','survival_time'])
a.tail()


# ==========================================================================
x_data = [a.iloc[i,:-1].tolist() for i in a.index.values]
X = x_data

y_data = [[i] for i in a['survived'].tolist()]
y = y_data


# 데이터 나누기
from sklearn.model_selection import train_test_split
x_data, X_test, y_data, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)


# x scaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_data.shape
np.array(y_data).shape
x_data.shape


# 두근두근 모델링=================================================================================================
tf.set_random_seed(7)
X = tf.placeholder(tf.float32,shape=[None,35])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([35,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid : tf.div(1., 1. + tf.exp(tf.matmul(X,W)))
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype = tf.float32))

# start training
for step in range(10001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, 'cost:',cost_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X:x_data, Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)


# =====================================================================================
# 0과 1로만 생존 확인
# (현재 가장 최적의 상태)

# tf.set_random_seed(7)
# learning_rate=0.001
# range(10001)
# step % 200
# 10000 cost: 0.56200665
# Accuracy: 0.7157429


# =========================================================================================
# 오늘의 목표!!
# 변수들 빼가면서 최대한 높이고 나서 (우선은 90까지 해보기)
# 64로 분류해보기!!!
