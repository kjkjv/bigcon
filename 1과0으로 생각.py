import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# play_time 총량
activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
activity_df.tail()
act=activity_df.groupby(['acc_id'],as_index=False).sum()
act_df=act[['acc_id','playtime']]
act_df.tail()
act_df.shape

# fishing 총량
fishing=activity_df.groupby(['acc_id'],as_index=False).sum()
fishing_df=fishing[['acc_id','fishing']]
fishing_df.tail()
fishing_df.shape

# private_shop 총량
private_shop=activity_df.groupby(['acc_id'],as_index=False).sum()
private_shop_df=private_shop[['acc_id','private_shop']]
private_shop_df.tail()
private_shop_df.shape

# 생존
label_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')
label_df.tail()
label_df.shape

label_1=label_df.loc[label_df["survival_time"] == 64 , "survived"] = 1
label_1=label_df.loc[label_df["survival_time"] < 64 , "survived"] = 0
label_1

mg = pd.merge(act_df,fishing_df,how='inner',on='acc_id')
mg2 = pd.merge(mg,private_shop_df,how='inner',on='acc_id')
mg3 = pd.merge(mg2,label_df,how='inner',on='acc_id')
result = mg3
result.shape
result.tail()
result.fillna(0)


x_data = [result.iloc[i,:-1].tolist() for i in result.index.values]
X = x_data

y_data = [[i] for i in result['survival_time'].tolist()]
y = y_data


# 데이터 나누기
from sklearn.model_selection import train_test_split
x_data, X_test, y_data, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)


# x 조절
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_data.shape  #(26666, 6)
np.array(y_data).shape   #(26666, 32)
x_data.shape


# 두근두근 모델링=================================================================================================
tf.set_random_seed(7)
X = tf.placeholder(tf.float32,shape=[None,6])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([6,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid : tf.div(1., 1. + tf.exp(tf.matmul(X,W)))
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*
                       tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation
# cast = 형변환을 해준다.
# 기준을 삼는 과정
# 0과 1을 출력

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),
                                 dtype = tf.float32))

# start training
for step in range(14301):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 500 == 0:
        print(step, 'cost:',cost_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X:x_data, Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)

