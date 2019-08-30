# ======================================================================================================================
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
# ================================================================================
result=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/AllMergeJamDonny.csv')
# result.loc[result["survival_time"] == 64 , "survived"] = 1
# result.loc[result["survival_time"] < 64 , "survived"] = 0

# 여기 주석
result = result[result["survival_time"] < 60]
# result.columns

# 변수 조정은 여기서!!!, 뒤에 숫자도 꼭 조정!

a=result[['combat_play_time','death','enchant_count','etc_cnt',
          'etc_cnt_plg','exp_recovery','level','non_combat_play_time',
          'num_opponent','party_exp','play_char_cnt','pledge_combat_cnt',
          'private_shop','quest_exp','random_attacker_cnt','random_attacker_cnt_plg',
          'random_defender_cnt','random_defender_cnt_plg','same_pledge_cnt_plg',
          'sell_item_cnt','temp_cnt','temp_cnt_plg','survival_time']]
a.tail()
# ==========================================================================
x_data = [a.iloc[i,:-1].tolist() for i in range(len(a.index.values))]
X = x_data

y_data = [[i] for i in result['survival_time'].tolist()]
y = y_data


# 데이터 나누기
from sklearn.model_selection import train_test_split
x_data, X_test, y_data, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

# x scaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
X = x_data

# # x scaler
# from sklearn.preprocessing import RobustScaler
# robustScaler = RobustScaler()
# robustScaler.fit(x_data)
# x_data = robustScaler.transform(x_data)


# print(type(X_train))
# print(type(Y_train))
# print(type(X_test))
# print(type(Y_test))


# Build Graph
tf.set_random_seed(3209)
nb_classes = 63
X = tf.placeholder(tf.float32, shape=[None, 22])
Y = tf.placeholder(tf.int32, shape=[None, 1])

# Y 값을 one_hot encoding으로 변환, Y값은 반드시 int형으로 입력
Y_one_hot = tf.one_hot(Y, nb_classes)                # [None, 1, 7]
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # numpy에서 행을 자동으로 크기에 맞게 다시 조절한다 "-1"


W1 = tf.Variable(tf.random_normal([22, nb_classes]), name='weight1')
b1 = tf.Variable(tf.random_normal([nb_classes]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X,W1) + b1 )

W2 = tf.Variable(tf.random_normal([nb_classes, nb_classes]), name='weight2')
b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)

W3 = tf.Variable(tf.random_normal([nb_classes, nb_classes]), name='weight3')
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias3')
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3)

# W4 = tf.Variable(tf.random_normal([nb_classes, nb_classes]), name='weight3')
# b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias3')
# L4 = tf.nn.relu(tf.matmul(L3,W4) + b4)

logits = tf.matmul(L3,W3) + b3

hypothesis = tf.nn.softmax(logits)
# hypothesis


# cost function
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)


# 경사하강법
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습단계(start learning)

for step in range(100001):
    cost_val, W1_val, b1_val,W2_val, b2_val , W3_val, b3_val ,_ = \
        sess.run([cost, W1, b1, W2, b2, W3, b3,optimizer],
                 feed_dict = {X : x_data, Y : y_data})
    if step % 200 == 0:
        print('\nstep:', step,
              '\ncost_val:', cost_val)


# 검증단계(test, 정확도 측정)
# Accuracy Computation(정확도 계산)
predict = tf.argmax(hypothesis, 1) # 1:행단위 # 예측값 행에서 가장 큰 값, 확률값을 구한다.
# predict = tf.argmax(hypothesis,) # 0:열단위

correct_predict = tf.equal(predict, tf.argmax(Y_one_hot, 1))  # predict와 y 값을 비교한다.
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32)) # 숫자로 바꾼 뒤 평균을 낸다.

h, p, a = sess.run([hypothesis, predict, accuracy],feed_dict={X:x_data, Y:y_data})

print('Hypothesis:' , h,
      '\nPredict:',p,
      '\nAccuracy:', a)


# # 결과=====================================================================
# 변수들=====================================================================
# 'playtime', 'npc_kill', 'solo_exp', 'party_exp', 'quest_exp',
#        'boss_monster', 'death', 'exp_recovery', 'fishing', 'private_shop',
#        'enchant_count', 'level', 'random_attacker_cnt', 'random_defender_cnt',
#        'temp_cnt', 'num_opponent', 'sell_item_cnt', 'buy_item_cnt',
#        'play_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt_plg',
#        'random_defender_cnt_plg', 'same_pledge_cnt_plg', 'temp_cnt_plg',
#        'etc_cnt_plg','amount_spent','survival_time'
# ==========================================================================
# Accuracy: 0.36218962
# cost : 2.57
# 3층
# for step in range(50001):
# if step % 2000 == 0:
# ==========================================================================
# Accuracy: 0.2476254
# cost : 3.086735
# 2층
# for step in range(10001):
# if step % 200 == 0:
# ==========================================================================
# Accuracy:0.1757207
# cost :3.313125
# 4층
# for step in range(10001):
# if step % 200 == 0:
# ==========================================================================
# Accuracy: 0.28670222
# cost : 2.921914
# 3층
# for step in range(10001):
# if step % 200 == 0:
# ==========================================================================
# Accuracy: 0.20579903
# cost : 3.2921083
# 3층
# for step in range(5401):
# if step % 200 == 0:
# ============================================================================
# 유리한 변수로 만들기(4개 뽑아서 )