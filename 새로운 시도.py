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

# 여기!!!================================================================================
result=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/AllMergeJamDonny.csv')
result.columns
# result.loc[result["survival_time"] == 64 , "survived"] = 1
# result.loc[result["survival_time"] < 64 , "survived"] = 0

result = result[result["survival_time"] < 64]
result.columns

# 변수 조정은 여기서!!!, 뒤에 숫자도 꼭 조정!
result=result.drop(columns=['Unnamed: 0','acc_id', 'combat_char_cnt',
                       'same_pledge_cnt','game_money_change','revive','pledge_cnt',
                       'amount_spent_pay','etc_cnt','combat_play_time', 'non_combat_play_time'])
result.columns

a=result[['playtime', 'npc_kill', 'solo_exp', 'party_exp', 'quest_exp',
       'boss_monster', 'death', 'exp_recovery', 'fishing', 'private_shop',
       'enchant_count', 'level', 'random_attacker_cnt', 'random_defender_cnt',
       'temp_cnt', 'num_opponent', 'sell_item_cnt', 'buy_item_cnt',
       'play_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt_plg',
       'random_defender_cnt_plg', 'same_pledge_cnt_plg', 'temp_cnt_plg',
       'etc_cnt_plg','amount_spent','survival_time']]
a.shape
# ==========================================================================
x_data = [a.iloc[i,:-1].tolist() for i in range(len(a.index.values))]
X = x_data


y_data = [[i] for i in result['survival_time'].tolist()]
y = y_data


# 데이터 나누기
from sklearn.model_selection import train_test_split
x_data, X_test, y_data, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

# x scaler
# scaler = MinMaxScaler()
# scaler.fit(x_data)
# x_data = scaler.transform(x_data)
# X = x_data

# x scaler
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
robustScaler.fit(x_data)
x_data = robustScaler.transform(x_data)



learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32,shape=[None,784])
Y = tf.placeholder(tf.float32,shape=[None,10])

W1 = tf.get_variable("W1", shape = [784,512],
            initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
# L1 = tf.sigmoid(tf.matmul(X,W1) + b1 )
L1 = tf.nn.relu(tf.matmul(X,W1) + b1 )

W2 = tf.get_variable("W2", shape = [512,512],
            initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
# L2 = tf.sigmoid(tf.matmul(L1,W2) + b2 )
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2 )

W3 = tf.get_variable("W3", shape = [512,512],
            initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
# L3 = tf.sigmoid(tf.matmul(L2,W3) + b3 )
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3 )

W4 = tf.get_variable("W4", shape = [512,512],
            initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name='bias4')
# L4 = tf.sigmoid(tf.matmul(L3,W4) + b4 )
L4 = tf.nn.relu(tf.matmul(L3,W4) + b4 )

W5 = tf.get_variable("W5", shape = [512,10],
            initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name='bias5')

logits = tf.matmul(L4,W5) + b5
hypothesis = tf.nn.softmax(logits)

cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# start training
for epoch in range(training_epochs) :  # 15
    avg_cost = 0
    # 550 = 55000/100
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = { X:batch_xs , Y:batch_ys}
        c,_= sess.run([cost,optimizer],feed_dict = feed_dict )
        avg_cost += c / total_batch
    print('Epoch:','%04d'%(epoch + 1), 'cost:','{:.9f}'.format(avg_cost))
