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
x_data.shape
np.array(y_data).shape
x_data.shape


# 두근두근 모델링=================================================================================================

X = tf.placeholder(tf.float32,shape=[None,26])
Y = tf.placeholder(tf.float32,shape=[None,1])

nb_classes = 64 # 분류 갯수

W = tf.Variable(tf.random_normal([26,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

# 방법 1 : log함수를 사용하여 수식을 직접 표현
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

# 방법 2 : softmax_cross_entropy_with_logits() 함수 사용
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                             labels = Y)
cost =  tf.reduce_mean(cost_i)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(20001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, cost_val)



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
# ========================================================================================================================
# 가로의 기록
# 기록 1
# Accuracy: 0.6915173
# a=result.drop(columns=['Unnamed: 0','acc_id','survival_time','playtime', 'npc_kill', 'solo_exp', 'party_exp', 'quest_exp',
#                        'boss_monster', 'death'])

# 기록 2
# Accuracy: 0.69042975
# a=result.drop(columns=['Unnamed: 0','acc_id','survival_time','revive', 'exp_recovery', 'fishing', 'private_shop',
#                        'game_money_change', 'enchant_count', 'level'])

# 기록 3
# Accuracy: 0.7124053
#  a=result.drop(columns=['Unnamed: 0','acc_id','survival_time','pledge_cnt',
#                        'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
#                        'same_pledge_cnt', 'etc_cnt', 'num_opponent'])

# 기록 4
# Accuracy: 0.71506786
# a=result.drop(columns=['Unnamed: 0','acc_id','survival_time','amount_spent_pay', 'sell_item_cnt', 'buy_item_cnt', 'play_char_cnt',
#        'combat_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt_plg'])

# 기록 5
# Accuracy: 0.7127053
# a=result.drop(columns=['Unnamed: 0','acc_id','survival_time','random_defender_cnt_plg',
#        'same_pledge_cnt_plg', 'temp_cnt_plg', 'etc_cnt_plg',
#        'combat_play_time', 'non_combat_play_time', 'amount_spent'])
# ====================================================================================================
# 세로의 기록
# 기록 1
# Accuracy: 0.7125178

# 기록 2
# Accuracy: 0.70445514

# 기록 3
# Accuracy: 0.69759244

# 기록 4
# Accuracy: 0.6921173

# 기록 5
# Accuracy: 0.71409285

# 기록 6
# Accuracy: 0.7095177

# 기록 7
# Accuracy: 0.7095177


# =======================================================================
# 도출 1
# Accuracy: 0.7157054
# 'Unnamed: 0','acc_id','survival_time', 'combat_char_cnt',
# 'same_pledge_cnt','game_money_change','revive','pledge_cnt',
# 'amount_spent_pay'

#