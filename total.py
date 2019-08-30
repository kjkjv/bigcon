# ======================================================================================================================
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

scaler = MinMaxScaler()
random.seed(54654)

matplotlib.rcParams['font.family']='Malgun Gothic'   # 한글 사용
matplotlib.rcParams['axes.unicode_minus'] = False

label = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')


# ======================================================================================================================


activity = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')

# activity acc_id로 groupby
# - 평균을 내지 않은 이유 : 평균을 냈을 경우 캐릭터는 많지만
#   한 캐릭터만으로 활동한 사람의 정보가 과소평가 될 가능성이 있음
activity = activity.groupby(['acc_id'], as_index = False).sum()
activity.drop(columns = ['day','char_id'], inplace = True)
# print(activity.head())



# ======================================================================================================================


combat = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_combat.csv')

combat.drop(columns = ['day', 'server', 'char_id', 'class'], inplace = True)
combat_a = combat.groupby(['acc_id'], as_index = False).sum()
combat_a.drop(columns = 'level', inplace = True)

combat_b = combat.groupby(['acc_id'], as_index = False).max()
#  combat.groupby('acc_id', as_index = False).sum().sort_values('acc_id')

# acc_id 기준으로 정리

combat_b = combat_b[['acc_id', 'level']]
combat = pd.merge(combat_b, combat_a, how = 'inner', on = 'acc_id')



# ======================================================================================================================


payment = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_payment.csv')

payment = payment.groupby('acc_id', as_index = False).sum()
payment.drop(columns = 'day', inplace = True)

payment.rename(columns = {'amount_spent' : 'amount_spent_pay'}, inplace = True)


# ======================================================================================================================


trade = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_trade.csv')

# 거래에 참여한 횟수를 기준으로

# 판매자로서 활동한 acc_id
trade_a = trade.groupby('source_acc_id', as_index = False).count()
trade_a = trade_a[['source_acc_id', 'day']]

# 구매자로서 활동한 acc_id
trade_b = trade.groupby('target_acc_id', as_index = False).count()
trade_b = trade_b[['target_acc_id', 'day']]

x = trade_a['day'].sum() - trade_b['day'].sum()
# print(x) # 0

trade_a.rename(columns={'source_acc_id':'acc_id',
                        'day':'sell_item_cnt'}, inplace=True)
trade_b.rename(columns={'target_acc_id':'acc_id',
                        'day':'buy_item_cnt'}, inplace=True)

trade = pd.merge(trade_a, trade_b, how = 'outer', on = 'acc_id')

# 실제 데이터 검색
# trade[trade['source_acc_id'] == 6].count()

# 데이터 검증
# trade_a[trade_a['source_acc_id'] == 6]


# ======================================================================================================================


pledge = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_pledge.csv')

ple_1 = pledge.groupby(['server', 'pledge_id', 'day'], as_index = False).mean()
ple_1.drop(columns = ['acc_id', 'char_id', 'day'], inplace = True)
ple_1 = ple_1.groupby(['server', 'pledge_id'], as_index = False).sum()

ple_a = pledge.groupby(['acc_id', 'char_id', 'server', 'pledge_id', 'day'], as_index = False).mean()
ple_a = ple_a.groupby(['acc_id', 'char_id', 'server', 'pledge_id'], as_index = False).sum()
ple_a = ple_a[['acc_id', 'char_id', 'server', 'pledge_id']]

pledge = pd.merge(ple_a, ple_1, how = 'outer', on = ['server', 'pledge_id'])
pledge.drop(columns = ['char_id', 'server', 'pledge_id'], inplace = True)
pledge_total = pledge.groupby(['acc_id'], as_index = False).mean()

pledge_total.rename(columns = {'random_attacker_cnt' : 'random_attacker_cnt_plg',
                              'random_defender_cnt' : 'random_defender_cnt_plg',
                              'same_pledge_cnt' : 'same_pledge_cnt_plg',
                              'temp_cnt' : 'temp_cnt_plg',
                              'etc_cnt':'etc_cnt_plg'}, inplace = True)

# ple_1[(ple_1['pledge_id'] == 25467) & (ple_1['server'] == 'aq')]
# ple_a[(ple_a['pledge_id'] == 25467) & (ple_a['server'] == 'aq')]


# ======================================================================================================================


# label + activity
label_a = pd.merge(label, activity, how = 'outer', on = 'acc_id')

# (label + activity) + combat
label_b = pd.merge(label_a, combat, how = 'outer', on = 'acc_id')

# (label + activity + combat) + payment
label_c = pd.merge(label_b, payment, how = 'outer', on = 'acc_id')

# (label + activity + combat + payment) + trade
label_d = pd.merge(label_c, trade, how = 'outer', on = 'acc_id')
label_d = label_d[label_d['survival_time'] >= 1]

# (label + activity + combat + payment + trade) + pledge_total
label_z = pd.merge(label_d, pledge_total, how = 'outer', on = 'acc_id')

data = label_z.fillna(0)
print(data.columns)

data[['playtime', 'npc_kill',
       'solo_exp', 'party_exp', 'quest_exp', 'boss_monster', 'death', 'revive',
       'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
       'enchant_count', 'level', 'pledge_cnt', 'random_attacker_cnt',
       'random_defender_cnt', 'temp_cnt', 'same_pledge_cnt', 'etc_cnt',
       'num_opponent', 'amount_spent_pay', 'sell_item_cnt', 'buy_item_cnt',
       'play_char_cnt', 'combat_char_cnt', 'pledge_combat_cnt',
       'random_attacker_cnt_plg', 'random_defender_cnt_plg',
       'same_pledge_cnt_plg', 'temp_cnt_plg', 'etc_cnt_plg',
       'combat_play_time', 'non_combat_play_time']] = \
    scaler.fit_transform(data[['playtime', 'npc_kill',
       'solo_exp', 'party_exp', 'quest_exp', 'boss_monster', 'death', 'revive',
       'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
       'enchant_count', 'level', 'pledge_cnt', 'random_attacker_cnt',
       'random_defender_cnt', 'temp_cnt', 'same_pledge_cnt', 'etc_cnt',
       'num_opponent', 'amount_spent_pay', 'sell_item_cnt', 'buy_item_cnt',
       'play_char_cnt', 'combat_char_cnt', 'pledge_combat_cnt',
       'random_attacker_cnt_plg', 'random_defender_cnt_plg',
       'same_pledge_cnt_plg', 'temp_cnt_plg', 'etc_cnt_plg',
       'combat_play_time', 'non_combat_play_time']])

data = data.fillna(0)
print(data.columns)

# ======================================================================================================================

data.loc[data['amount_spent_pay'] == 0, "cash"] = 0
data.loc[data['amount_spent_pay'] > 0, "cash"] = 1

# ======================================================================================================================


# 무과금

data_1 = data[data["cash"] == 0]
print(data_1.count())
data_1_corr = data_1.corr()

data_1.loc[data_1['survival_time'] < 64, "survived"] = 0
data_1.loc[data_1['survival_time'] == 64, "survived"] = 1

data_1_2 = data_1.corr()

# ======================================================================================================================

# 무과금 + 이탈

data_2 = data_1[data_1["survived"] == 0]

data_2_1 = data_2.corr()
"""

# 서로 반비례관계에 있을 시 제외

survival_time :
    playtime # 0.19
        npc_kill                # 0.5
        party_exp               # 0.17
        death                   # 0.2
        revive                  # 0.15
        private_shop            # 0.33
        level                   # 0.13
        random_defender_cnt     # 0.37
        etc_cnt                 # 0.15
        num_opponent            # 0.17
        sell_item_cnt           # 0.15
        random_defender_cnt_plg # 0.15
        combat_play_time        # 0.29
        non_combat_play_time    # 0.33
    npc_kill # 0.15
        rich_monster            # 0.12
        death                   # 0.11
        level                   # 0.40
        random_defender_cnt     # 0.26
        etc_cnt                 # 0.24
        num_opponent            # 0.23
    rich_monster
        solo_exp                # 0.47
        quest_exp               # 0.30
        death                   # 0.22
        revive                  # 0.23
        level                   # 0.17
        num_opponent            # 0.24
        play_char_cnt           # 0.14
        pledge_combat_cnt       # 0.14
        random_defender_cnt_plg # 0.18
    level
        combat_play_time        # 0.31
    num_opponent
        pledge_cnt              # 0.41
    
"""
# print(data_2.columns)
x_e = data_2[[    'acc_id', 'playtime', 'npc_kill',
                'solo_exp', 'party_exp', 'quest_exp', 'boss_monster', 'death', 'revive',
                'private_shop', 'level', 'pledge_cnt', 'random_defender_cnt', 'etc_cnt',
                'num_opponent', 'sell_item_cnt', 'play_char_cnt', 'pledge_combat_cnt',
                'random_defender_cnt_plg', 'combat_play_time', 'non_combat_play_time'    ]]
y_e = data_2[[    'acc_id', 'survived'    ]]
x = x_e.set_index('acc_id')
y = y_e.set_index('acc_id')

print(y.shape)
print(x.shape)
print(y.columns)
print(y)



X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state= 8431)

# Build Graph
X = tf.placeholder(tf.float32, shape = [None, 20])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([20, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


# 예측 방정식(Hypothesis)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)


# 비용 함수 (logistic regression)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))

# SGD Optimizer = 경사하강법
optimizer = tf.train.AdamOptimizer(learning_rate= 0.00001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 학습단계
print('learning start')
for step in range(100001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={X:X_train, Y:Y_train})

    if step % 10000 == 0:
        print('\nStep: ', step,
              '\nCost: ', cost_val)
print('learning finish')


# 정확도 계산(accuracy computation)
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict , Y), dtype = tf.float32))
# tf.equal : 괄호 안에 있는 두가지가 같으면 True 다르면 False
# tf.cast : True 면 1 False 면 0 으로 바꿔준다.
# tf.reduce_mean : 다 더한 후 평균을 계산

h,p,a = sess.run([hypothesis, predict, accuracy],
                 feed_dict= {X:X_test , Y:Y_test})

print('\nHypothesis:', h, '\nPredict:', p, '\nAccuracy:', a) # Accuracy: 0.7763158

# ----------------------------------------------------------------------------------------------------------------------
#
#
# # 무과금 + 비이탈
# data_3 = data_1[data_1["survived"] == 1]
#
#
# # ======================================================================================================================
#
#
# # 이탈 예측
#
# # sigmoid 활용
#
#
#
#
# # ======================================================================================================================
#
#
#
# # 과금
# data_2 = data[data["cash"] == 1]
# print(data_2.count())
# data_2_corr = data_2.corr()