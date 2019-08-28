import numpy as np
import pandas as pd

payment_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_payment.csv')
label_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')

payment_df.tail()
label_df.tail()

a=payment_df.groupby('acc_id').sum()
a=a.reset_index()
a.shape    # (23726, 3)
b=label_df.groupby('acc_id').sum()
b=b.reset_index()
b.shape    # (40000, 3)
c=pd.merge(b,a,how='outer',on='acc_id')
c.shape  # (40000, 5)
c

d=c.drop(columns=['day','survival_time'])
d=d.fillna(0)
d
d['amount_spent']=d['amount_spent_x'] + d['amount_spent_y']
all_money=d.drop(columns=['amount_spent_x','amount_spent_y'])
all_money.shape # (40000,)
all_money

# ==========================================================================================
# 돈과 생존은 관련이 있을 것인가.

label_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')
label_df.shape
a=label_df.drop(columns='amount_spent')

rktjf1=pd.merge(all_money,a,how='inner',on='acc_id')
rktjf1.shape
rktjf1
# (40000, 3)

# ==============================================================================================
# 돈 안쓰고 생존한 사람
anrhkrma = rktjf1[(rktjf1.loc[:,'amount_spent']==0) & (rktjf1.loc[:,'survival_time']>63)]
anrhkrma
# [6408 rows x 3 columns]

# 돈 안쓴사람
anrhkrma1 = rktjf1[(rktjf1.loc[:,'amount_spent']==0)]
anrhkrma1
# [12612 rows x 3 columns]

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

# 돈 쓰고 생존
rktjf1_1=rktjf1[(rktjf1.loc[:,'amount_spent'] > 0) & (rktjf1.loc[:,'survival_time']>63)]
rktjf1_1
# [15588 rows x 3 columns]

# 돈 쓰고 이탈
rktjf1_2=rktjf1[(rktjf1.loc[:,'amount_spent'] > 0) & (rktjf1.loc[:,'survival_time']<64)]
rktjf1_2
# [11800 rows x 3 columns]


# =========================================================================
# 플레이 총량과 돈 쓰는 것 관계!

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

# play_time 총량
activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
activity_df.tail()
act=activity_df.groupby(['acc_id'],as_index=False).sum()
act_df=act[['acc_id','playtime']]
act_df.tail()

# outer로 합치기
ehs_vmffpdl=pd.merge(rktjf1_3,act_df,how='inner',on='acc_id')
ehs_vmffpdl.tail()
ehs_vmffpdl.corr()
#                  acc_id  amount_spent  survival_time  playtime
# acc_id         1.000000      0.000500      -0.006711  0.003166
# amount_spent   0.000500      1.000000       0.021679  0.083952
# survival_time -0.006711      0.021679       1.000000  0.270864
# playtime       0.003166      0.083952       0.270864  1.000000





# ===========================변수 시작=================================================
# =====================================================================================
# 단발성 전투 횟수의 합과 돈

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

pledge_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_pledge.csv')
pledge_df.tail()
plg=pledge_df.groupby(['acc_id'],as_index=False).sum()
plg2=plg[['acc_id','temp_cnt']]
plg2

qustn1 = pd.merge(rktjf1_3,plg2,how='left',on='acc_id')
qustn1


# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]

pledge_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_pledge.csv')

plg=pledge_df.groupby(['acc_id'],as_index=False).sum()
plg3=plg[['acc_id','play_char_cnt']]
plg3

qustn2 = pd.merge(rktjf1_3,plg3,how='left',on='acc_id')

qustn2=qustn2[['acc_id','play_char_cnt']]
qustn2



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
activity_df.tail()
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg4=plg[['acc_id','boss_monster']]
plg4

qustn3 = pd.merge(rktjf1_3,plg4,how='left',on='acc_id')
qustn3=qustn3[['acc_id','boss_monster']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
activity_df.tail()
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg5=plg[['acc_id','fishing']]
plg5

qustn4 = pd.merge(rktjf1_3,plg5,how='left',on='acc_id')
qustn4=qustn4[['acc_id','fishing']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

combat_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_combat.csv')
combat_df.tail()
plg=combat_df.groupby(['acc_id'],as_index=False).sum()
plg6=plg[['acc_id','num_opponent']]
plg6

qustn5 = pd.merge(rktjf1_3,plg6,how='left',on='acc_id')
qustn5=qustn5[['acc_id','num_opponent']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

pledge_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_pledge.csv')
pledge_df.tail()
plg=pledge_df.groupby(['acc_id'],as_index=False).sum()
plg7=plg[['acc_id','combat_play_time']]
plg7

qustn6 = pd.merge(rktjf1_3,plg7,how='left',on='acc_id')
qustn6=qustn6[['acc_id','combat_play_time']]

# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
rktjf1_3
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
activity_df.tail()
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg8=plg[['acc_id','npc_kill']]
plg8

qustn7 = pd.merge(rktjf1_3,plg8,how='left',on='acc_id')
qustn7=qustn7[['acc_id','npc_kill']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg9=plg[['acc_id','quest_exp']]

qustn8 = pd.merge(rktjf1_3,plg9,how='left',on='acc_id')
qustn8=qustn8[['acc_id','quest_exp']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

pledge_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_pledge.csv')
plg=pledge_df.groupby(['acc_id'],as_index=False).sum()
plg10=plg[['acc_id','random_defender_cnt']]

qustn9 = pd.merge(rktjf1_3,plg10,how='left',on='acc_id')
qustn9=qustn9[['acc_id','random_defender_cnt']]




# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

pledge_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_pledge.csv')
plg=pledge_df.groupby(['acc_id'],as_index=False).sum()
plg11=plg[['acc_id','pledge_combat_cnt']]

qustn10 = pd.merge(rktjf1_3,plg11,how='left',on='acc_id')
qustn10=qustn10[['acc_id','pledge_combat_cnt']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

combat_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_combat.csv')
plg=combat_df.groupby(['acc_id'],as_index=False).sum()
plg12=plg[['acc_id','level']]

qustn11 = pd.merge(rktjf1_3,plg12,how='left',on='acc_id')
qustn11=qustn11[['acc_id','level']]



# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

combat_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_combat.csv')
plg=combat_df.groupby(['acc_id'],as_index=False).sum()
plg13=plg[['acc_id','class']]

qustn12 = pd.merge(rktjf1_3,plg13,how='left',on='acc_id')
qustn12=qustn12[['acc_id','class']]

# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg14=plg[['acc_id','solo_exp']]

qustn13 = pd.merge(rktjf1_3,plg14,how='left',on='acc_id')
qustn13=qustn13[['acc_id','solo_exp']]


# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg15=plg[['acc_id','playtime']]

qustn14 = pd.merge(rktjf1_3,plg15,how='left',on='acc_id')
qustn14=qustn14[['acc_id','playtime']]


# =====================================================================================

# 돈 쓴 사람
rktjf1_3 = rktjf1[(rktjf1.loc[:,'amount_spent'] > 0)]
# [27388 rows x 3 columns]

activity_df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
plg=activity_df.groupby(['acc_id'],as_index=False).sum()
plg16=plg[['acc_id','game_money_change']]

qustn15 = pd.merge(rktjf1_3,plg16,how='left',on='acc_id')
qustn15=qustn15[['acc_id','game_money_change']]


# 변수 전체를 merge==================================================================================

result=pd.merge(qustn1,qustn2,how='left',on='acc_id')
result1=pd.merge(result,qustn3,how='left',on='acc_id')
result2=pd.merge(result1,qustn4,how='left',on='acc_id')
result3=pd.merge(result2,qustn5,how='left',on='acc_id')
result4=pd.merge(result3,qustn6,how='left',on='acc_id')
result5=pd.merge(result4,qustn7,how='left',on='acc_id')
result6=pd.merge(result5,qustn8,how='left',on='acc_id')
result7=pd.merge(result6,qustn9,how='left',on='acc_id')
result8=pd.merge(result7,qustn10,how='left',on='acc_id')
result9=pd.merge(result8,qustn11,how='left',on='acc_id')
result10=pd.merge(result9,qustn12,how='left',on='acc_id')
result11=pd.merge(result10,qustn13,how='left',on='acc_id')
result12=pd.merge(result11,qustn14,how='left',on='acc_id')
result13=pd.merge(result12,qustn15,how='left',on='acc_id')

result_a=result13
result_a.shape    # (27388, 18)
result_a.columns

# 컬럼순서변경
result_b=pd.DataFrame(result_a, columns=['temp_cnt', 'play_char_cnt',
       'boss_monster', 'fishing', 'num_opponent', 'combat_play_time',
       'npc_kill', 'quest_exp', 'random_defender_cnt', 'pledge_combat_cnt',
       'level', 'class', 'solo_exp', 'playtime', 'game_money_change','amount_spent',])

result_b.fillna(0,inplace=True)

y = list(result_b.iloc[:,-1])
result_b = result_b.iloc[:,:-1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(result_b)
result_b = scaler.transform(result_b)
type(result_b)
result_b = pd.DataFrame(result_b)
# 두근두근 학습===================================================================================

import tensorflow as tf
import numpy as np

x1_data = list(result_b.iloc[:,0])
x2_data = list(result_b.iloc[:,1])
x3_data = list(result_b.iloc[:,2])
x4_data = list(result_b.iloc[:,3])
x5_data = list(result_b.iloc[:,4])
x6_data = list(result_b.iloc[:,5])
x7_data = list(result_b.iloc[:,6])
x8_data = list(result_b.iloc[:,7])
x9_data = list(result_b.iloc[:,8])
x10_data = list(result_b.iloc[:,9])
x11_data = list(result_b.iloc[:,10])
x12_data = list(result_b.iloc[:,11])
x13_data = list(result_b.iloc[:,12])
x14_data = list(result_b.iloc[:,13])
x15_data = list(result_b.iloc[:,14])

y_data = y

X = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)
X4 = tf.placeholder(tf.float32)
X5 = tf.placeholder(tf.float32)
X6 = tf.placeholder(tf.float32)
X7 = tf.placeholder(tf.float32)
X8 = tf.placeholder(tf.float32)
X9 = tf.placeholder(tf.float32)
X10 = tf.placeholder(tf.float32)
X11 = tf.placeholder(tf.float32)
X12 = tf.placeholder(tf.float32)
X13 = tf.placeholder(tf.float32)
X14 = tf.placeholder(tf.float32)
X15 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)



W1 = tf.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
W3 = tf.Variable(tf.random_normal([1]), name='weight3')
W4 = tf.Variable(tf.random_normal([1]), name='weight4')
W5 = tf.Variable(tf.random_normal([1]), name='weight5')
W6 = tf.Variable(tf.random_normal([1]), name='weight6')
W7 = tf.Variable(tf.random_normal([1]), name='weight7')
W8 = tf.Variable(tf.random_normal([1]), name='weight8')
W9 = tf.Variable(tf.random_normal([1]), name='weight9')
W10 = tf.Variable(tf.random_normal([1]), name='weight10')
W11 = tf.Variable(tf.random_normal([1]), name='weight11')
W12 = tf.Variable(tf.random_normal([1]), name='weight12')
W13 = tf.Variable(tf.random_normal([1]), name='weight13')
W14 = tf.Variable(tf.random_normal([1]), name='weight14')
W15 = tf.Variable(tf.random_normal([1]), name='weight15')


b = tf.Variable(tf.random_normal([1]), name='bias')


# 예측 방정식
hypothesis = X*W1 + X2*W2 + X3*W3+ X4*W4+ X5*W5+ X6*W6+ X7*W7+ X8*W8+ X9*W9+ X10*W10+ X11*W11+ X12*W12+ X13*W13+ X14*W14+ X15*W15 + b

# 비용함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

    # 학습 시작
    # start training
for step in range(10001):
    cost_val, W1_val,W2_val,W3_val,\
    W4_val,W5_val,W6_val,W7_val,W8_val,W9_val,\
    W10_val,W11_val,W12_val,W13_val,W14_val,W15_val,b_val, _ = \
        sess.run([cost, W1, W2, W3, W4, W5, W6, W7, W8, W9,W10, W11, W12,W13, W14, W15,b, train],
                 feed_dict={X:x1_data,X2:x2_data,X3:x3_data,
                            X4:x4_data,X5:x5_data,X6:x6_data,
                            X7:x7_data,X8:x8_data,X9:x9_data,
                            X10:x10_data,X11:x11_data,X12:x12_data,
                            X13:x13_data,X14:x14_data,X15:x15_data,Y:y_data})
    if step % 100 == 0:
        print(step, cost_val, W1_val,W2_val,W3_val, b_val)

    # # predict : test model
    # x1_data = [73.,93.,89.,96.,73.]
    # x2_data = [80.,88.,91.,98.,66.]
    # x3_data = [75.,93.,90.,100.,70.]
    #
    # print(sess.run(hypothesis, feed_dict = {x1:x1_data,x2:x2_data,x3:x3_data}))