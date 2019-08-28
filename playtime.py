# # =========================================================
# # polyfit은 여기를 보세용!!
#
# # polyfit 결과 (앞에가 기울기, 뒤에가 절편)
# # cp.polyfit(x,y,차수)
#
# x = [1,2,3,4]
# y = [1,2,3,4]
#
# # 기울기가 1이 나오도록 설정해본다.
# gg = cp.polyfit(x,y,1)
# # print(gg)
# # [1.00000000e+00 8.25508711e-16]
# # 기울기 1, 절편도 사실상 0 (e-16 때문)
#
# # ===============================================================
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import scipy as cp
#
# train_label_df=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')
# train_label_df.tail()
#
# train_activity_df=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
# train_activity_df.tail()
#
# label_activity=pd.merge(train_label_df,train_activity_df,on='acc_id',how='inner')
# label_activity.tail()
#
# ptm=label_activity[['acc_id','day','playtime']]
# ptm.tail()
#
# # 이탈 비이탈 로지컬로 정확성을 따져보자
# # 플레이 타임을 acc_id와 day로 묶고 플레이 타임을 sum해버리고
# # 기울기를 구해서 음수면 줄어드는 것이고 양수면 늘어나는 것으로 판단.
#
#
# play_time=ptm.groupby(['acc_id','day'], as_index=False).sum()
# play_time.tail()
# play_time.idxmax()
# play_time.shape
# play_time[play_time.loc[:,'acc_id'] == ]
# sr=play_time[play_time['acc_id']==True]
# print(sr)
#
# # 플레이타임 갯수로 x를 만드는 것 (x축)
# # acc_id를 뽑고 플레이 타임(y축)
#
#
# play_time.dtypes
#
# play_time.loc[(130473)]
# play_time.loc[(130473, 28)]
#
#
# list(play_time.loc['acc_id'])
# list(play_time.loc[130473].acc_id)
# play_time.drop(columns='day')
#
#
# play_time.loc[:,'day']
# list(set(play_time.loc[:,'acc_id']))
# # 130473_끝자리 유저 id
# # 40000_중복없이 유저 id 수
#
# play_time.loc[:,'playtime']
# play_time.dtypes
# len(play_time['playtime'])
#
#
#

# ================================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as cp

train_label_df=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')
train_label_df.tail()

train_activity_df=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_activity.csv')
train_activity_df.tail()

label_activity=pd.merge(train_label_df,train_activity_df,on='acc_id',how='inner')
label_activity.tail()

ptm=label_activity[['acc_id','day','playtime']]
ptm.tail()

play_time=ptm.groupby(['acc_id','day'], as_index=False).sum()
play_time.tail()

# 중복없이 리스트로 뽑기
list(set(play_time.loc[:,'acc_id']))

# acc_id 걸리게 해주기
# play_time[play_time.loc[:,'acc_id'] == '원하는 acc_id']

# 기울기만 뽑기
# cp.polyfit(df['day'], df['playtime'], 1)[0]


# 시작
new_acc=list(set(play_time.loc[:,'acc_id']))
W = []
for i in new_acc:
    acc = play_time[play_time.loc[:,'acc_id'] == i]
    W.append(cp.polyfit(acc['day'], acc['playtime'], 1)[0])
print(W)

# acc_id 와 같은지
# len(W)
# 40000_길이


# acc_id, 와 W 합치기
woals=list(set(play_time.loc[:,'acc_id']))

# 기울기 100으로 키우기
[x*100 for x in W]

worhkd = [woals,W]

worhkd1 = pd.DataFrame(worhkd).T
print(worhkd1)

# rename
worhkd1.rename(columns={0:'acc_id'})
ehsl = worhkd1.rename(columns={1:'W',0:'acc_id'})
print(ehsl)

# float를 int형으로 바꾸기
ehsl['acc_id'] = ehsl['acc_id'].astype(int)
ehsl

# 1000을 곱해서 차이를 만들어보쟈.
# 상관관계 보고

label=pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/train_label.csv')
label.loc[label["survival_time"] == 64 , "survived"] = 1
label.loc[label["survival_time"] < 64 , "survived"] = 0
label=label.drop(columns=['amount_spent','survived'])


# 데이터 합치기
result=pd.merge(ehsl,label,how='inner')
result



# 학습==============================================================================================
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 원하는 컬럼만 MinMaxScaler
scaler = MinMaxScaler()
# result[['W']] = scaler.fit_transform(result['W'])

# 전체를 MinMaxScaler
print(scaler.fit(result))
result = scaler.transform(result)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(result[1], result[2],
                                                    test_size = 0.3, random_state= 123)

print(X_train.shape)
print(X_test.shape)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

nb_classes = 64

X = tf.placeholder(tf.float32, shape = [None, 1])
Y = tf.placeholder(tf.int32, shape = [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([1, nb_classes], name = 'weight'))
b = tf.Variable(tf.random_normal([nb_classes], name = 'bias'))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                    labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(50001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict = {X : X_train, Y : Y_train})
    if step % 500 == 0:
        print('step:', step,
              'cost_val:', cost_val,
              'W_val', W_val,
              'b_val', b_val)

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))

h, p, a = sess.run([hypothesis, predict, accuracy],feed_dict={X:X_test, Y:Y_test})

print('Hypothesis:' , h,
      '\nPredict:', p,
      '\nAccuracy:', a)