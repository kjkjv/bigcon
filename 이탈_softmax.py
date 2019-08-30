import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ==========================================================================================
# 원하는 컬럼만 MinMaxScaler
# scaler = MinMaxScaler()
# result[['W']] = scaler.fit_transform(result['W'])

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


mg = pd.merge(act_df,fishing_df,how='inner',on='acc_id')
mg2 = pd.merge(mg,private_shop_df,how='inner',on='acc_id')
mg3 = pd.merge(mg2,label_df,how='inner',on='acc_id')
result = mg3
result.shape
result.tail()
# =============================================================================================
Allmerge2 = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/github/bigcon/Allmerge2.csv')
Allmerge2.tail()
Allmerge2.shape
Allmerge3 = Allmerge2.drop(columns=['Unnamed: 0', 'class','level', 'acc_id'])
Allmerge3.columns

all_merge = pd.DataFrame(Allmerge3,columns=['pledge_cnt','same_pledge_cnt', 'private_shop',
                                            'pledge_combat_cnt','survival_time'])

all_merge.tail()
all_merge.shape

all_merge.iloc[:,-1].tolist()
all_merge.iloc[:,:-1]




# ===============================================================================================


x_data = [result.iloc[i,:-1].tolist() for i in result.index.values]
X = x_data

y_data = [[i] for i in result['survival_time'].tolist()]
y = y_data


# ========================================================================================

# 데이터 나누기
from sklearn.model_selection import train_test_split
x_data, X_test, y_data, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

# =========================================================================================


# x 조절
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_data.shape  #(26666, 32)
np.array(y_data).shape   #(26666, 32)



# 두근두근 모델링=========================================================================

nb_classes =  64

X = tf.placeholder(tf.float32,shape=[None,5])
Y = tf.placeholder(tf.int32,shape=[None,1])

Y_one_hot = tf.one_hot(Y,nb_classes)  # [None,1,7]
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes]) # [None,7]

W = tf.Variable(tf.random_normal([5,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y_one_hot)
cost =  tf.reduce_mean(cost_i)

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,
                                     dtype = tf.float32))

# start training
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, 'cost:',cost_val)

h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)
