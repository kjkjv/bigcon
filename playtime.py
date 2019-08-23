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

# 이탈 비이탈 로지컬로 정확성을 따져보자
# 플레이 타임을 acc_id와 day로 묶고 플레이 타임을 sum해버리고
# 기울기를 구해서 음수면 줄어드는 것이고 양수면 늘어나는 것으로 판단.


play_time=ptm.groupby(['acc_id','day'], as_index=False).sum()
play_time.tail()
play_time.idxmax()
play_time.shape

# 플레이타임 갯수로 x를 만드는 것 (x축)
# acc_id를 뽑고 플레이 타임(y축)


play_time.dtypes

play_time.loc[(130473)]
play_time.loc[(130473, 28)]


list(play_time.loc['acc_id'])
list(play_time.loc[130473].acc_id)
play_time.drop(columns='day')


play_time.loc[:,'day']
list(set(play_time.loc[:,'acc_id']))

play_time.loc[:,'playtime']
play_time.dtypes


# =========================================================
# polyfit은 여기를 보세용!!

# polyfit 결과 (앞에가 기울기, 뒤에가 절편)
# cp.polyfit(x,y,차수)

x = [1,2,3,4]
y = [1,2,3,4]

# 기울기가 1이 나오도록 설정해본다.
gg = cp.polyfit(x,y,1)
print(gg)
# [1.00000000e+00 8.25508711e-16]
# 기울기 1, 절편도 사실상 0 (e-16 때문)

