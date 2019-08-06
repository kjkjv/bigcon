# =============test1_pledge,temp_cnt============================================

import pandas as pd
import numpy as np

df = pd.read_csv('train_combat.csv')
print(df)
df['temp_cnt']
df['temp_cnt'].isnull().sum()

r=df[['acc_id','temp_cnt']]
print(r)


label = pd.read_csv('train_label.csv')
label.drop(columns=['amount_spent'],inplace = True)
print(label)

r2 = pd.merge(r,label,how='inner',on='acc_id',right_index=False)
print(r2)

r2.drop(columns=['acc_id'],inplace=True)
print(r2)

a=df.corr()
tkdwjd = pd.merge(r,label,how='inner',on='acc_id')

a = tkdwjd.corr()

todwhs = []
wnrdma = []
for i in label['survival_time'] :
    if i > 63 :
        todwhs.append(i)
    else:
        wnrdma.append(i)
print(todwhs)
print(wnrdma)



# =====================활동이 활발한 혈맹에 소속된 캐릭일수록 이탈이 적은가?==========================

# 1. 활발한 혈맹을 조사
# 2. 이탈률을 조사
# 3. 위 두개를 합쳐봄

df = pd.read_csv('C:/Users/CPB06GameN/Documents/GitHub/bigcon/train_pledge.csv')
df.columns
df['pledge_id']

# 혈맹 아이디로 모아보자
pledge_mean = df.groupby('pledge_id').mean()
pledge_mean.columns

# 여기서 잠깐, 활발하다는 기준은?
# 1) 접속된 혈맹 캐릭터 수가 많다
# 2) 전투에 참여한 캐릭터가 많다
# 3) 혈맹 전투 횟수가 많다
# 4) 단발성 전투가 많다
# 5) 막피가 많다
# 6) 동일 혈맹 전투 횟수가 많다
# 7) 기타 전투 횟수
# 8) 전투 캐릭터 플레이 시간
# 9) 비전투 캐릭터 플레이 시간
# 결국 다 봐야하네;;;;;

pledge_mean.corr()

# 1)
pledge_mean['play_char_cnt']
pledge_mean['play_char_cnt'].sort_values(ascending=False)

# 혈맹의 갯수 21860 행
# 10개의 등급으로 나눕니다
# 각 등급에 2186개씩 들어감


# =========================================0801=========================================================
import pandas as pd



activity = pd.read_csv('C:/Users/CPB06GameN/Documents/GitHub/bigcon/train_activity.csv')
label = pd.read_csv('C:/Users/CPB06GameN/Documents/GitHub/bigcon/train_label.csv')
combat = pd.read_csv('C:/Users/CPB06GameN/Documents/GitHub/bigcon/train_combat.csv')

activity.columns

# activity와 label을 우선적으로 merge
meg_act_lab = pd.merge(label, activity, how = 'inner', on = 'acc_id')
print(meg_act_lab.head())
print(meg_act_lab.columns)
print(label.columns)
print(combat.columns)

# char_id 기준으로 groupby.sum() / activity에서 사용할 columns 추출
activity_group = activity.groupby('char_id').sum()[['playtime', 'npc_kill', 'solo_exp', 'party_exp',
                                                  'quest_exp','boss_monster', 'death', 'revive', 'exp_recovery',
                                                  'fishing','private_shop', 'game_money_change', 'enchant_count']]

# char_id 기준으로 groupby.sum() / combat에서 사용할 columns 추출
combat_group = combat.groupby('char_id').sum()[['pledge_cnt','random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
                                                  'same_pledge_cnt','etc_cnt','num_opponent']]

# char_id 기준 level 추출
combat_lv = combat.groupby('char_id').max()['level']


# char_id 를 label 에 추가하는 과정
selected_group = meg_act_lab[['acc_id', 'char_id', 'survival_time', 'amount_spent']]

# char_id를 기준으로 level 과 acc_id 를 추가하는 과정
# ( activity_group에 직접 추가를 해도 되지만, acc_id와 survival_time 이 필요하기 때문에 과정을 한번 더 거친다.)
selected_group2 = pd.merge(selected_group, combat_lv, how = 'inner', on = 'char_id')

# 이 부분에서 중복되는 것을 막기 위해 char_id를 기준으로 최대값들을 구해준다.
selected = selected_group2.groupby('char_id').max()

# 앞서 activity에서 사용할 colums을 추출한 데이터와 char_id를 기준으로 level, survival_time 등을 가져온 데이터를 merge
activity_group3 = pd.merge(activity_group, selected, how = 'inner', on = 'char_id')

combat_group1 = pd.merge(combat_group, selected, how = 'inner', on = 'char_id')