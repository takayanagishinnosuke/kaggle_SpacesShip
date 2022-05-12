#%%
from copyreg import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from requests import head
import seaborn as sns
import pandas_profiling
from ipywidgets import HTML, Button, widgets
from pycaret.classification import *
from sqlalchemy import true
# %%
"""データの読み込み"""
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

# %%
"""データの確認"""
train_data.head()
train_data.info()
train_data.isnull().sum()
train_data.describe()
# %%
"""データのレポートで可視化"""
train_data.profile_report()

#レポートからルームサービスやフードコートやスパの利用と相関が無いのではと推測
#一旦、それらのカラムを削除して試してみる事にする


# %%
"""いらないカラムの削除と欠損値の穴埋め"""

"""まずはデータを結合してオールデータにする"""
data_all = pd.concat([train_data, test_data], sort=True)
data_all
# %%
"""いらなそうなカラムの削除"""
drop_columns = ['Name', 'ShoppingMall', 'Spa' , 'RoomService' ,'FoodCourt', 'Cabin']

data_all.drop(drop_columns, axis=1, inplace=True)
data_all
# %%
"""VRdeckも消す"""
data_all.drop('VRDeck', axis=1, inplace=True)
data_all
# %%
"""Ageの穴埋め"""
data_all['Age'].fillna(data_all['Age'].median(), inplace=True)
data_all['Age'].isnull().sum()
# %%
"""CryoSleepの穴埋め"""
data_all['CryoSleep'].fillna(data_all['CryoSleep'].mode().iloc[0] , inplace=True)
data_all['CryoSleep'].isnull().sum()
# %%
data_all.isnull().sum()
# %%
"""Destinationの穴埋め"""
data_all['Destination'].fillna(data_all['Destination'].mode().iloc[0] , inplace=True)
data_all['Destination'].isnull().sum()

# %%
"""HomePlanetの穴埋め"""
data_all['HomePlanet'].fillna(data_all['HomePlanet'].mode().iloc[0] , inplace=True)
data_all['HomePlanet'].isnull().sum()
# %%
"""VIPの穴埋め"""
data_all['VIP'].fillna(data_all['VIP'].mode().iloc[0] , inplace=True)
data_all['VIP'].isnull().sum()
# %%
"""教師データとテストデータを分離"""
train_data = data_all[:len(train_data)]
test_data = data_all[len(train_data):]

#%%
"""プロットしてみる"""
sns.countplot(x='CryoSleep', data=train_data, hue='Transported')


#仮説としてCryoSleepがTrue.つまりコールドスリープに入っていた人間は、時空に飛ばされていない事が伺える

# %%
"""
ここまで基本的な前処理を終了
Pycaretを起動する
"""

clf1 = setup(train_data, 
            target='Transported', #目的変数
            numeric_imputation= 'median', #数値データ欠損値の穴埋め{平均:mean 中央:median} 
            categorical_imputation='mode', #カテゴリ変数の穴埋め{mode:最頻値}
            )

# %%
'''モデルを比較'''
compare_models()

# 一旦、gbcを採用する事にする

# %%
"""モデルの定義"""
model = create_model('gbc')
print(model)
# %%
"""ハイパーパラメータの最適化"""
tuned_gbc = tune_model(model, optimize='Accuracy')
# %%
"""精度の可視化"""
evaluate_model(tuned_gbc)
# %%
"""モデルの確定"""
final_gbc = finalize_model(tuned_gbc)
print(final_gbc)
# %%
"""予測を実行させてみる"""
predict_ans = predict_model(final_gbc, data=test_data)
# %%
"""確認"""
predict_ans

"""
<結果>
# スコアがとてもよくない…
# 再度EDAをおこなうこととする
"""

# %%
"""!!!!---------EDAパート2----------!!!!"""

"""もっかいデータの読み込み"""
train_data2 = pd.read_csv('data/train.csv')
test_data2 = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

#%%
"""データの結合オールデータにする"""
data_all2 = pd.concat([train_data2, test_data2], sort=True)
data_all2

# %%
"""いらないカラムを消す"""
data_all2.drop('Name', axis=1, inplace=True)
# %%
data_all2
# %%
"""PassengerIDの小番号に_01と_02があるのを発見。まずは分けてみる"""
data_all2['ID01'] = data_all2['PassengerId'].str.split('_',expand=True)[0]
data_all2['ID02'] = data_all2['PassengerId'].str.split('_',expand=True)[1]

data_all2.head()
data_all2.info()

#%%
"""id01と02がobjectなので数字に変換"""
data_all2['ID01']=data_all2['ID01'].astype(int)
data_all2['ID02']=data_all2['ID02'].astype(int)

data_all2.info()

"""pltしてみる"""
sns.countplot(data_all2['ID02'], hue=data_all2['Transported'])
# 少し関係しそうな予感がみえる…


#%%
"""Ageの穴埋めと整数に変換しとく"""
data_all2['Age'].fillna(data_all2['Age'].median(), inplace=True)
data_all2['Age']= data_all2['Age'].astype(int)

data_all2.info()
#%%
"""CryoSleepの穴埋め"""
data_all2['CryoSleep'].fillna(data_all2['CryoSleep'].mode().iloc[0] , inplace=True)

"""Destinationの穴埋め"""
data_all2['Destination'].fillna(data_all2['Destination'].mode().iloc[0] , inplace=True)

"""HomePlanetの穴埋め"""
data_all2['HomePlanet'].fillna(data_all2['HomePlanet'].mode().iloc[0] , inplace=True)

"""VIPの穴埋め"""
data_all2['VIP'].fillna(data_all2['VIP'].mode().iloc[0] , inplace=True)

#%%
"""Cabin"""
data_all2['Cabin'].head(10)
# 客室は3区画で別れているのかな？それぞれ分けてみる

data_all2['Cabin1']= data_all2['Cabin'].str.split('/',expand=True)[0]
data_all2['Cabin2']= data_all2['Cabin'].str.split('/',expand=True)[1]
data_all2['Cabin3']= data_all2['Cabin'].str.split('/',expand=True)[2]

data_all2

#%%
"""plotしてみるか。まずはCabin1から。"""
sns.countplot(data_all2['Cabin1'], hue=data_all2['Transported'])
##FとGが圧倒的。庶民の客室って感じか?
data_all2['Cabin1'].isnull().sum() #299の欠損がある

data_all2['Cabin1'].fillna('F', inplace=True) #欠損はFで埋めとく


#%%
"""次にCabin2。"""
# sns.countplot(data_all2['Cabin2'], hue=data_all2['Transported'])
## plotは細かいな。ただの客室番号か？ カウントしてみる
data_all2['Cabin2'].value_counts()

data_all2['Cabin2'].fillna('2000', inplace=True) #欠損は架空の部屋番号で埋めとく
data_all2['Cabin2'].isnull().sum()

#%%
"""Cabin2のつづき"""
data_all2['Cabin2']=data_all2['Cabin2'].astype(int) # (全部数字にして…)

plt.figure(figsize=(16,9)) #表示倍率変えられるみたい
plt.hist(data_all2['Cabin2'],bins=200,color='red') #縮尺はbinsで指定
"""0~300,300~600,600~1500,1500~2000 みたいなグループで分けられそう"""

#%%
"""Cabin2のつづき(パート2)"""
data_all2.loc[(0<=data_all2['Cabin2'])&(data_all2['Cabin2']<300),'Cabin2']=1
data_all2.loc[(300<=data_all2['Cabin2'])&(data_all2['Cabin2']<600),'Cabin2']=2
data_all2.loc[(600<=data_all2['Cabin2'])&(data_all2['Cabin2']<1500),'Cabin2']=3
data_all2.loc[(1500<=data_all2['Cabin2'])&(data_all2['Cabin2']<2000),'Cabin2']=4

data_all2['Cabin2'].value_counts()

#%%
## 1のグループが多いので、欠損は1のグループで統一することにする
data_all2.loc[data_all2['Cabin2']>=2000,'Cabin2']=1

data_all2['Cabin2'].value_counts()

"""もっかいPlotしてみる"""
sns.countplot(data_all2['Cabin2'],hue=data_all2['Transported'])
# (いいかんじでグループ分けできた)

# %%
"""最後にCabin3"""
sns.countplot(data_all2['Cabin3'], hue=data_all2['Transported'])
# Sクラスのほうが若干飛ばされてる感じ? うーん。

data_all2['Cabin3'].fillna(data_all2['Cabin3'].mode().iloc[0] , inplace=True)
# (最頻値で埋めておく)

data_all2['Cabin3'].isnull().sum()

# %%
'''
さて…FoodCourtとRoomService、Shoopingmall、Spa、VRDeck
これらの娯楽データは何か特徴量になりえるのか???
'''

data_all2.head()
data_all2.isnull().sum()

#%%
"""とりあえず、全てmedianで埋めておこう"""
data_all2['FoodCourt'].fillna(data_all2['FoodCourt'].median(), inplace=True)
data_all2['RoomService'].fillna(data_all2['RoomService'].median(), inplace=True)
data_all2['ShoppingMall'].fillna(data_all2['ShoppingMall'].median(), inplace=True)
data_all2['Spa'].fillna(data_all2['Spa'].median(), inplace=True)
data_all2['VRDeck'].fillna(data_all2['VRDeck'].median(), inplace=True)

data_all2.isnull().sum()


# %%
"""娯楽データを合計して特徴量を作ってみる"""
data_all2['Pay'] = data_all2['FoodCourt']+data_all2['RoomService']+data_all2['ShoppingMall']+data_all2['Spa']+data_all2['VRDeck']

data_all2['Pay'].head()

#%%
"""カウントしてみる"""
data_all2['Pay'].value_counts()
#(だいたいお金かけてない…ケチな人たち多いのかな…)

"""ヒストグラムで表示"""
plt.hist(data_all2['Pay'],bins=100,color='blue') #縮尺はbinsで指定
#(うーん,,,,浪費しない人がおおすぎるな…このままでいいか)

# %%
data_all2.info()
data_all2
# %%
"""改めて使わないデータを消しておく"""
data_all2 = data_all2.drop(['Cabin','PassengerId'], axis=1)

# %%
data_all2.head()
# %%
"""文字データをエンコーディングしておく"""
from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()

data_all2['Destination']= lbl.fit_transform(data_all2['Destination'])
data_all2['HomePlanet']= lbl.fit_transform(data_all2['HomePlanet'])
data_all2['Cabin1']= lbl.fit_transform(data_all2['Cabin1'])
data_all2['Cabin3']= lbl.fit_transform(data_all2['Cabin3'])

# %%
data_all2.head()
# %%
"""さらに、いらなそうなデータを消しておこう"""
data_all2 = data_all2.drop(['RoomService','ShoppingMall','Spa','VRDeck'], axis=1)
# %%
data_all2.head() ##すっきりした
# %%
"""学習用とテスト用で分けましょう！"""
train_data2 = data_all2[:len(train_data2)]
test_data2 = data_all2[len(train_data2):]

# %%
train_data2 ##確認OK
#%%
#嘘、foodcourt落としてない
train_data2 = train_data2.drop(['FoodCourt'], axis=1)

train_data2
##(完璧！)

# %%
"""テストデータの説明変数は消しておく"""
test_data2 = test_data2.drop(['Transported'], axis=1)
test_data2
## 確認OK

# %%
"""PycaretでAutoML"""
clf2 = setup(train_data2, 
            target='Transported', #目的変数
            numeric_imputation= 'median', #数値データ欠損値の穴埋め{平均:mean 中央:median} 
            categorical_imputation='mode', #カテゴリ変数の穴埋め{mode:最頻値}
            )
# %%
'''モデルを比較'''
compare_models()
#(LightGBMがいい値出てる)
#後のアンサンブルに向けて(lr)(svm)あたりをマーク
# %%
"""モデル定義"""
model2 = create_model('lightgbm')
print(model2)
# Ac 0.75 : Prec 0.875 : AUC 0.815    
# %%
"""ハイパーパラメータの最適化"""
tuned_lgbm = tune_model(model2,  choose_better = True)
# %%
"""予測させてみる"""
predict_ans = predict_model(tuned_lgbm, data=test_data2)
# %%
predict_ans.head()

#%%
"""学習済みモデルの保存"""
import pickle

with open('lgbm_model.pickle', mode='wb') as f:
  pickle.dump(tuned_lgbm, f)

# %%
"""精度のレポート"""
evaluate_model(tuned_lgbm)

"""""""""""""""
<2回目の学習結果>
id01とAgeが効いてるっぽい
うーん？？？学習ではAccuracy:0.7635
微妙な雰囲気…とりあえず1回提出してみる…
"""""""""""""""

# %%
"""Submitファイルの作成"""
sub['Transported'] = predict_ans['Label']
sub.to_csv('submission0.csv', index=False)
sub.head()

# %%
"""""""""

<Submitの結果>
- Score: 0.75099 
- 上位は 0.81を超えている

"""""""""
train_data2
# %%
"""全ての特徴量をintに,id01を削除?"""
train_data2['CryoSleep']= lbl.fit_transform(train_data2['CryoSleep'])
train_data2['Transported']= lbl.fit_transform(train_data2['Transported'])
train_data2['VIP']= lbl.fit_transform(train_data2['VIP'])
# train_data2 = train_data2.drop(['ID01'], axis=1)

# %%
train_data2.head(15)
# %%
"""再度LightGBMで学習"""
clf2 = setup(train_data2, 
            target='Transported', #目的変数
            )
# %%
compare_models()
# %%
"""モデル定義"""
model3 = create_model('lightgbm')
print(model3)
# %%
tuned_lgbm2 = tune_model(model3, optimize='Accuracy')
# %%
"""推論"""
predict_ans = predict_model(tuned_lgbm2, data=test_data2)
predict_ans.head()
# %%
"""精度のレポート"""
evaluate_model(tuned_lgbm2)
# %%
"""Submitファイルの作成パート2"""
sub['Transported'] = predict_ans['Label'].astype(bool)
sub.to_csv('submission1.csv', index=False)
sub.head()

"""
スコアが下がってしまった…
何がいけない?

1.ランダムサーチの回数が少なすぎた。(グリッドサーチにしてみる?)
2.特徴量を整数にした
3.パラメータが適切でない
4.特徴量が甘い
(正直、AutoMLだとこれまでのEDAが正しいのか正しくないのか分からん)
"""
# %%
"""
<再チャレンジ!!>
AutoMLに頼らず、オーダーでmodel構築してみる!!
"""
train_data2

#%%
"""train_data2をXとYで分ける"""
x_data = train_data2.drop(['Transported'], axis=1)
x_data

#%%
y_drop_col = ['Age','CryoSleep','Destination','HomePlanet','VIP','ID01','ID02','Cabin1','Cabin2','Cabin3','Pay']
y_data = train_data2.drop(y_drop_col, axis=1)

y_data.head()

# %%
"""テストデータも同じように用意"""
test_data2['CryoSleep']= lbl.fit_transform(test_data2['CryoSleep'])
test_data2['VIP']= lbl.fit_transform(test_data2['VIP'])
test_data2= test_data2.drop(['FoodCourt'],axis=1)
test_data2.head()
# %%
"""LightGBMで学習"""
"""まずは必要なライブラリをimport"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

#%%
"""学習用と検証用でsplitする"""
X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(
  x_data, y_data, test_size=0.3, random_state=0, stratify=y_data)

# X_data_train.head() #学習用の説明変数
# X_data_valid.head() #検証用の説明変数
# y_data_train.head() #学習用の目的変数
# y_data_valid.head() #検証用の目的変数

# %%
"""モデル作成"""
#モデル用にデータを格納する
lgb_data_train = lgb.Dataset(X_data_train,y_data_train)
lgb_data_eval = lgb.Dataset(X_data_valid,y_data_valid, reference=lgb_data_train)

"""パラメータの設定"""
params = {
  'objective': 'binary' ##目的変数は二値分類だよと設定
}

"""ハイパーパラメータの設定"""
model4 = lgb.train(params, lgb_data_train, valid_sets=lgb_data_eval,verbose_eval=10,num_boost_round=1000,early_stopping_rounds=10)
#(パラメータ, 学習データ, valid_sets=検証データ, verbose_eval=10回の学習毎に画面に表示するよ, num_boost_round=勾配Boostingを何回, early_stopping_rounds=過学習を防ぐために様子見何回するか)

#%%
"""予測してみる"""
y_data_pred = model4.predict(test_data2, num_iteration=model4.best_iteration)
#(予測データ, num_iteration=作成モデル名.best_iteration(ベストな探索回数で予測してね))

y_data_pred
# %%
"""0.5以上で1と予測したとみなす"""
y_data_pred = (y_data_pred > 0.5).astype(int)
y_data_pred[:5]

# %%
"""訓練データと検証データの正解率を確認"""
print(metrics.accuracy_score(y_data_train['Transported'],np.round(model4.predict(X_data_train))))
#1:訓練データの正解率は0.8036

print(metrics.accuracy_score(y_data_valid['Transported'],np.round(model4.predict(X_data_valid))))
#1:検証データ正解率は0.7553

# %%
"""Submitファイルの作成パート3"""
df_pre = pd.DataFrame(y_data_pred, columns=['Transported']).astype(bool)
## カラムを直して、Bool型になおして…
# df_pre

sub['Transported'] = df_pre['Transported']
#提出用ファイルのsubとくっつけて…

sub.to_csv('submission3.csv', index=False)
sub.head()
## csvで保存と…

"""

スコア!!
0.75146 
いちおう過去最高スコア

"""
#%%
#pickleで保存しておく
# import pickle

# with open('lgbm_model.pkl', mode='wb') as f:
#   pickle.dump(model4, f)

# %%
