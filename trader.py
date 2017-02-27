# coding: utf-8

# 統計学基礎知識 : coef_ 回帰係数, intercept_ 切片, score 決定係数
from sklearn import linear_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

trader_data = pd.read_csv(".csvファイルまでのpath")
# 正規化
trader_data_normalization = trader_data.apply(lambda x : (x-np.mean(x))/(np.max(x) - np.min(x)))
# DataFrameで返ってくる
trader_data_normalization.head()
# 指定したヘッダー以外を説明変数として扱う
except_selector_data = trader_data.drop("selector", axis = 1)
X = except_selector_data.as_matrix()
# 目的変数
Y = trader_data["selector"].as_matrix()

print(X)
clf = linear_model.LinearRegression()
clf.fit(X, Y)

# 偏回帰係数
print(pd.DataFrame({"名前": except_selector_data.columns,
	"偏回帰係数" : np.abs(clf.coef_)}).sort_values(by = "偏回帰係数"))
# 切片は初期値みたいなもの
print("切片 : ", clf.intercept_)
print("決定係数 : ", clf.score(X, Y))
学習データを保存する
joblib.dump(clf, "保存したいディレクトリまでのpath")
print("ok!")