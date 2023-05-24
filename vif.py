import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X = pd.read_csv('woe_data2.csv')
y=X['STATUS']
x = X.drop(['STATUS','order_id'],axis=1)

import statsmodels.api as sm
x[np.isnan(x)] = 0
x[np.isinf(x)] = 0
# 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
print(x.columns)
print(vif)

vif = pd.DataFrame(vif)
vif.to_csv('vif.csv')
