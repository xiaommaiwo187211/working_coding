import pandas as pd 
df = pd.read_csv('data_502235.csv',index_col=0)

from sklearn.model_selection import train_test_split
import lightgbm as lgb

cols = ['pre_qtty_rolling1', 'pre_qtty_rolling7',
       'pre_qtty_rolling30', 'pre_qtty_rolling180', 'pre_netprice_rolling1',
       'pre_netprice_rolling7', 'pre_netprice_rolling30',
       'pre_netprice_rolling180', 'pre_uv_rolling1', 'pre_uv_rolling7',
       'pre_uv_rolling30', 'pre_uv_rolling180', 'pre_stock_qtty_rolling1',
       'pre_stock_qtty_rolling7', 'pre_stock_qtty_rolling30',
       'pre_stock_qtty_rolling180', 'netprice',
       'stock_qtty', 'day_of_year_fourier_cos_6', 'predicted_trend_train_1',
       'cid3', 'day_of_year_fourier_sin_2', 'day_of_year_fourier_sin_4',
        'predicted_trend_validation_1',
       'day_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3',
       'week_of_year_fourier_sin_3', 'week_of_year_fourier_sin_2',
       'day_of_year_fourier_sin_6', 'base_qtty', 'day_of_year_fourier_cos_1',
       'day_of_year_fourier_sin_5', 'week_of_year_fourier_sin_1',
       'day_of_year_fourier_cos_7', 'day_of_year_fourier_sin_1',
       'day_of_year_fourier_cos_3', 'baseprice', 'days_flag',
       'week_of_year_fourier_cos_2', 'day_of_year_fourier_sin_7',
       'day_of_year_fourier_cos_5', 'day_of_year_fourier_cos_4',
       'week_of_year_fourier_cos_1', 'day_of_year_fourier_cos_2']

train_x, test_x, train_y,  test_y = train_test_split(df[cols], df[['qtty']], test_size=0.2)

train_x[['cid3']] = train_x[['cid3']].astype('category')
test_x[['cid3']] = test_x[['cid3']].astype('category')
train_y[['qtty']] = train_y[['qtty']].astype('float')
test_y[['qtty']] = test_y[['qtty']].astype('float')

model = lgb.LGBMRegressor(n_jobs=1)
model.fit(train_x, train_y['qtty'])

from sklearn.metrics import mean_absolute_error
loss = mean_absolute_error(model.predict(test_x),test_y)
from sklearn.model_selection import GridSearchCV

params = {"num_leaves":[3,5,7,9],"max_depth":[10,13,50]}

model = lgb.LGBMRegressor(n_jobs=1)
gdmodel = GridSearchCV(model,params)
gdmodel_final = gdmodel.fit(train_x, train_y['qtty'])
# best_params_, best_score_

## next step lime and shap

from lime import lime_tabular as lmt
#### 将训练集数据的分布输入，这样在测试集上扰动时知道扰动的范围，虽然只需扰动即可，那不需要范围呀。但若是分类变量，肯定是要知道类别数目。
lmt_explain = lmt.LimeTabularExplainer(train_x.values, mode = 'regression', training_labels = train_x.columns,feature_selection = 'lasso_path')
#### 输出
exp =lmt_explain.explain_instance(test_x.values[93],gdmodel_final.predict)
exp.show_in_notebook()

import shap
feature_names = list(X_train.columns)
### 只需要指定需要被解释的模型，然后给出X_test即可。
explainer = shap.TreeExplainer(model_final)
shap_values_test = explainer.shap_values(X_test)
ind = test_df.query("item_sku_id == '3280890' and  dt== '2018-05-26'").index[0]
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values_test, feature_names=feature_names)
shap.summary_plot(shap_values_test, X_test)