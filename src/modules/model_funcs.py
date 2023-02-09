import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, RepeatedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import shap
import pickle
import scipy.stats as stats
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor


# FUNCIONES PARA EL MODELADO ------------------------------------

def model_evaluation(test, prediction, model, typedf):
  # Interesante para entender las métricas: https://sitiobigdata.com/2018/08/27/machine-learning-metricas-regresion-mse/

  #test = np.expm1(test)
  #prediction = np.expm1(prediction)

  r2 = metrics.r2_score(test,prediction).round(4)
  mse = metrics.mean_squared_error(test,prediction).round(4)
  rmse = metrics.mean_squared_error(test,prediction,squared=False).round(4)
  mae = metrics.mean_absolute_error(test,prediction).round(4)

  row = {'model': [model], 'r2': [r2], 'mse': [mse], 'rmse': [rmse], 'mae': [mae], 'type': [typedf]}
  eval = pd.DataFrame(row, columns = ['model','r2','mse','rmse','mae','type'])

  return eval

#---------------------------------------------------------------

def cross_validate_models(X,y,model):
  #Validación cruzada con cross_validate
  cv = KFold(n_splits=5, shuffle=True, random_state=42) #RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  scores = cross_validate(model, X, y, scoring=('r2', 'neg_mean_squared_error','neg_mean_absolute_error'), cv=cv, n_jobs=-1)
  print(scores)

  r2 = scores['test_r2'].mean().round(4)
  mse = abs(scores['test_neg_mean_squared_error']).mean().round(4)
  rmse = np.sqrt(abs(scores['test_neg_mean_squared_error']).mean().round(4))
  mae = abs(scores['test_neg_mean_absolute_error']).mean().round(4)

  row = {'model': [model], 'r2': [r2], 'mse': [mse], 'rmse': [rmse], 'mae': [mae]}
  eval = pd.DataFrame(row, columns = ['model','r2','mse','rmse','mae'])

  return eval

#---------------------------------------------------------------

def search_cv(X_train, y_train, model, parameters, cv=5, scoring='neg_mean_squared_error', cvtype='grid'):
  if cvtype == 'grid':
    print('GridSearchCV')
    grid = GridSearchCV(model,
                          parameters,
                          cv = cv,
                          scoring = scoring,
                          n_jobs = -1,
                          verbose=True)
  else:
    print('RandomizedSearchCV')
    grid = RandomizedSearchCV(model,
                          parameters,
                          cv = cv,
                          scoring = scoring,
                          n_jobs = -1,
                          verbose=True)

  grid.fit(X_train, y_train)
  params = {'best_score (RMSE)': np.sqrt(abs(grid.best_score_)).round(4), 'best_params': grid.best_params_}

  print(params)
  return grid

#---------------------------------------------------------------

def test_model_predictions(df_model,df_original,model,index=-1):

  #df_original = df_model.reset_index(drop=True)
  df_original = df_original[df_original.index.isin(df_model.index)].reset_index(drop=True)
  df_model = df_model.reset_index(drop=True)

  if index == -1:
    index = random.randint(1, len(df_model.index))
  #real_log_price = df_model[df_model.index == index]['price_log'].values
  real_price = df_original[df_original.index == index]['price'].values

  try:
    predicted_price = model.predict(df_model[df_model.index == index].drop('price',axis=1)).astype('float')
  except:
    predicted_price = model.predict(df_model[df_model.index == index]).astype('float')

  #log_diff = real_log_price - predicted_price
  diff = real_price - predicted_price

  row = {'index': index, 'real_price': real_price, 'predicted_price': predicted_price, 'difference': diff}
  test = pd.DataFrame(row, columns = ['index','real_price','predicted_price','difference'])
  #print(df[df.index == index])

  return test

#---------------------------------------------------------------

def plot_error(real_price,predicted_price):
  abs_error = np.abs(real_price-predicted_price)
  df_to_plot = pd.DataFrame({'real_price':real_price,'predicted_price':predicted_price,'abs_error':abs_error})
  df_to_plot = df_to_plot.sort_values('real_price').reset_index()
  fig, ax = plt.subplots(1,1,figsize = (20,15))
  ax.plot(df_to_plot['real_price'],color='red',alpha=.7,label='real price')
  ax.plot(df_to_plot['predicted_price'],color='blue',alpha=.4,label='predicted price')
  ax.plot(df_to_plot['abs_error'],color='green',alpha=.7,label='abs error')
  ax.set_title('Real price vs. Price prediction', fontsize=20)
  ax.set(ylabel='price')
  ax.legend()

#---------------------------------------------------------------

def plot_matching(y_test,y_hat):
  colors = ['#E74C3C','#3498DB']
  fig, ax = plt.subplots(1,1,figsize = (20,15))
  sns.scatterplot(x=y_test,y=y_hat, alpha=.2, ax=ax)
  sns.regplot(x=y_test, y=y_hat, ax=ax, scatter=False, line_kws={'color': colors[0]}, scatter_kws={'color': colors[1]}, ci=100)
  ax.set(xlabel='real values', ylabel='predicted values')
  ax.set_title('Matching predictions with actual price values',fontsize=22)

  return None

#---------------------------------------------------------------

def plot_learning(results,eval):
  try:
    epochs = len(results['validation_0'][eval])
  except:
    epochs = len(results['training'][eval])
  x_axis = range(0,epochs)

  fig, ax = plt.subplots(figsize=(15,5))

  ax.set_title(f"Learning curve ({eval})",fontsize=15)
  #sns.lineplot(y=np.expm1(results['validation_0'][eval]),x=x_axis, ax=ax, label='Train');
  #sns.lineplot(y=np.expm1(results['validation_1'][eval]),x=x_axis, ax=ax, label='Test');
  try:
    sns.lineplot(y=results['validation_0'][eval],x=x_axis, ax=ax, label='Train');
    sns.lineplot(y=results['validation_1'][eval],x=x_axis, ax=ax, label='Test');
  except:
    sns.lineplot(y=results['training'][eval],x=x_axis, ax=ax, label='Train');
    sns.lineplot(y=results['valid_1'][eval],x=x_axis, ax=ax, label='Test');

  plt.xlabel('Epochs');
  plt.ylabel(eval);
  return None

#---------------------------------------------------------------

def compare_prediction(df, X_test, yhat,by='propertyType'):
  yhat = pd.DataFrame(yhat)
  df = df[df.index.isin(X_test.index)]

  df = pd.merge(df,yhat,how='inner',left_index=True, right_index=True)
  df.rename(columns={0: "yhat"},inplace=True)

  df['diff'] = df['price']-df['yhat']
  #print(df)
  fig, ax = plt.subplots(figsize=(25,15))
  ax.set_title(f"Highest difference (price by {by})",fontsize=15)
  sns.scatterplot(y=df['diff'],x=df[by], ax=ax, hue=df[by]);
  sns.boxplot(y=df['diff'],x=df[by], ax=ax, hue=df[by]);

  return None

#---------------------------------------------------------------

def pickle_dump(model_name, model):
  with open('%s.pkl' % model_name,'wb') as pk:
    pickle.dump(model, pk)

  return print(f'{model_name} saved as pickle file.')


#---------------------------------------------------------------

def test_regression_models(df,mod='all'):

  def plot_error(real_price,predicted_price):
    abs_error = np.abs(real_price-predicted_price)
    df_to_plot = pd.DataFrame({'real_price':real_price,'predicted_price':predicted_price,'abs_error':abs_error})
    df_to_plot = df_to_plot.sort_values('real_price').reset_index()
    fig, ax = plt.subplots(1,1,figsize = (20,15))
    ax.plot(df_to_plot['real_price'],color='red',alpha=.7,label='real price')
    ax.plot(df_to_plot['predicted_price'],color='blue',alpha=.4,label='predicted price')
    ax.plot(df_to_plot['abs_error'],color='green',alpha=.7,label='abs error')
    ax.set_title('Real price vs. Price prediction', fontsize=20)
    ax.set(ylabel='price')
    ax.legend()

    return None

  def plot_matching(y_test,y_hat):
    colors = ['#E74C3C','#3498DB']
    fig, ax = plt.subplots(1,1,figsize = (20,15))
    sns.scatterplot(x=y_test,y=y_hat, alpha=.2, ax=ax)
    sns.regplot(x=y_test, y=y_hat, ax=ax, scatter=False, line_kws={'color': colors[0]}, scatter_kws={'color': colors[1]}, ci=100)
    ax.set(xlabel='real values', ylabel='predicted values')
    ax.set_title('Matching predictions with actual price values',fontsize=22)

    return None

  evaluation = pd.DataFrame()
  dict_models = {}
  if mod == 'all':
    models = ['Linear Regression','Decision Tree Regression','Random Forest Regression','AdaBoost Regression','Gradient Boosting Regression','XGBoost Regression','CatBoost Regression','LightBM Regression']
  else:
    models = [mod]

  X = df.drop(['price'],axis=1)
  y = df['price']

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)

  for model in models:
    if model == 'Linear Regression':
      LR_model = LinearRegression()
      LR_model.fit(X_train, y_train);
      y_train_predicted = LR_model.predict(X_train)
      y_hat_predicted = LR_model.predict(X_test)
      dict_models[model] = LR_model
    elif model == 'Decision Tree Regression':
      DTR_model = DecisionTreeRegressor()
      DTR_model.fit(X_train, y_train)
      y_train_predicted = DTR_model.predict(X_train)
      y_hat_predicted = DTR_model.predict(X_test)
      dict_models[model] = DTR_model
    elif model == 'Random Forest Regression':
      RFR_model = RandomForestRegressor()
      RFR_model.fit(X_train, y_train)
      y_train_predicted = RFR_model.predict(X_train)
      y_hat_predicted = RFR_model.predict(X_test)
      dict_models[model] = RFR_model
    elif model == 'AdaBoost Regression':
      DTR_model = DecisionTreeRegressor()
      DTR_model.fit(X_train, y_train)
      ADA_model = AdaBoostRegressor(base_estimator = DTR_model)
      ADA_model.fit(X_train, y_train)
      y_train_predicted = ADA_model.predict(X_train)
      y_hat_predicted = ADA_model.predict(X_test)
      dict_models[model] = ADA_model
    elif model == 'Gradient Boosting Regression':
      GBR_model = GradientBoostingRegressor()
      GBR_model.fit(X_train, y_train)
      y_train_predicted = GBR_model.predict(X_train)
      y_hat_predicted = GBR_model.predict(X_test)
      dict_models[model] = GBR_model
    elif model == 'XGBoost Regression':
      XGB_model = xgb.XGBRegressor(objective ='reg:squarederror', verbosity = 0)
      XGB_model.fit(X_train, y_train)
      y_train_predicted = XGB_model.predict(X_train)
      y_hat_predicted = XGB_model.predict(X_test)
      dict_models[model] = XGB_model
    elif model == 'CatBoost Regression':
      CB_model = CatBoostRegressor()
      CB_model.fit(X_train, y_train,silent=True)
      y_train_predicted = CB_model.predict(X_train)
      y_hat_predicted = CB_model.predict(X_test)
      dict_models[model] = CB_model
    elif model == 'LightBM Regression':
      LGBM_model = LGBMRegressor()
      LGBM_model.fit(X_train, y_train)
      y_train_predicted = LGBM_model.predict(X_train)
      y_hat_predicted = LGBM_model.predict(X_test) 
      dict_models[model] = LGBM_model

    evaluation = pd.concat([evaluation,model_evaluation(y_train,y_train_predicted,model,'train')]).reset_index(drop=True)
    evaluation = pd.concat([evaluation,model_evaluation(y_test,y_hat_predicted,model,'test')]).reset_index(drop=True)
    

    if mod != 'all':
      print(f'{model}\n')
      display(evaluation)
      plot_error(y_test,y_hat_predicted)
      plot_matching(y_test,y_hat_predicted)
  
  if mod == 'all':
    print('TOTAL - MODEL BENCHMARKING\n')
    display(evaluation)
    print('\nTRAIN - MODEL BENCHMARKING\n')
    display(evaluation[evaluation['type'] == 'train'].sort_values(by=['rmse']).reset_index(drop=True))
    print('\nTEST - MODEL BENCHMARKING\n')
    display(evaluation[evaluation['type'] == 'test'].sort_values(by=['rmse']).reset_index(drop=True))
  return dict_models

#---------------------------------------------------------------

def target_encode(df,features=['']):
  categorical_columns = df.select_dtypes(exclude=["number"]).columns
  te_encoder = ce.TargetEncoder()

  f = all(elem in categorical_columns for elem in features)
  if f == True:
    categorical_columns = features
  
  te_encoder.fit(df[categorical_columns], df['price'])
  df_te = df.drop(categorical_columns, axis=1).join(te_encoder.transform(df[categorical_columns]))

  return df_te

#---------------------------------------------------------------

def one_hot_encode(df,features=[''],dropbinary = False,download=False):
  categorical_columns = df.select_dtypes(exclude=["number"]).columns
  f = all(elem in categorical_columns for elem in features)
  if f == True:
    categorical_columns = features

  if dropbinary == True:
    oneHotEncoder = OneHotEncoder(drop='if_binary')
  else:
    oneHotEncoder = OneHotEncoder()

  feature_array = oneHotEncoder.fit_transform(df[categorical_columns]).toarray()
  feature_labels = oneHotEncoder.categories_
  feature_labels = np.array(np.hstack(feature_labels),dtype="object").flatten()

  df_ohe = pd.DataFrame(feature_array, columns = feature_labels)
  df_onehot_encoding = pd.concat([df,df_ohe],axis=1).drop(categorical_columns, axis=1)

  if download == True:
    pickle_dump('oneHotEncoder',oneHotEncoder)

  return df_onehot_encoding

#---------------------------------------------------------------

def vif_info(df):
  vif_info = pd.DataFrame()
  vif_info['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
  vif_info['Column'] = df.columns
  vif_info.sort_values('VIF', ascending=False)

  return vif_info


