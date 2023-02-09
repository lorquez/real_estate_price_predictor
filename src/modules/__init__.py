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