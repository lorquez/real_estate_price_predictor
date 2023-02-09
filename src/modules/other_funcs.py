import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
import re

def fotocasa_m2_scraping(df):
  headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
  r = requests.get('https://www.fotocasa.es/indice-precio-vivienda/madrid-capital/todas-las-zonas', headers=headers)
  soup = BeautifulSoup(r.content, 'html.parser')
  elem = soup.find_all('table', class_='table-price')

  price_m2_df = pd.DataFrame()
  price_m2_df_final = pd.DataFrame()
  price_m2_ls = [elem[0].find_all('td')[i].text for i,x in enumerate(elem[0].find_all('td'))]


  for index,value in enumerate(price_m2_ls):
    if (index % 3 == 0 or index == 0):
        price_m2_df['district'] = [value]
        price_m2_df['price_m2_ft'] = price_m2_ls[index+1]
        price_m2_df['mean_price'] = price_m2_ls[index+2]
        price_m2_df_final = pd.concat([price_m2_df,price_m2_df_final])

  price_m2_df_final.loc[:,'mean_price'] = price_m2_df_final['mean_price'].str.replace(r' €|\.','', regex=True).astype('float')
  price_m2_df_final.loc[:,'price_m2_ft'] = price_m2_df_final['price_m2_ft'].str.replace(r' €/m²|\.','', regex=True).astype('float')
  price_m2_df_final.loc[price_m2_df_final['district'] == 'Fuencarral - El Pardo','district'] = 'Fuencarral'
  price_m2_df_final.loc[price_m2_df_final['district'] == 'Moncloa - Aravaca','district'] = 'Moncloa'
  price_m2_df_final.loc[price_m2_df_final['district'] == 'Chamartín','district'] = 'Chamartin'

  display(price_m2_df_final.reset_index(drop=True))

  # Fix Moratalaz
  r = requests.get('https://www.fotocasa.es/indice-precio-vivienda/madrid-capital/moratalaz', headers=headers)
  soup = BeautifulSoup(r.content, 'html.parser')
  elem = soup.find_all('div', class_='b-detail_title')
  
  price_m2_moratalaz = int(re.sub(r'<div class="b-detail_title">| €(/m²)?</div>|\.','',str(elem[0])))
  mean_price_moratalaz = int(re.sub(r'<div class="b-detail_title">| €(/m²)?</div>|\.','',str(elem[1])))
  
  df_with_pricem2 = df.merge(price_m2_df_final, how='left', on='district')

  df_with_pricem2.loc[df_with_pricem2['district'] == 'Moratalaz','price_m2_ft'] = price_m2_moratalaz
  df_with_pricem2.loc[df_with_pricem2['district'] == 'Moratalaz','mean_price'] = mean_price_moratalaz

  return df_with_pricem2

#-------------------------------------------
def iqr_outliers(df, features=['']):
  try:
    for feature in features:
      q1 = np.percentile(df[feature],25,interpolation='midpoint')
      q3 = np.percentile(df[feature],75,interpolation='midpoint')

      iqr = q3-q1
      upper = q3+1.5*iqr
      lower = q1-1.5*iqr

      thresholds = {'upper': upper, 'lower': lower}

      print(f"{feature}: upper = {thresholds['upper']}, lower = {thresholds['lower']}")
      df = df[(df[feature] <= thresholds['upper']) & (df[feature] >= thresholds['lower'])].reset_index(drop=True)
    return df
  except:
    print('Error')
    return None