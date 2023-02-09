import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
#import category_encoders as ce
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bs4 import BeautifulSoup
import requests
import json
import re


# Definición de página
st.set_page_config(
    page_title="Check Mode",
    page_icon="👋",
)

# Directorio actual
CURR_DIR = os.getcwd()

# Definición de estilos
primaryColor = st.get_option("theme.primaryColor")
backgroundColor = st.get_option("theme.backgroundColor")
secondaryBackgroundColor = st.get_option("theme.secondaryBackgroundColor")
textColor = st.get_option("theme.textColor")

style = f"""
<style>
div.stButton > button:first-child {{ background-color: {primaryColor}; color: #FFF; border: 1px solid {textColor}; }}
div.stButton > button:first-child:hover {{ background-color: #FFF; color: {primaryColor}; border: 1px solid {textColor}; }}
div.stButton > button:first-child:focus {{ background-color: #FFF; color: {primaryColor}; border: 1px solid {textColor}; }}
div.stAlert {{background-color: #F0FBFF; color: {textColor}; border: 1px solid {textColor};}}
<style>
"""

st.markdown(style, unsafe_allow_html=True)
#-----------------------------------------------------

# IMPORTACIÓN DE DATAFRAME Y MODELO
def pickle_load(model_name):
  with open(f'{CURR_DIR}/data/%s.pkl' % model_name,'rb') as pk:
    file = pickle.load(pk)

  return file
  

model = CatBoostRegressor()
model.load_model(f'{CURR_DIR}/data/CatBoost.bin')


df = pickle_load('df_clean')
df_model = pickle_load('df_model')
oneHotEncoder = pickle_load('oneHotEncoder')
cluster = pickle_load('kmeans')
gdf_madrid = pickle_load('gdf_madrid')
#-----------------------------------------------------

def main():
# Función que comprueba que la url introducida es válida.
  def is_covered(link):
    
    covered = False

    if bool(re.search('fotocasa.es',link)) == True:
      if bool(re.search('/madrid-capital/',link)) == True:
        covered = True

    return covered
# Función de transformación de la data extraída en el scrapping.
  def check_features(data):
      '''
      Función de comprobación y transformación de las features extraídas de cada
      vivienda. En el proceso se comprueba su existencia y se transforman en función
      de los ids asignados por Fotocasa. Si no existe el campo se rellena con un
      circunflejo (^).
      
      Args:
      - data (json): json contenedor de las features de cada vivienda.
      - page (int): número de la página.
      ''' 
    
      # Diccionario de features
      realestate = {
          'title': '',
          'link': '',
          'image_url': '',
          'country': '',
          'district': '',
          'neighborhood': '',
          'street': '',
          'zipCode': '',
          'province': '',
          'buildingType': '',
          'clientAlias': '',
          'latitude': '',
          'longitude': '',
          'isNewConstruction': '',
          'rooms': '',
          'bathrooms': '',
          'parking': '',
          'elevator': '',
          'furnished': '',
          'surface': '',
          'energyCertificate': '',
          'hotWater': '',
          'heating': '',
          'conservationState': '',
          'antiquity': '',
          'floor': '',
          'surfaceLand': '',
          'otherFeatures': '',
          'price': '',     
          }
      # Comienzan las comprobaciones feature a feature
      try:
          realestate['title'] = data['propertyTitle']    
      except:
          realestate['title'] = '^'
          
      try:
          realestate['link'] = 'https://www.fotocasa.es' + data['realEstate']['detail']['es-ES']   
      except:
          realestate['link'] = '^'

      try:
          realestate['image_url'] = data['realEstate']['multimedia'][1]['src']
      except:
          realestate['image_url'] = '^'
          
      try:
          realestate['country'] = data['realEstate']['address']['country']
      except:
          realestate['country'] = '^'
          
      try:
          realestate['district'] = data['realEstate']['address']['district']
      except:
          realestate['district'] = '^'
          
      try:
          realestate['neighborhood'] = data['realEstate']['address']['neighborhood']
      except:
          realestate['neighborhood'] = '^'
          
      try:
          realestate['street'] = data['realEstate']['location']
      except:
          realestate['street'] = '^'
          
      try:
          realestate['zipCode'] = data['realEstate']['address']['zipCode']
      except:
          realestate['zipCode'] = '^'
          
      try:
          realestate['province'] = data['realEstate']['address']['province']
      except:
          realestate['province'] = '^'
          
      try:
          realestate['buildingType'] = data['realEstate']['buildingType']
      except:
          realestate['buildingType'] = '^'

      try:
          realestate['clientAlias'] = data['realEstate']['clientAlias']
      except:
          realestate['clientAlias'] = '^'
          
      try:
          realestate['latitude'] = data['realEstate']['coordinates']['latitude']
      except:
          realestate['latitude'] = '^'

      try:
          realestate['longitude'] = data['realEstate']['coordinates']['longitude']
      except:
          realestate['longitude'] = '^'
          
      try:
          realestate['isNewConstruction'] = data['realEstate']['isNewConstruction']
      except:
          realestate['isNewConstruction'] = '^'
          
      try:
          realestate['rooms'] = data['realEstate']['features']['rooms']
      except:
          realestate['rooms'] = '^'
          
      try:
          realestate['bathrooms'] = data['realEstate']['features']['bathrooms']
      except:
          realestate['bathrooms'] = '^'

      try:
          featureList = data['realEstate']['featuresList']
          realestate['parking'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'parking'])
          
      except:
          realestate['parking'] = '^'

      try:
          featureList = data['realEstate']['featuresList']
          realestate['elevator'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'elevator'])
          
      except:
          realestate['elevator'] = '^'

      try:
          featureList = data['realEstate']['featuresList']
          realestate['furnished'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'furnished'])
          
      except:
          realestate['furnished'] = '^'
          
      try:
          realestate['surface'] = data['realEstate']['features']['surface']
      except:
          realestate['surface'] = '^'
          
      try:
          realestate['energyCertificate'] = data['realEstate']['energyCertificate']
      except:
          realestate['energyCertificate'] = '^'
          
      try:
          realestate['hotWater'] = data['realEstate']['features']['hotWater']
          featureList = data['realEstate']['featuresList']
          realestate['hotWater'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'hotWater'])
          
      except:
          realestate['hotWater'] = '^'
          
      try:
          realestate['heating'] = data['realEstate']['features']['heating']
          featureList = data['realEstate']['featuresList']
          realestate['heating'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'heating'])
        
      except:
          realestate['heating'] = '^'
          
      try:
          realestate['conservationState'] = data['realEstate']['features']['conservationState']
          featureList = data['realEstate']['featuresList']
          realestate['conservationState'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'conservationState'])
        
      except:
          realestate['conservationState'] = '^'
          
      try:
          realestate['antiquity'] = data['realEstate']['features']['antiquity']
          featureList = data['realEstate']['featuresList']
          realestate['antiquity'] = ''.join([featureList[index]['value'] for index,value in enumerate(featureList) if featureList[index]['label'] == 'antiquity'])
        
      except:
          realestate['antiquity'] = '^'
          
      try:
          realestate['floor'] = data['realEstate']['features']['floor']
      except:
          realestate['floor'] = '^'
          
      try:
          realestate['surfaceLand'] = data['realEstate']['features']['surfaceLand']
      except:
          realestate['surfaceLand'] = '^'
          
      try:
          realestate['otherFeatures'] = data['realEstate']['otherFeatures']
      except:
          realestate['otherFeatures'] = '^'
          
      try:
          realestate['price'] = data['realEstate']['price']
      except:
          realestate['price'] = 0
          
      #devuelve un diccionario
      return realestate


# Función que comprueba la url, hace una llamada y scrapea los datos de la misma para convertirlos, por medio de las transformaciones necesarias, en el registro a predecir.
  def user_input_parameters():    
    txt = st.sidebar.text_area('URL')
    st_features = pd.DataFrame()

    covered = is_covered(txt)
    
    if covered == True:
      if st.sidebar.button('Check url'):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(txt, headers=headers)
        soup = BeautifulSoup(r.content,'html.parser')
            
        # Obtención y limpieza de la fila del script que contiene la información de propiedades
        prop_scripts = soup.findAll('script')
        prop_features = ''.join([re.search('window.__INITIAL_PROPS__ = JSON.parse(.*)\n',str(x)).group(1) for x in prop_scripts if re.search('window.__INITIAL_PROPS__',str(x))])
        prop_features_clean = re.sub(r'\\"','"',prop_features)
        prop_features_clean = re.sub(r'\\\\"','',prop_features_clean)
        prop_features_clean = re.sub(r'\("|"\);','',prop_features_clean)
        prop_features_clean = re.sub(r',"seo":.*','}',prop_features_clean)

        prop_data = json.loads(prop_features_clean)
        realestate = check_features(prop_data)

        # CONVERSIONES
        realestate['hasSwimmingPool'] = 0
        realestate['hasGarden'] = 0
        realestate['hasTerrace'] = 0
        realestate['hasAirco'] = 0
        realestate['hasParking'] = 0
        realestate['hasLift'] = 0
        realestate['isNewDevelopment'] = int(realestate['isNewConstruction'])
        realestate['title'] = realestate['title'].split()[0].lower()

        if realestate['title'] == 'casa':
          realestate['propertyType'] = 'chalet'
        elif realestate['title'] == 'apartamento':
          realestate['propertyType'] = 'piso'
        elif realestate['title'] == 'ático':
          realestate['propertyType'] = 'atico'
        elif realestate['title'] == 'dúplex':
          realestate['propertyType'] = 'duplex'
        elif realestate['title'] == 'finca':
          realestate['propertyType'] = 'piso'
        else:
          realestate['propertyType'] = realestate['buildingType'].lower()

        if realestate['district'] == 'Fuencarral - El Pardo':
          realestate['district'] = 'Fuencarral'
        elif realestate['district'] == 'Moncloa - Aravaca':
          realestate['district'] = 'Moncloa'
        elif realestate['district'] == 'Chamartín':
          realestate['district'] = 'Chamartin'


        for key,value in realestate['otherFeatures'].items():
          if bool(re.search('Piscina',value)) == True:
            realestate['hasSwimmingPool'] = 1
          elif bool(re.search('Patio|Jard.n',value)) == True:
            realestate['hasGarden'] = 1
          elif bool(re.search('Terraza',value)) == True:
            realestate['hasTerrace'] = 1
          elif bool(re.search('Aire acondicionado',value)) == True:
            realestate['hasAirco'] = 1

        try:
          if bool(re.search('PARKING',realestate['parking'])) == True:
            realestate['hasParking'] = 1
        except:
          pass

        try:
          if bool(re.search('YES',realestate['elevator'])) == True:
            realestate['hasLift'] = 1
        except:
          pass

        if realestate['propertyType'] == 'flat':
          realestate['propertyType'] = 'piso'

        # Creación del registro final
        st_data = {'district': realestate['district'],
                'propertyType': realestate['propertyType'].lower(),
                'size': realestate['surface'] + realestate['surfaceLand'],
                'hasParking': int(realestate['hasParking']),
                'roomNumber': realestate['rooms'],
                'bathNumber': realestate['bathrooms'],
                'hasSwimmingPool':int(realestate['hasSwimmingPool']),
                'hasTerrace':int(realestate['hasTerrace']),
                'hasGarden':int(realestate['hasGarden']),
                'hasLift':int(realestate['hasLift']),
                'hasAirco':int(realestate['hasAirco']),
                'isNewDevelopment': int(realestate['isNewDevelopment']),
                'price_m2_ft': int(df[df['district'] == realestate['district']]['price_m2_ft'].unique()[0]),
                'price': realestate['price']
                }

        st_features = pd.DataFrame(st_data, index=[0]).reset_index(drop=True) # dataframe a predecir
        
    else:
      st.sidebar.write('The prediction only covers urls with fotocasa.es and /madrid-capital/. Ex.: https://www.fotocasa.es/es/comprar/vivienda/madrid-capital/jardin-ascensor/176196114/d?from=list')
    return st_features
    
    
#-----------------------------------------------------
# Función de encoding con One Hot Encoding
  def st_one_hot_encoding(st_df):
    st_df = st_df.reindex(columns=df.columns)

    categorical_columns = st_df.select_dtypes(exclude=["number"]).columns
    print(categorical_columns)

    feature_array = oneHotEncoder.transform(st_df[categorical_columns]).toarray()
    feature_labels = oneHotEncoder.categories_
    feature_labels = np.array(np.hstack(feature_labels),dtype="object").flatten()

    df_ohe = pd.DataFrame(feature_array, columns = feature_labels)
    df_onehot_encoding = pd.concat([st_df,df_ohe],axis=1).drop(categorical_columns, axis=1)

    #st.write(df_onehot_encoding)
    return df_onehot_encoding


#-----------------------------------------------------
# Función que muestra los gráficos de cada barrio. Puede mostrar datos generales o datos con la predicción realizada.
  def about_district(df,user_df,all=True,yhat=0):
    user_propertyType = user_df['propertyType'][0]
    user_district = user_df['district'][0]
    df = df[(df['district'].str.contains(user_district) == True)]

    user_price_mean = round(df[(df['propertyType'].str.contains(user_propertyType) == True) & (df['district'].str.contains(user_district) == True)]['price'].mean(),2)
    user_size_mean = round(df[(df['propertyType'].str.contains(user_propertyType) == True) & (df['district'].str.contains(user_district) == True)]['size'].mean(),2)

	
    st.info(f"The average price of {user_propertyType} in {user_district} is {user_price_mean}€.")
    if all == True:
      fig = px.histogram(df,
          x='price',
          color='propertyType',
          barmode='overlay',
          marginal="box",
          labels={'propertyType':'Tipo','price':'Precio'}
          )
      fig.update_layout(
          width=600,
          height=600,
      )
      fig.add_vline(x=user_price_mean, line_width=2, line_dash="dot", line_color="blue")
      st.plotly_chart(fig)
    else:
      fig = px.histogram(df[(df['propertyType']==user_propertyType) & (df['district'] == user_district)],
      x='price',
      color='propertyType',
      barmode='overlay',
          marginal="box",
          labels={'propertyType':'Type','price':'Price'}
          )
      fig.update_layout(
          width=600,
          height=600,
      )
      fig.add_vline(x=user_price_mean, line_width=2, line_dash="dot", line_color="blue")
      fig.add_vline(x=yhat, line_width=2, line_dash="dash", line_color="green")
      fig.add_vline(x=user_df["price"].values[0], line_width=3, line_dash="dash", line_color="red")
      st.plotly_chart(fig)
	
    st.info(f"The average size of {user_propertyType} in {user_district} is {user_size_mean}m2.")

    if all == True:
      fig = px.histogram(df,
          x='size',
      color='propertyType',
      barmode='overlay',
          marginal="box",
          labels={'propertyType':'Tipo','size':'Tamaño'}
          )
      fig.update_layout(
          width=600,
          height=600,
      )
      fig.add_vline(x=user_size_mean, line_width=2, line_dash="dot", line_color="blue")
      st.plotly_chart(fig)
    else:
      fig = px.histogram(df[(df['propertyType']==user_propertyType) & (df['district'] == user_district)],
          x='size',
      color='propertyType',
      barmode='overlay',
          marginal="box",
          labels={'propertyType':'Type','size':'Size'}
          )
      fig.update_layout(
          width=600,
          height=600,
      )
      fig.add_vline(x=user_size_mean, line_width=2, line_dash="dot", line_color="blue")
      st.plotly_chart(fig)

    if all == True:
      df_grouped = df.groupby(by='propertyType')['price'].count().reset_index().sort_values(by='price', ascending=False).reset_index(drop=True)
      st.info(f"The most abundant type of housing in {user_district} is {df_grouped['propertyType'][0]} with {df_grouped['price'][0]}.")
    
      fig = px.histogram(df,
          x='propertyType',
          #barmode='group',
          color='propertyType',
          text_auto=True,
          labels={'propertyType':'Type','count':'Number'}
          )
      fig.update_layout(
          width=600,
          height=600,
      )
      st.plotly_chart(fig)
	
    return None

#-----------------------------------------------------	
# Función que extrae la similitud del coseno entre los registros del DataFrame general y el introducir por el usuario.
  def similarity(df, comparation):
    similarities = {}
    similarities = {i : float(cosine_similarity(comparation,df[df.index == i])[0]) for i,v in df.iterrows()}
    similarities = pd.DataFrame([similarities]).T.rename(columns={0: "cosine_similarity"})
    similarities = similarities.reset_index().sort_values(by='cosine_similarity', ascending=False).reset_index(drop=True)[:5]
    similarities = df[(df.index.isin(similarities['index'])) & (~df.index.isin(comparation.index))]

    return similarities

#-----------------------------------------------------
# Función que muestra el mapa coroplético con las ubicaciones de las recomendaciones.
  def show_map(recommender):
    gdf_count_recommendations = pd.merge(recommender.groupby(by='district')['price'].count(),gdf_madrid,how='right',left_on='district',right_on='NOMBRE')
    gdf_count_recommendations = gpd.GeoDataFrame(gdf_count_recommendations, crs="EPSG:4326", geometry='geometry').fillna(0)
	
    fig = px.choropleth(gdf_count_recommendations,
        geojson=gdf_count_recommendations.geometry,
        locations=gdf_count_recommendations.index,
        color='price',
        hover_name='NOMBRE',
        color_continuous_scale="blues",
        labels={'price':'Recommendations','index':'index'}
        )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="Datos de cada distrito de Madrid")
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=6.6,
        width=700,
        height=600,
    )
    st.plotly_chart(fig)  
  
    return None

  #-----------------------------------------------------	
	
  st_df = pd.DataFrame()
  madrid = True

  #SIDEBAR
  st.sidebar.image(f'./img/logo.png')
  st.sidebar.header('User input parameters')
  st.sidebar.write('Enter a real url of a property from Fotocasa (Madrid):')

  st_df = user_input_parameters()

  #BODY
  st.title('What will be the price of your ideal home?')
  st.write('The prediction is based on data extracted from the real estate portals Idealista and Fotocasa, so the price we show you is based entirely on the state of the market.')
  st.write('Please enter an url in the sidebar.')
  if st_df.empty == False: # Si el dataframe no está vacío, es decir, se ha pulsado en el botón Check url
    st.write('The following parameters have been selected:')
    st.write(st_df[['price','district','propertyType','size','roomNumber','bathNumber']])
    st_df_encoded = st_one_hot_encoding(st_df) # codifico el df
    st_df_encoded = st_df_encoded.drop('price',axis=1) # borro la columna precio, que no hace falta
    st_df_encoded['cluster'] = cluster.predict(st_df_encoded) # le añado la feature de cluster
    yhat = model.predict(st_df_encoded)[0].round(0) # hago la predicción
    st.success(f'The price of the property will be: {yhat} €. The real price is {st_df["price"].values[0]} €.')
    diff = st_df["price"].values[0]-yhat

    if diff >= 0:
       st.warning(f'The difference is {diff} €. The price is higher than in the real estate market.')
    else:
       st.warning(f'The difference is {diff} €. The price is lower than in the real estate market.')

    st_df_encoded['price'] = yhat

    st.subheader(f"About {st_df[st_df.index == 0]['district'][0]}")
    about_district(df,st_df, all=False,yhat=yhat)
    st.subheader('Similar Real Estates')
    st_df_encoded = st_df_encoded.reindex(columns=df_model.columns)
    recommender = similarity(df_model, st_df_encoded[st_df.index==0])
    recommender = df[df.index.isin(recommender.index)] # extraigo los registros más parecidos
    st.write(recommender[['price','district','propertyType','size','roomNumber','bathNumber']])
    show_map(recommender) # muestro el mapa con las ubicaciones de las recomendaciones
	
    
    
if __name__ == '__main__':
  main()