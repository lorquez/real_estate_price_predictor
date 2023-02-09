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

# Definición de página de inicio
st.set_page_config(
    page_title="Prediction Mode",
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
# Función de tratamiento de datos introducidos por el usuario. Solo se muestra los datos existentes para la dupla Distrito-Tipo de propiedad.
  def user_input_parameters():    

    st_district = st.sidebar.selectbox('District',
                                        df['district'].unique())
    st_type = st.sidebar.selectbox('Type',
                                    df[df['district'] == st_district]['propertyType'].unique())
    st_size = st.sidebar.slider('Size (m2)',
                                int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['size'].min()),
                                int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['size'].max()),
                                int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['size'].mean()))
    st_rooms = st.sidebar.slider('Room number',
                                  int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['roomNumber'].min()),
                                  int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['roomNumber'].max()),
                                  int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['roomNumber'].mean()))
    st_bathrooms = st.sidebar.slider('Bath number',
                                      int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['bathNumber'].min()),
                                      int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['bathNumber'].max()),
                                      int(df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['bathNumber'].mean()))
    st_hasparking = st.sidebar.selectbox('Parking',df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['hasParking'].unique())
    st_hasswimmingpool = st.sidebar.selectbox('Swimmingpool',df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['hasSwimmingPool'].unique())
    st_hasterrace = st.sidebar.selectbox('Terrace',df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['hasTerrace'].unique())
    st_hasgarden = st.sidebar.selectbox('Garden',df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['hasGarden'].unique())
    st_haslift = st.sidebar.selectbox('Lift',df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['hasLift'].unique())
    st_hasairco = st.sidebar.selectbox('Air conditioning',df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['hasAirco'].unique())
    st_isNewDevelopment = st.sidebar.selectbox('Is new development?',
                                      df[(df['district'] == st_district) & (df['propertyType'] == st_type)]['isNewDevelopment'].unique())

    st_data = {'district': st_district,
               'propertyType':st_type,
               'size':st_size,
               'hasParking':int(st_hasparking),
               'roomNumber':st_rooms,
               'bathNumber': st_bathrooms,
               'hasSwimmingPool':int(st_hasswimmingpool),
               'hasTerrace':int(st_hasterrace),
               'hasGarden':int(st_hasgarden),
               'hasLift':int(st_haslift),
               'hasAirco':int(st_hasairco),
               'isNewDevelopment': int(st_isNewDevelopment),
               'price_m2_ft': int(df[df['district'] == st_district]['price_m2_ft'].unique()[0]),
               'price': 0
               }
    #st.write(st_data)
    st_features = pd.DataFrame(st_data, index=[0]).reset_index(drop=True)

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

  #SIDEBAR
  st.sidebar.image('img/logo.png')
  st.sidebar.header('User input parameters')
  st.sidebar.write('Select the features of your ideal home:')

  st_df = user_input_parameters()


  #BODY
  st.title('What will be the price of your ideal home?')
  st.write('The prediction is based on data extracted from the real estate portals Idealista and Fotocasa, so the price we show you is based entirely on the state of the market.')
  st.write('The following parameters have been selected:')
  st.write(st_df[['district','propertyType','size','roomNumber','bathNumber']])

  
  #PREDICT PRICE BUTTON
  if st.button('PREDICT THE PRICE!'):
    st_df_encoded = st_one_hot_encoding(st_df) # codifico el df
    st_df_encoded = st_df_encoded.drop('price',axis=1) # borro la columna precio, que no hace falta
    st_df_encoded['cluster'] = cluster.predict(st_df_encoded) # le añado la feature de cluster
    yhat = model.predict(st_df_encoded)[0].round(0) # hago la predicción
    st.success(f'The price of the property will be: {yhat} €')
    st_df_encoded['price'] = yhat

    st.subheader(f"About {st_df[st_df.index == 0]['district'][0]}")
    about_district(df,st_df, all=False,yhat=yhat) # muestro los gráficos
    st.subheader('Similar Real Estates')
    
    st_df_encoded = st_df_encoded.reindex(columns=df_model.columns)
    recommender = similarity(df_model, st_df_encoded[st_df.index==0])
    recommender = df[df.index.isin(recommender.index)] # extraigo los registros más parecidos
    st.write(recommender[['price','district','propertyType','size','roomNumber','bathNumber']])
    show_map(recommender) # muestro el mapa con las ubicaciones de las recomendaciones
  else:
    #DISTRICT INFO
    st.subheader(f"About {st_df[st_df.index == 0]['district'][0]}")
    about_district(df,st_df, all=True)
	
    
    
if __name__ == '__main__':
  main()