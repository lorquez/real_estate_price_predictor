import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# THEME
CURRENT_THEME = "custom"

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
  with open('%s.pkl' % model_name,'rb') as pk:
    file = pickle.load(pk)

  return file
  

model = xgb.Booster()
model.load_model('XGB_model.bin')

df_dpt = pickle_load('dataframe_dpt')
df_ht = pickle_load('dataframe_ht')
df_ec = pickle_load('dataframe_ec')
df_floor = pickle_load('dataframe_floor')
df_original = pickle_load('df_clean')
df_model = pickle_load('df_model')
df_translations = [df_dpt,df_ht,df_ec,df_floor]
gdf_madrid = pickle_load('gdf_madrid')
#-----------------------------------------------------

def main():
  def user_input_parameters():    
    df_district_propertyType = pd.DataFrame(df_original['district_propertyType'])
    df_district_propertyType = df_district_propertyType.groupby(by="district_propertyType").count().reset_index()

    for index,value in df_district_propertyType.iterrows():
        df_district_propertyType.loc[df_district_propertyType.index == index,'district'] = value['district_propertyType'].rsplit(' ', 1)[0]
        df_district_propertyType.loc[df_district_propertyType.index == index,'propertyType'] = value['district_propertyType'].rsplit(' ', 1)[1]


    df_district_propertyType = df_district_propertyType.drop(['district_propertyType'],axis=1)
	
    st_neighbourhood = st.sidebar.selectbox('Neighborhood',df_district_propertyType['district'].unique())
    st_type = st.sidebar.selectbox('Type',df_district_propertyType[df_district_propertyType['district'].str.contains(st_neighbourhood) == True]['propertyType'].unique())
    st_floor = st.sidebar.selectbox('Floor',df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['floor'].unique())
    st_size = st.sidebar.slider('Size (m2)',int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['size'].min()),int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['size'].max()),int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['size'].mean()))
    st_rooms = st.sidebar.slider('Room number',int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['roomNumber'].min()),int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['roomNumber'].max()),int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['roomNumber'].mean()))
    st_bathrooms = st.sidebar.slider('Bath number',int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['bathNumber'].min()),int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['bathNumber'].max()),int(df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['bathNumber'].mean()))
    st_hasparking = st.sidebar.selectbox('Parking',[True,False])
    st_hasswimmingpool = st.sidebar.selectbox('Swimmingpool',[True,False])
    st_hasterrace = st.sidebar.selectbox('Terrace',[True,False])
    st_hasgarden = st.sidebar.selectbox('Garden',[True,False])
    st_haslift = st.sidebar.selectbox('Lift',[True,False])
    st_heating = st.sidebar.selectbox('Heating',df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['heatingType'].unique())
    st_energyCertificate = st.sidebar.selectbox('Energy Certificate',df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['energyCertificate'].unique())
    st_propertycondition = st.sidebar.selectbox('Property condition',df_original[df_original['district_propertyType'].str.contains(st_neighbourhood) == True]['propertyCondition'].unique())

    st_data = {'district': st_neighbourhood,
               'propertyType':st_type,
               'floor':st_floor,
               'size':st_size,
               'hasParking':int(st_hasparking),
               'roomNumber':st_rooms,
               'bathNumber': st_bathrooms,
               'hasSwimmingPool':int(st_hasswimmingpool),
               'hasTerrace':int(st_hasterrace),
               'hasGarden':int(st_hasgarden),
               'hasLift':int(st_haslift),
               'heatingType':st_heating,
               'energyCertificate':st_energyCertificate,
               'propertyCondition':int(st_propertycondition),
               }
    #st.write(st_data)
    st_features = pd.DataFrame(st_data, index=[0]).reset_index(drop=True)
    return st_features
#-----------------------------------------------------
  def st_encoding(df,df_translations):
    df['district_propertyType'] = df['district'].str.cat(df['propertyType'], sep=' ')
    df = df.drop(['district','propertyType'], axis=1)
    df['room_bath_rate'] = df['roomNumber'].astype(float)/df['bathNumber'].astype(float)*100

    df = pd.merge(df,df_translations[0], 
                  how='left', 
                  left_on='district_propertyType', 
                  right_on='district_propertyType').drop(['district_propertyType'],axis=1).rename(columns = {'district_propertyType_target':'district_propertyType'})

    df = pd.merge(df,df_translations[3], 
                  how='left', 
                  left_on='floor', 
                  right_on='floor').drop(['floor'],axis=1).rename(columns = {'floor_target':'floor'})

    df = pd.merge(df,df_translations[1], 
                  how='left', 
                  left_on='heatingType', 
                  right_on='heatingType').drop(['heatingType'],axis=1).rename(columns = {'heatingType_target':'heatingType'})
    
    df = pd.merge(df,df_translations[2], 
                  how='left', 
                  left_on='energyCertificate', 
                  right_on='energyCertificate').drop(['energyCertificate'],axis=1).rename(columns = {'energyCertificate_target':'energyCertificate'})

    df = df[['size', 'hasParking', 'roomNumber', 'bathNumber',
       'hasSwimmingPool', 'hasTerrace', 'hasGarden', 'hasLift',
       'room_bath_rate', 'propertyCondition', 'floor', 'district_propertyType',
       'heatingType', 'energyCertificate']]


    return df
#-----------------------------------------------------	
  def similarity(df, df_clean, comparation):
    district = float(comparation['district_propertyType'][0])
    similarities = {}
    #similarities = {i : float(cosine_similarity(comparation,df[df.index == i])[0]) for i,v in df[df['district_propertyType'] == district].iterrows()}
    similarities = {i : float(cosine_similarity(comparation,df[df.index == i])[0]) for i,v in df.iterrows()}
    similarities = pd.DataFrame([similarities]).T.rename(columns={0: "cosine_similarity"})
    similarities = similarities.reset_index().sort_values(by='cosine_similarity', ascending=False).reset_index(drop=True)[:5]
    similarities = df_clean[df_clean.index.isin(similarities['index'])].reset_index(drop=True)

    return similarities
#-----------------------------------------------------
  def show_map(recommender):
    recommender.loc[:,'aux'] = recommender['district_propertyType'].str.rsplit(' ', 1)
    for index,value in recommender.iterrows():
        recommender.loc[recommender.index == index,'district'] = value['aux'][0]
        recommender.loc[recommender.index == index,'propertyType'] = value['aux'][1]
    recommender = recommender.drop(['aux'],axis=1)
		
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
  def about_district(df,user_df):
    user_propertyType = user_df['propertyType'][0]
    user_district = user_df['district'][0]
    df = df[(df['district_propertyType'].str.contains(user_district) == True)]
  
    df.loc[:,'aux'] = df['district_propertyType'].str.rsplit(' ', 1)
    for index,value in df.iterrows():
        df.loc[df.index == index,'district'] = value['aux'][0]
        df.loc[df.index == index,'propertyType'] = value['aux'][1]
    df = df.drop(['aux'],axis=1)  
  

    user_price_mean = round(df[(df['district_propertyType'].str.contains(user_propertyType) == True) & (df['district_propertyType'].str.contains(user_district) == True)]['price'].mean(),2)
    user_size_mean = round(df[(df['district_propertyType'].str.contains(user_propertyType) == True) & (df['district_propertyType'].str.contains(user_district) == True)]['size'].mean(),2)

	
    st.info(f"El precio medio de los {user_propertyType}s en {user_district} es de {user_price_mean}€.")
	
    fig = px.histogram(df,
        x='price',
		color='propertyType',
		#log_x = True,
        #text_auto=True,
		barmode='overlay',
        marginal="box",
        labels={'propertyType':'Tipo','price':'Precio'}
        )
    fig.update_layout(
        width=600,
        height=600,
    )
    st.plotly_chart(fig)
	
    st.info(f"El tamaño medio de los {user_propertyType}s en {user_district} es de {user_size_mean}m2.")

    fig = px.histogram(df,
        x='size',
		color='propertyType',
		#log_x = True,
        #text_auto=True,
		barmode='overlay',
        marginal="box",
        labels={'propertyType':'Tipo','size':'Tamaño'}
        )
    fig.update_layout(
        width=600,
        height=600,
    )
    st.plotly_chart(fig)


    df_grouped = df.groupby(by='propertyType')['price'].count().reset_index().sort_values(by='price', ascending=False).reset_index(drop=True)
    st.info(f"El tipo de vivienda más abundante en {user_district} es el de {df_grouped['propertyType'][0]} con {df_grouped['price'][0]}.")
	
    fig = px.histogram(df,
        x='propertyType',
        #barmode='group',
        color='propertyType',
        text_auto=True,
        labels={'propertyType':'Tipo','count':'Número'}
        )
    fig.update_layout(
        width=600,
        height=600,
    )
    st.plotly_chart(fig)
	
    return None

#-----------------------------------------------------	
	
  st_df = pd.DataFrame()

  #SIDEBAR
  st.sidebar.image('img/logo.png')
  st.sidebar.header('User input parameters')
  st.sidebar.write('Selecciona las características de tu vivienda ideal:')

  st_df = user_input_parameters()


  #BODY
  st.title('¿Cuál será el precio de tu vivienda ideal?')
  #st.subheader('Based on Idealista and Fotocasa datasets.')
  st.write('La predicción se basa en los datos extraídos de los portales inmobiliarios de Idealista y Fotocasa, por lo que el precio que te mostramos se basa totalmente en el estado del mercado.')
  st.write('Los parámetros seleccionados son los siguientes:')
  #st.subheader(st_model)
  st.write(st_df)

  
  #PREDICT PRICE BUTTON
  if st.button('¡PREDECIR EL PRECIO!'):
    st_df_encoded = st_encoding(st_df,df_translations) 
    yhat = model.predict(xgb.DMatrix(st_df_encoded))[0].round(0)
    st.info(f'El precio de la vivienda será de: {yhat} €')
    st_df_encoded['price'] = yhat
	
    st_df_encoded = st_df_encoded[['price', 'size', 'hasParking', 'roomNumber', 'bathNumber',
    'hasSwimmingPool', 'hasTerrace', 'hasGarden', 'hasLift',
    'room_bath_rate', 'propertyCondition', 'floor', 'district_propertyType',
    'heatingType', 'energyCertificate']]	
	
    st.subheader('Similar Real Estates')
    recommender = similarity(df_model, df_original, st_df_encoded[st_df.index==0])
    st.write(recommender)
    show_map(recommender)
	
  #DISTRICT INFO
  st.subheader(f"Sobre {st_df[st_df.index == 0]['district'][0]}")
  about_district(df_original,st_df)
	
    
    
if __name__ == '__main__':
  main()
