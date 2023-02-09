import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import pyproj

#---------------------------------------------------------------

def plot_correlation_matrix(df,mask=False):
  '''
  Function that plots the Pearson correlation matrix
  ---------
  df: dataframe on which the function is applied
  mask: if set to true draws only the lower triangle of the matrix
  '''
  fig, ax = plt.subplots(1,1,figsize=(20,10))
  ax.set_title('Correlation matrix',fontsize=22)
  corr = df.corr()

  if mask == False:
    sns.heatmap(corr, annot=True, ax=ax);
  elif mask == True:
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, ax=ax, mask=mask);

  return None

def plot_feature_correlation_matrix(df,feature):
  fig, ax = plt.subplots(1,1,figsize=(20,10))
  ax.set_title(f'Correlation with {feature}',fontsize=22)
  sns.heatmap(pd.DataFrame(df.corr()[feature]).sort_values(by=feature,ascending=False),annot=True)

  return None


def regplot_correlations(df,feature1='price',feature2='',hue='district'):
  '''
  Function that draws all correlations of a dataframe, either together or separately.
  ---------
  df: dataframe on which the function is to be applied
  feature1: feature to be correlated
  feature1: second feature to correlate. If left empty, all correlations will be drawn.
  hue: feature by which the graph is to be differentiated chromatically
  '''
  if (feature1 == 'price') & (feature2 == ''):
    corr_indexes=df.corrwith(df[feature1]).sort_values(ascending=False).index
    fig = plt.figure(figsize=(15,30))  
    for i, col in enumerate(corr_indexes):
      fig.add_subplot(7,3,1+i)
      sns.regplot(data=df, x=col, y=feature1)  
      plt.title(f'Correlation {df[col].corr(df[feature1]).round(2)}')
      plt.xlabel(col)
      plt.ylabel(feature1)  
      fig.tight_layout()
  else:
    colors = ['#E74C3C','#3498DB']
    fig, ax = plt.subplots(1,1,figsize = (20,15))
    if hue != False:
      sns.scatterplot(data=df,x=feature1,y=feature2, ax=ax, hue=hue)
    else:
      sns.scatterplot(data=df,x=feature1,y=feature2, ax=ax)
    sns.regplot(data=df, x=feature1, y=feature2, ax=ax, scatter=False, line_kws={'color': colors[0]}, scatter_kws={'color': colors[1]}, ci=100)
    ax.set(xlabel=feature1, ylabel=feature2)
    ax.set_title(f"{feature1} vs {feature2} - Correlation {df[feature1].corr(df[feature2]).round(2)}",fontsize=22)

  return None


  #---------------------------------------------------------------

def plot_choropletic_map(df,feature=''):
  '''
  Function that draws interactive choropleth maps
  ----------
  df: dataframe from which the information is extracted
  feature: feature that draws the map. Only values like id, size and price are allowed.
  '''
  mapa_distritos = '/content/drive/MyDrive/Kschool_TFM/raw_data/mapa_madrid/Distritos_20210712.shp'
  gpd_madrid = gpd.read_file(mapa_distritos)
  gpd_madrid.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

  gpd_madrid.loc[gpd_madrid['NOMBRE'] == 'San Blas - Canillejas','NOMBRE'] = 'San Blas'
  gpd_madrid.loc[gpd_madrid['NOMBRE'] == 'Salamanca','NOMBRE'] = 'Barrio de Salamanca'
  gpd_madrid.loc[gpd_madrid['NOMBRE'] == 'Chamartín','NOMBRE'] = 'Chamartin'
  gpd_madrid.loc[gpd_madrid['NOMBRE'] == 'Fuencarral - El Pardo','NOMBRE'] = 'Fuencarral'
  gpd_madrid.loc[gpd_madrid['NOMBRE'] == 'Moncloa - Aravaca','NOMBRE'] = 'Moncloa'

  if feature == '':
    fig = px.choropleth(gpd_madrid,
                      geojson=gpd_madrid.geometry,
                      locations=gpd_madrid.index,
                      color='Area_m2',
                      hover_name='NOMBRE',
                      color_continuous_scale="turbid",
                      labels={'Area_m2':'Metros cuadrados del distrito',
                              'index':'index'}
                    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="Datos de cada distrito de Madrid")
    fig.show()

  elif feature == 'id':
    gpd_madrid_countrealestates = pd.merge(df.groupby(by='district')['id'].count(),gpd_madrid,how='inner',left_on='district',right_on='NOMBRE')
    gpd_madrid_countrealestates = gpd.GeoDataFrame(gpd_madrid_countrealestates, crs="EPSG:4326", geometry='geometry')

    fig = px.choropleth(gpd_madrid_countrealestates,
                    geojson=gpd_madrid_countrealestates.geometry,
                    locations=gpd_madrid_countrealestates.index,
                    color='id',
                    hover_name='NOMBRE',
                    color_continuous_scale="turbid",
                    labels={'id':'Número de propiedades en el dataset',
                          'index':'index'}
                    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="Datos de cada distrito de Madrid")
    fig.show()
  elif feature == 'size':
    gpd_madrid_meansize = pd.merge(df.groupby(by='district')['size'].mean(),gpd_madrid,how='inner',left_on='district',right_on='NOMBRE')
    gpd_madrid_meansize = gpd.GeoDataFrame(gpd_madrid_meansize, crs="EPSG:4326", geometry='geometry')

    fig = px.choropleth(gpd_madrid_meansize,
                    geojson=gpd_madrid_meansize.geometry,
                    locations=gpd_madrid_meansize.index,
                    color='size',
                    hover_name='NOMBRE',
                    color_continuous_scale="turbid",
                    labels={'size':'Dimensión media de las propiedades (m2)',
                          'index':'index'}
                    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="Datos de cada distrito de Madrid")
    fig.show()

  elif feature == 'price':
    gpd_madrid_meanprice = pd.merge(df.groupby(by='district')['price'].mean(),gpd_madrid,how='inner',left_on='district',right_on='NOMBRE')
    gpd_madrid_meanprice = gpd.GeoDataFrame(gpd_madrid_meanprice, crs="EPSG:4326", geometry='geometry')

    fig = px.choropleth(gpd_madrid_meanprice,
                    geojson=gpd_madrid_meanprice.geometry,
                    locations=gpd_madrid_meanprice.index,
                    color='price',
                    hover_name='NOMBRE',
                    color_continuous_scale="turbid",
                    labels={'price':'Precio medio de las propiedades (€)',
                          'index':'index'}
                    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="Datos de cada distrito de Madrid")
    fig.show()

  return None
