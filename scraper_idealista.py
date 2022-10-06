from bs4 import BeautifulSoup
import concurrent.futures
import pandas as pd
import numpy as np
import requests
import time
import json
import os
import re

class idealista_scraper:

    def __init__(self, url:str) -> None:
        self._url = url
        self._IDEALISTA_HOSTNAME = 'https://www.idealista.com'
        self.api_endpoint = 'http://async.scraperapi.com/jobs'
        f = open('scraping_apikey','r')
        self.api_key = f.read()
        f.close()
             
    def _proxy_requests(self, url:str) -> dict:
        json = {
            'apiKey':self.api_key,
            'url':url
            }
        
        response_job = requests.post(self.api_endpoint, json=json)
        for _ in range(10):
            if requests.get(url = response_job.json()['statusUrl']).json()['status'] == "finished":
                break        
            time.sleep(5)
        
        response = requests.get(url = response_job.json()['statusUrl'])
        return response.json()['response']
        
    def _links_from_breadcrumb(self, thisUrl:str) -> list:
        '''
        Gets an area url as input and returns a list of links for all 
        the areas (district or subdistrict) with 1800 houses or less
        If it finds areas with more than 1800 houses, it runs itself
        recursively.
        The data is obtained from the breadcrumb element of the HTML page

        Args:
        - thisUrl (str): the url of the area to look into

        '''
        
        print(f'starting from {thisUrl}')
        response = self._proxy_requests(thisUrl)
        if response['statusCode'] != 200:
            raise f'Could not retrieve data for {thisUrl}'
        soup = BeautifulSoup(response['body'],'html.parser')
        
        current_level_links = [x for x in soup.select('li.breadcrumb-dropdown-element.highlighted a')]
        current_level_house_numbers = [int(x.text.replace('.','')) for x in soup.select('li.breadcrumb-dropdown-element.highlighted .breadcrumb-navigation-sidenote') if x.text]

        current_level_list = list(zip(current_level_links,current_level_house_numbers))

        # Separating areas by the number of houses
        more_than_1800_houses = [x for x in current_level_list if x[1] > 1800]
        print(f'found {len(more_than_1800_houses)} areas with more than 1800 houses')

        ready_for_scraping = [x for x in current_level_list if x[1] <= 1800]
        print(f'found {len(ready_for_scraping)} areas with less than 1800 houses')
        
        # Recursively searching through areas with more than 1.800 houses
        for area,_ in more_than_1800_houses:
            area_url = self._IDEALISTA_HOSTNAME+area.get('href')
            print(f'recursion over: {area_url}')
            areas = self._links_from_breadcrumb(area_url)
            ready_for_scraping.extend(areas)
        
        return ready_for_scraping

    def get_areas_df(self) -> None:
        '''
        Has to be run first. It creates a urls list for each area
        that can be found as a subarea of the url given as input.

                Writes a new class property named "areas_df" (pandas DataFrame)

        '''

        areas_list = [[x[0].text,x[0].get('href'),x[1]] for x in self._links_from_breadcrumb(self._url)]
        self.areas_df = pd.DataFrame(areas_list, columns=['area_name','area_url','n_houses']).sort_values(by='n_houses')
        self.areas_df = self.areas_df.reset_index().drop('index',axis=1)
        self.areas_df['page'] = 1
        self.areas_df['done'] = False
        self.areas_df.to_csv('../areas_df.csv', index=False)

    def generate_properties_links_df(self) -> None:
        '''
        Has to be run after get_areas_df as it needs areas_df to work.
        It iterates through each area stored in areas_df DataFrame, and extract
        houses' links from each page of the area (district or subdistrict)

        It creates a new class property "properties_links_df"

        '''

        if not hasattr(self,'areas_df'):
            try:
                print(f'No areas_df dataframe. Try to read from CSV.')
                self.areas_df = pd.read_csv('../areas_df.csv')
            except:
                raise 'Please run get_areas_df first'
            
        
        print(f'Areas found: {self.areas_df.shape[0]}')
        
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(
                self._generate_single_area_property_links,
                self.areas_df[self.areas_df['done']==False].iterrows()
                )
            

        self.properties_links_df = pd.read_csv('../properties_links_df.csv')

    def _generate_single_area_property_links(self, area) -> None:
        '''
        Used do retrieve the properties' links for a specific area using parallel processing
        
        '''

        page = int(area['page'])
        path = f"{self._IDEALISTA_HOSTNAME}{area['area_url']}pagina-{str(page)}.htm?ordenado-por=fecha-publicacion-asc"
        print(f"Getting properties\' url for {area['area_name']}")

        while True:
            print(f'Path is {path}')
            response = self._proxy_requests(path)
            if response['statusCode'] != 200:
                raise f"Cannot retrieve data for page {page} of {area['area_name']}"
            soup = BeautifulSoup(response['body'],'html.parser')
            
            # extracting links from page and creating a 3 columns df
            thisPageLinks = [[area['area_name'],self._IDEALISTA_HOSTNAME+x.get('href'),False] for x in soup.select('a.item-link')]
            thisPageLinks_df = pd.DataFrame(thisPageLinks, columns=['area_name','property_link','done'])

            # create or concat data to properties_links_df
            if hasattr(self,'properties_links_df'):
                self.properties_links_df = pd.concat([self.properties_links_df, thisPageLinks_df])
            else:
                try:
                    self.properties_links_df = pd.concat(pd.read_csv('../properties_links_df.csv'),thisPageLinks_df)
                except:
                    self.properties_links_df = thisPageLinks_df.copy()
            
            header = not os.path.exists('../properties_links_df.csv')
            thisPageLinks_df.to_csv('../properties_links_df.csv', mode='a', index=False, header=header)

            print(f'Page {page}: property links added')
            print(f'properties_links_df now has {len(self.properties_links_df)} links')

            # if there is a next page
            next_page = soup.select_one('.pagination .next a')
            if bool(next_page):
                # done with this page
                self._headers['Referrer'] = path
                path = self._IDEALISTA_HOSTNAME+next_page.get('href')
                # storing next page on areas_df in case this breaks
                page += 1
                self.areas_df.at[area.Index,"page"] = page
                print(f'Next page: {path}, which is number {page}')
                self.areas_df.to_csv('../areas_df.csv', index=False)
            else:
                # done with this area
                self.areas_df.at[area.Index,"done"] = True
                print(f"all properties\' links from {area['area_name']} have been extracted")
                break  

    def get_properties_data(self) -> None:
        '''
        Has to run after generate_properties_links_df. It takes the properties links, 
        access them and dump the html as text files locally.

        '''

        if not hasattr(self,'properties_links_df'):
            try:
                self.properties_links_df = pd.read_csv('../properties_links_df.csv')
            except:
                raise 'Cannot find properties\' links'

        print(f'Properties\' links found: {self.properties_links_df.shape[0]}.')

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            executor.map(
                self._dump_single_property_data,
                self.properties_links_df[self.properties_links_df['done']==False].iterrows()
                )

    def _dump_single_property_data(self,row) -> None:
        property_id = row['property_link'].split('/')[-2]
        print(f"{property_id}: retrieving data")
        response = self._proxy_requests(row.property_link)
        if response['statusCode'] == 404:
            print(f'{property_id}: property not found {row.property_link}')
            self.properties_links_df = self.properties_links_df.drop(row.Index)
            self.properties_links_df.to_csv('../properties_links_df.csv',index=False)
            return None        

        with open(f'../properties/{property_id}','w') as f:
            f.write(response['body'])
        
        self.properties_links_df.at[row.Index,'done'] = True
        self.properties_links_df.to_csv('../properties_links_df.csv',index=False)

        print(f'Property {row.Index+1} of {self.properties_links_df.shape[0]} dumped')

    def create_dataset(self) -> None:
        '''
        Access the dumped html code saved as text files for all the properties and
        retrieve the properties' features from them. It creates a new class property "dataset"

        '''
        dumped_data_files = os.listdir('../properties')
        if not hasattr(self,'properties_links_df'):
            try:
                self.properties_links_df = pd.read_csv('../properties_links_df.csv')
            except:
                print('WARNING: Could not find properties_links_df.')

        # if dataset exists then this is an update so skip the files already processed
        if hasattr(self,'dataset'):
            dumped_data_files = [x for x in dumped_data_files if not (self.dataset['id']==x).any()]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._get_single_property_data,dumped_data_files)

        self.dataset = pd.read_csv('../idealista_dataset.csv')

    def _get_single_property_data(self,prop_dumped_data_file_name) -> None:
        '''
        Used do retrieve single property data with parallel processing

        '''
        property_id = prop_dumped_data_file_name

        with open(f'../properties/{property_id}','r') as f:
            soup = BeautifulSoup(f.read(),'html.parser')

        print(f'{property_id}: soup parsed')

        if soup.select('#notFoundWithSuggestions'):
            print(f'{property_id}: this property has been removed')
            try:
                row_to_remove = self.properties_links_df[self.properties_links_df['property_link'].str.contains(property_id)]
                self.properties_links_df = self.properties_links_df.drop(row_to_remove.Index)
                self.properties_links_df.to_csv('../properties_links_df.csv',index=False)
            except:
                pass
            return None

        try:
            utag_script = list(filter(lambda x: 'utag_data' in x.get_text(),soup.select('script')))[0]
            utag_data = json.loads(str(utag_script).split(';')[0].split(' ')[7])
        except:
            print(f'{property_id}: cannot retrieve data')
            return None
        property_data = {
            'id':utag_data['ad']['id'],
            'propertyType':soup.select_one('.main-info .typology').text.strip().lower(),
            'title':soup.select_one('.main-info .txt-body').text.strip().lower(),
            'locationId':utag_data['ad']['address']['locationId'],
            'price':utag_data['ad']['price'],
            'size':utag_data['ad']['characteristics']['constructedArea'],
            'hasParking':utag_data['ad']['characteristics'].get('hasParking',0), # if not exist, get 0
            'roomNumber':utag_data['ad']['characteristics']['roomNumber'],
            'bathNumber':utag_data['ad']['characteristics']['bathNumber'],
            'hasSwimmingPool':utag_data['ad']['characteristics'].get('hasSwimmingPool',0), # if not exist, get 0
            'hasTerrace':utag_data['ad']['characteristics'].get('hasTerrace',0), # if not exist, get 0
            'hasGarden':utag_data['ad']['characteristics'].get('hasGarden',0), # if not exist, get 0
            'hasLift':utag_data['ad']['characteristics'].get('hasLift',0), # if not exist, get 0
            'isGoodCondition':utag_data['ad']['condition']['isGoodCondition'],
            'isNeedsRenovating':utag_data['ad']['condition']['isNeedsRenovating'],
            'isNewDevelopment':utag_data['ad']['condition']['isNewDevelopment'],
            'featureTags': [x.get_text().strip() for x in soup.select('.info-features-tags')]
        }

        heatingData = list(filter(lambda x: 'calefacción' in x.get_text().lower(),soup.select('.details-property_features li')))
        if heatingData:
            property_data['heatingType'] = heatingData[0].get_text()
        else:
            property_data['heatingType'] = "no info"

        if property_data['propertyType'] == 'piso':
            info_features = soup.select('.info-features > span')
            if [x for x in info_features if "interior" in x.get_text()]:
                property_data['interiorExterior'] = "interior"
            elif [x for x in info_features if "exterior" in x.get_text()]:
                property_data['interiorExterior'] = "exterior"
            else:
                property_data['interiorExterior'] = "no info"
            floor_info = [x for x in info_features if re.search("bajo|sótano|planta", x.get_text().lower())]
            if floor_info:
                property_data['floor'] = floor_info[0].select_one('span').get_text().lower().strip()
            else:
                property_data['floor'] = "no info"
        else:
            property_data['floor'] = property_data['interiorExterior'] = "does not apply"
        
        property_data_df = pd.DataFrame.from_dict(property_data,orient='index').T
        print(f'{property_id}: data converted to DF')

        header = not os.path.exists('../idealista_dataset.csv')
        property_data_df.to_csv('../idealista_dataset.csv', mode='a', index=False, header=header)

    def get_location_ids_mapper(self) -> dict:
        '''
        By default, every record in the dataset has de feature "location id" which
        is the website own id to identify the district. This function scrape and
        creates a new class property named "location_ids_mapper", a dict to map 
        location ids to district's names

        '''
        
        if os.path.exists('../location_ids.json'):
            with open('../location_ids.json','r') as f:
                location_ids_mapper = dict(f.read())
        else:
            response = self._proxy_requests(self._url)
            if response['statusCode'] != 200:
                raise f'Could not retrieve the location mapper'
            soup = BeautifulSoup(response['body'],'html.parser')

            locations_list = [(x.get('data-location-id'),x.select_one('a').get_text()) for x in soup.select('.breadcrumb-dropdown-subitem-element-list')]

            location_ids_mapper = {key:value for key,value in locations_list}

            with open('../location_ids.json','w') as f:
                f.write(json.dumps(location_ids_mapper))

        return location_ids_mapper

    def full_scrape(self) -> None:
        '''
        Runs a full scrape of idelista properties. Firstly it will run get_areas_df(),
        then generate_properties_links_df() and get_properties_data(). It will then complete
        the dataset by running the get_location_ids_mapper() and applying the mapper to the 
        dataset. Finally, it will export the dataset calling the file "idealista_dataset.csv"

        Receives no arguments and returns None
        '''
        self.get_areas_df()
        
        print('Found '+self.areas_df['n_houses'].sum()+' houses')

        self.generate_properties_links_df()

        self.properties_links_df = self.properties_links_df.drop_duplicates()

        self.get_properties_data()

        mapper = self.get_location_ids_mapper()
        self.dataset['locationId'] = self.dataset['locationId'].apply(lambda x: "-".join(x.split("-")[:8]))
        self.dataset['area_name'] = self.dataset['locationId'].map(mapper)

        self.dataset.to_csv('../idealista_dataset.csv')



url = 'https://www.idealista.com/venta-viviendas/madrid-madrid/'
scraper = idealista_scraper(url)

scraper.full_scrape()

idealista_dataset = scraper.dataset.copy()

idealista_dataset['locationId'] = idealista_dataset['locationId'].apply(lambda x: "-".join(x.split("-")[:8]))
idealista_dataset['area_name'] = idealista_dataset['locationId'].map(mapper)
idealista_dataset['price_m2'] = (idealista_dataset['price'] / idealista_dataset['size']).round(2)

# Idealista often suffer from fraudolent announces so it's better to aggregate 
# the price_m2_per_area using the median as it is more robust than the mean
price_m2_per_area = idealista_dataset[['area_name','price_m2']].groupby(['area_name']).median().reset_index()
price_m2_per_area = {key:value for key,value in price_m2_per_area.values}

idealista_dataset['area_price'] = idealista_dataset['area_name'].map(price_m2_per_area).astype(float).round(2)





import requests
import pandas as pd
import time
import concurrent.futures

df = pd.read_csv('../properties_links_df.csv')

proxies = {
        "http": "http://065fa1de2a6e43a084d3dcbdb652f8a2:@proxy.crawlera.com:8011/",
        "https": "http://065fa1de2a6e43a084d3dcbdb652f8a2:@proxy.crawlera.com:8011/"
        }

def get_data(url):
    property_id = url.split('/')[-2]
    print(f'{property_id} start')
    while True:
        response = requests.get(
            url,
            proxies=proxies,
            verify='../zyte-proxy-ca.crt' 
        )
        if response.text != 'All download attempts failed. Please retry.':
            print(f'{property_id} attempt succeded')
            break

        print(f'{property_id} attempt failed')
        time.sleep(2)


    file = open('../properties/'+property_id,'w')
    file.write(response.text)
    file.close()
    print(f'{property_id} done')
    return 'done'



with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    results = executor.map(get_data,list(df[df['done']==False][:10000]['property_link'].values))

list(results)

########################################

import os
import pandas as pd

properties_already_downloaded = os.listdir('../properties/')
start_url = 'https://www.idealista.com/venta-viviendas/madrid-madrid/?ordenado-por=fecha-publicacion-desc'
downloaded = os.listdir('../properties')

def generate_properties_links_df(url) -> None:
    
    path = url
    page = 1
    while True:
        print(f'Path is {path}')
        response_job = scraper._proxy_requests(path)
        while requests.get(url = response_job.json()['statusUrl']).json()['status'] != "finished":
            print(f'Job not ready')
            time.sleep(5)
        response = requests.get(url = response_job.json()['statusUrl'])
        if response.status_code != 200:
            raise f"Cannot retrieve data"
        soup = BeautifulSoup(json.loads(response.text)['response']['body'],'lxml')
        print(response.content)
        
        # extracting links from page and creating a 3 columns df
        thisPageLinks = []
        for link in soup.select('a.item-link'):
            link_url = link.get('href')
            if link_url.split('/')[-2] in downloaded:
                break
            
            thisPageLinks.append(['madrid_recent',scraper._IDEALISTA_HOSTNAME+link_url,False])

        thisPageLinks_df = pd.DataFrame(thisPageLinks, columns=['area_name','property_link','done'])
        print(f'thisPageLinks_df: {thisPageLinks_df}')

        header = not os.path.exists('../properties_links_df.csv')
        thisPageLinks_df.to_csv('../properties_links_df.csv', mode='a', index=False, header=header)

        print(f'Page {page}: property links added')

        if len(thisPageLinks)<30:
            print('Done extracting new properties')
            return None

        # if there is a next page
        next_page = soup.select_one('.pagination .next a')
        if bool(next_page):
            # done with this page
            path = scraper._IDEALISTA_HOSTNAME+next_page.get('href')
            # storing next page on areas_df in case this breaks
            page += 1
            print(f'Next page: {path}, which is number {page}')
        else:
            # done with this area
            print(f'all properties\' links have been extracted')
            break

def get_data_proxy(url):

    property_id = url.split('/')[-2]
    print(f'{property_id} start')
    
    for _ in range(10):
        response_job = scraper._proxy_requests(url)
        print(f'{property_id} job submitted')

        while requests.get(url = response_job.json()['statusUrl']).json()['status'] != "finished":
            print(f'{property_id} job not ready')
            time.sleep(5)
        
        response = requests.get(url = response_job.json()['statusUrl'])
        if response.status_code==200:
            print(f'{property_id} job ready with code 200')
            file = open('../properties/'+property_id,'w')
            file.write(str(json.loads(response.text)['response']['body']))
            file.close()
            break
        print(f'{property_id} job returned an error.')

df = pd.read_csv('../properties_links_df.csv')

generate_properties_links_df(start_url)

with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    executor.map(get_data_proxy,list(df[df['done']==False]['property_link'].values))

df['done'] = df['property_link'].apply(lambda x: x.split('/')[-2] in downloaded)

df.to_csv('../properties_links_df.csv',index=False)

