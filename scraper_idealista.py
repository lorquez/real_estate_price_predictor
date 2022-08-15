from bs4 import BeautifulSoup
from latest_user_agents import get_random_user_agent
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ProxyError,HTTPError,ConnectTimeout,ConnectionError
import time
import json
import os

class idealista_scraper:

    def __init__(self, url:str) -> None:
        self._scraping_var_wait_time = 10 # this will be multiplied for a random percentage
        self._scraping_fix_wait_time = 5 # this will be added to wait time as is
        self._IDEALISTA_HOSTNAME = 'https://www.idealista.com'
        self._url = url
        self.reset_headers()

    def reset_headers(self) -> None:
        '''
        Reset the headers to their initial value. Useful after editing them to get around
        the website anti-scraping protections.
        '''

        self._headers = {
            "Host": "www.idealista.com",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "TE": "trailers"
        }
    
    def _get_free_proxies(self) -> None:
        '''
        This function scrapes the free-proxy-list.net website looking for
        free proxies and puts them in a class property called "proxies"
        as a list.

        Args:
        - headers: the user agent needed to avoid 403 error when accessing the free-proxy-list.net website

        '''
        endpoint = 'https://api.proxyscrape.com/v2/'
        params = {
            'request':'displayproxies',
            'protocol':'http',
            'timeout':10000,
            'country':'all', # or gb,es,pt,it,fr,de,nl
            'ssl':'all',
            'anonymity':'all'
        }

        response = requests.get(endpoint,params=params)
        
        self.proxies = response.text.split('\r\n')[:-1]
        print(f'found {len(self.proxies)} proxies')

    def set_wait_time(self, min:int, max:int=0) -> None:
        '''
        Set the minimum and maximum amount of time the algoritm has to
        wait in between requests to the website to avoid having an IP
        address blacklisted.

        Args:
        - min: the minimum amount of time
        - max: the maximum amount of time
        '''
        if max != 0 and max < min:
            print('Specify a max value higher than min or leave it blank')
            return None

        self._scraping_fix_wait_time = min
        self._scraping_var_wait_time = max - min
    
        if min == max or max == 0:
            print(f'Wait time set to {min}')
        else:
            print(f'Wait time set from {min} to {max}')

    def get_wait_time(self) -> None:
        '''
        Prints the current min and max wait time on screen
        '''
        min = self._scraping_fix_wait_time
        max = self._scraping_var_wait_time + min
        if min == max or max == 0:
            print(f'Wait time set to {min}')
        else:
            print(f'Wait time set from {min} to {max}')

    def wait_time(self) -> None:
        '''
        Wrapper for time.sleep function to wait an amount of seconds that
        is based on the property "scraping_var_wait_time" multiplied
        for a random number (from 0 to 1) generated with numpy plus the 
        property "scraping_fix_wait_time". Default config: min 10 sec,
        max 1 min.

        No arguments

        '''
        
        wait_for =  self._scraping_fix_wait_time
        wait_for += self._scraping_var_wait_time*np.random.rand()
        print(f'waiting for {wait_for} seconds')
        time.sleep(wait_for)

    def _proxy_requests_api(self, url:str) -> str:

        endpoint = ' https://api.scraperapi.com'
        params = {
          'api_key':'b07852a87a699b4a8284c62bc639547b',
          'url':url
        }
        response = requests.get(endpoint, params=params)
        return response

    def _proxy_requests(self, url:str, timeout:int=5) -> requests.models.Response:
        '''
        Wrapper for the requests.get funcion. It runs the _get_free_proxies function
        and rotates through each proxy to complete the request. The proxy list is saved
        as a class property. Every time the connection to a proxy fails, this function
        removes that proxy from the proxies' list.

        Args:
        - timeout (int): how many seconds to wait before throwing TimeoutError (default: 5)

        Returns: the request's response

        '''

        if not hasattr(self,"real_ip"):
            self.real_ip = True

        if self.real_ip: # do not retry with real IP if it failed already
            print('First try with real IP')
            response = requests.get(url,headers=self._headers)

            if response.status_code == 200:
                return response
            
            print('Failed using real IP. Trying with proxies')
            
            self.real_ip = False

        # Rotating through proxies  
        if not hasattr(self,"proxies"): 
            self._get_free_proxies()

        while (self.proxies): # as long as there are proxies
            proxy = self.proxies[0]
            print(f'Trying to connect to {url} from ip {proxy}')
            dict_proxy = {
                'http':proxy,
                'https':proxy
            }
            self._headers['User-Agent'] = get_random_user_agent()
            try:
                response = requests.get(url, headers=self._headers, proxies=dict_proxy, timeout=timeout)
                response.raise_for_status()
                return response
            except ProxyError:
                print(f'Connection to proxy {proxy} has failed. Moving to next proxy.')
            except HTTPError:
                print(f'Response code: {response.status_code}. Descr: {response.reason}')
            except:# (TimeoutError,ConnectTimeout,ConnectionError):
                current_user_agent = self._headers['User-Agent']
                print(f'Connection failed to {url} using proxy {proxy} and user_agent {current_user_agent}')
            
            # if the proxy does not work, removes it
            self.proxies.remove(proxy)
            print(f'{len(self.proxies)} proxies left')

        del self.proxies

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
        soup = BeautifulSoup(response.text,'lxml')
        
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
            self.wait_time()
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
        self.areas_df.to_csv('./areas_df.csv', index=False)

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
                self.areas_df = pd.read_csv('./areas_df.csv')
            except:
                raise 'Please run get_areas_df first'
            
            print('areas_df found.')
        
        
        for area in self.areas_df[self.areas_df['done']==False].itertuples():
            path = f'{self._IDEALISTA_HOSTNAME}{area.area_url}pagina-{str(area.page)}.htm?ordenado-por=fecha-publicacion-asc'
            print(f'Getting properties\' url for {area.area_name}')
            page = int(area.page)

            while True:
                print(f'Path is {path}')
                response = self._proxy_requests(path)
                if response.status_code != 200:
                    raise f"Cannot retrieve data for page {page} of {area.area_name}"
                soup = BeautifulSoup(response.text,'lxml')
                
                # extracting links from page and creating a 3 columns df
                thisPageLinks = [[area.area_name,self._IDEALISTA_HOSTNAME+x.get('href'),False] for x in soup.select('a.item-link')]
                thisPageLinks_df = pd.DataFrame(thisPageLinks, columns=['area_name','property_link','done'])

                # create or concat data to properties_links_df
                if hasattr(self,'properties_links_df'):
                    self.properties_links_df = pd.concat([self.properties_links_df, thisPageLinks_df])
                else:
                    try:
                        self.properties_links_df = pd.concat(pd.read_csv('./properties_links_df.csv'),thisPageLinks_df)
                    except:
                        self.properties_links_df = thisPageLinks_df.copy()
                
                header = not os.path.exists('./properties_links_df.csv')
                thisPageLinks_df.to_csv('./properties_links_df.csv', mode='a', index=False, header=header)

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
                    self.areas_df.to_csv('./areas_df.csv', index=False)
                else:
                    # done with this area
                    self.areas_df.at[area.Index,"done"] = True
                    print(f'all properties\' links from {area.area_name} have been extracted')
                    break
                
                self.wait_time()

            self.properties_links_df = pd.read_csv('./properties_links_df.csv')

    def get_properties_data(self) -> None:
        '''
        Has to run after generate_properties_links_df. It takes the properties links, 
        access them and get properties data out of them. It creates a new class 
        property "dataset"

        '''

        if not hasattr(self,'properties_links_df'):
            try:
                self.properties_links_df = pd.read_csv('./properties_links_df.csv')
            except:
                raise 'Cannot find properties\' links'

        print(f'Properties\' links found: {self.properties_links_df.shape[0]}.')

        for row in self.properties_links_df[self.properties_links_df['done']==False].itertuples():
            self._get_single_property_data(row)

    def _get_single_property_data(self,row) -> None:
        print(row)
        response = self._proxy_requests_api(row.property_link)
        if response.status_code == 404:
            print(f'Property not found {row.property_link}')
            return pd.DataFrame()
        

        soup = BeautifulSoup(response.text,'html.parser')
        
        print(f'Soup parsed for {row.property_link}')

        if soup.select('#notFoundWithSuggestions'):
            print(f'This property has been removed')
            self.properties_links_df = self.properties_links_df.drop(row.Index)
            self.properties_links_df.to_csv('./properties_links_df.csv',index=False)
            return pd.DataFrame()
        
        try:
            utag_script = list(filter(lambda x: 'utag_data' in x.get_text(),soup.select('script')))[0]
            utag_data = json.loads(str(utag_script).split(';')[0].split(' ')[7])
        except:
            print(f'Cannot retrieve data for {row.property_link}')
            return pd.DataFrame()
        property_data = {
            'id':utag_data['ad']['id'],
            'propertyType':soup.select_one('.main-info .typology').text.strip().lower(),
            'title':soup.select_one('.main-info .txt-body').text.strip().lower(),
            'locationId':utag_data['ad']['address']['locationId'],
            'link':row.property_link,
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
            'featureTags': [x.get_text() for x in soup.select('info-features-tags')]
        }

        heatingData = list(filter(lambda x: 'calefacción' in x.get_text().lower(),soup.select('.details-property_features li')))
        if heatingData:
            property_data['heatingType'] = heatingData[0]
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
            floor_info = [x for x in info_features if re.search("bajo|sotano|planta", x.get_text().lower())]
            if floor_info:
                property_data['floor'] = floor_info[0].select_one('span').get_text().lower().strip()
            else:
                property_data['floor'] = "no info"
        else:
            property_data['floor'] = property_data['interiorExterior'] = "does not apply"
        
        property_data_df = pd.DataFrame.from_dict(property_data,orient='index').T
        print(f'Data converted to DF for {row.property_link}')
            
        # create or concat data to idealista_dataset
        if hasattr(self,'dataset'):
            self.dataset = pd.concat([self.dataset, property_data_df])
        else:
            try:
                self.dataset = pd.concat(pd.read_csv('./idealista_dataset.csv'), property_data_df)
            except:
                self.dataset = property_data_df.copy()

        header = not os.path.exists('./idealista_dataset.csv')
        property_data_df.to_csv('./idealista_dataset.csv', mode='a', index=False, header=header)


        self.properties_links_df.at[row.Index,'done'] = True
        self.properties_links_df.to_csv('./properties_links_df.csv',index=False)

        print(f'Property {row.Index+1} of {self.properties_links_df.shape[0]} done')

    def get_location_ids_mapper(self) -> dict:
        '''
        By default, every record in the dataset has de feature "location id" which
        is the website own id to identify the district. This function scrape and
        creates a new class property named "location_ids_mapper", a dict to map 
        location ids to district's names

        '''
        
        response = self._proxy_requests(self._url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text,'html.parser')

        locations_list = [(x.get('data-location-id'),x.select_one('a').get_text()) for x in soup.select('.breadcrumb-dropdown-subitem-element-list')]

        self.location_ids_mapper = {key:value for key,value in locations_list}

    def full_scrape(self) -> None:
        self.get_areas_df()
        
        print('Found '+self.areas_df['n_houses'].sum()+' houses')

        self.generate_properties_links_df()

        self.properties_links_df = self.properties_links_df.drop_duplicates()

        self.get_properties_data()

        self.get_location_ids_mapper()
        self.dataset['area_name'] = self.dataset['locationId'].map(self.location_ids_mapper)


url = 'https://www.idealista.com/venta-viviendas/madrid-madrid/'
scraper = idealista_scraper(url)

scraper.get_properties_data()

scraper.full_scrape()

idealista_dataset = scraper.dataset.copy()

idealista_dataset['locationId'] = idealista_dataset['locationId'].apply(lambda x: "-".join(x.split("-")[:8]))
idealista_dataset['area_name'] = idealista_dataset['locationId'].map(scraper.location_ids_mapper)
idealista_dataset['price_m2'] = (idealista_dataset['price'] / idealista_dataset['size']).round(2)

# Idealista often suffer from fraudolent announces so it's better to aggregate 
# the price_m2_per_area using the median as it is more robust than the mean
price_m2_per_area = idealista_dataset[['area_name','price_m2']].groupby(['area_name']).median().reset_index()
price_m2_per_area = {key:value for key,value in price_m2_per_area.values}

idealista_dataset['area_price'] = idealista_dataset['area_name'].map(price_m2_per_area).astype(float)