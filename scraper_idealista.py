from bs4 import BeautifulSoup
from latest_user_agents import get_random_user_agent
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ProxyError,HTTPError,ConnectTimeout,ConnectionError
import time
import json


class idealista_scraper:

    def __init__(self) -> None:
        self.headers = {
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
        self.scraping_var_wait_time = 10 # this will be multiplied for a random percentage
        self.scraping_fix_wait_time = 5 # this will be added to wait time as is
        self.IDEALISTA_HOSTNAME = 'https://www.idealista.com'

    def reset_headers(self) -> None:

        self.headers = {
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
    
    def get_free_proxies(self) -> None:
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

    def wait_time(self) -> None:
        '''
        Wrapper for time.sleep function to wait an amount of seconds that
        is based on the property "scraping_var_wait_time" multiplied
        for a random number (from 0 to 1) generated with numpy plus the 
        property "scraping_fix_wait_time". Default config: min 10 sec,
        max 1 min.

        No arguments

        '''
        
        wait_for =  self.scraping_fix_wait_time
        wait_for += self.scraping_var_wait_time*np.random.rand()
        print(f'waiting for {wait_for} seconds')
        time.sleep(wait_for)

    def proxy_requests(self, url:str, timeout:int=5) -> requests.models.Response:
        '''
        Wrapper for the requests.get funcion. It runs the get_free_proxies function
        and rotates through each proxy to complete the request. The proxy list is saved
        as a class property. Every time the connection to a proxy fails, this function
        removes that proxy from the proxies' list.

        Args:
        - url (str): the url for the request
        - timeout (int): how many seconds to wait before throwing TimeoutError (default: 5)

        Returns: the request's response

        '''

        if not hasattr(self,"real_ip"):
            self.real_ip = True

        if self.real_ip: # do not retry with real IP if it failed already
            print('First try with real IP')
            response = requests.get(url,headers=self.headers)

            if response.status_code == 200:
                return response
            
            print('Failed using real IP. Trying with proxies')
            
            self.real_ip = False

        # Rotating through proxies  
        if not hasattr(self,"proxies"): 
            self.get_free_proxies()

        while (self.proxies): # as long as there are proxies
            proxy = self.proxies[0]
            print(f'Trying to connect to {url} from ip {proxy}')
            dict_proxy = {
                'http':proxy,
                'https':proxy
            }
            self.headers['User-Agent'] = get_random_user_agent()
            try:
                response = requests.get(url, headers=self.headers, proxies=dict_proxy, timeout=timeout)
                response.raise_for_status()
                return response
            except ProxyError:
                print(f'Connection to proxy {proxy} has failed. Moving to next proxy.')
            except HTTPError:
                print(f'Response code: {response.status_code}. Descr: {response.reason}')
            except:# (TimeoutError,ConnectTimeout,ConnectionError):
                current_user_agent = self.headers['User-Agent']
                print(f'Connection failed to {url} using proxy {proxy} and user_agent {current_user_agent}')
            
            # if the proxy does not work, removes it
            self.proxies.remove(proxy)
            print(f'{len(self.proxies)} proxies left')

        del self.proxies

    def links_from_breadcrumb(self, thisUrl:str) -> list:
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
        response = self.proxy_requests(thisUrl)
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
        for area,total_houses in more_than_1800_houses:
            self.wait_time()
            area_url = self.IDEALISTA_HOSTNAME+area.get('href')
            print(f'recursion over: {area_url}')
            areas = self.links_from_breadcrumb(area_url)
            ready_for_scraping.extend(areas)
        
        return ready_for_scraping

    def get_areas_df(self, url:str) -> None:
        '''
        Has to be run first. It creates a urls list for each area
        that can be found as a subarea of the url given as input.

        Args:
        url (str): the url of main area to look into

        Writes a new class property named "areas_df" (pandas DataFrame)

        '''

        areas_list = [[x[0].text,x[0].get('href'),x[1]] for x in self.links_from_breadcrumb(url)]
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
            path = f'{self.IDEALISTA_HOSTNAME}{area.area_url}pagina-{str(area.page)}.htm?ordenado-por=fecha-publicacion-asc'
            print(f'Getting properties\' url for {area.area_name}')
            page = int(area.page)

            while True:
                print(f'Path is {path}')
                response = self.proxy_requests(path)
                if response.status_code != 200:
                    raise f"Cannot retrieve data for page {page} of {area.area_name}"
                soup = BeautifulSoup(response.text,'lxml')
                
                # extracting links from page and creating a 3 columns df
                thisPageLinks = [[area.area_name,self.IDEALISTA_HOSTNAME+x.get('href'),False] for x in soup.select('a.item-link')]
                thisPageLinks_df = pd.DataFrame(thisPageLinks, columns=['area_name','property_link','done'])

                # create or concat data to properties_links_df
                if hasattr(self,'properties_links_df'):
                    self.properties_links_df = pd.concat([self.properties_links_df, thisPageLinks_df])
                else:
                    try:
                        self.properties_links_df = pd.concat(pd.read_csv('./properties_links_df.csv'),thisPageLinks_df)
                    except:
                        self.properties_links_df = thisPageLinks_df.copy()
                
                thisPageLinks_df.to_csv('./properties_links_df.csv', mode='a', index=False, header=False)

                print(f'Page {page}: property links added')
                print(f'properties_links_df now has {len(self.properties_links_df)} links')

                # if there is a next page
                next_page = soup.select_one('.pagination .next a')
                if bool(next_page):
                    # done with this page
                    self.headers['Referrer'] = path
                    path = self.IDEALISTA_HOSTNAME+next_page.get('href')
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

    def get_properties_data(self) -> pd.DataFrame:
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
            response = self.proxy_requests(row.property_link)
            if response.status_code == 404:
                continue
            
            response.raise_for_status()

            soup = BeautifulSoup(response,'lxml')
            
            utag_script = list(filter(lambda x: 'utag_data' in x.get_text(),soup.select('script')))[0]
            utag_data = json.loads(str(utag_script).split(';')[0].split(' ')[7])
            property_data = {
                'id':utag_data['ad']['id'],
                'propertyType':soup.select_one('.main-info .typology').text.strip().lower(),
                'title':soup.select_one('.main-info .txt-body').text.strip().lower(),
                'area':utag_data['ad']['address']['locationId'],
                'link':row.property_link,
                'price':utag_data['ad']['price'],
                'size':utag_data['ad']['characteristics']['constructedArea'],
                'hasParking':utag_data['ad']['characteristics'].get('hasParking',0), # if does not exist, get 0
                'roomNumber':utag_data['ad']['characteristics']['roomNumber'],
                'bathNumber':utag_data['ad']['characteristics']['bathNumber'],
                'heatingType': list(filter(lambda x: 'calefacción' in x.text.lower(),soup.select('.details-property_features li')))[0],
                'hasSwimmingPool':utag_data['ad']['characteristics'].get('hasSwimmingPool',0), # if does not exist, get 0
                'hasTerrace':utag_data['ad']['characteristics'].get('hasTerrace',0), # if does not exist, get 0
                'hasGarden':utag_data['ad']['characteristics'].get('hasGarden',0), # if does not exist, get 0
                'hasLift':utag_data['ad']['characteristics'].get('hasLift',0), # if does not exist, get 0
                'isGoodCondition':utag_data['ad']['condition']['isGoodCondition'],
                'isNeedsRenovating':utag_data['ad']['condition']['isNeedsRenovating'],
                'isNewDevelopment':utag_data['ad']['condition']['isNewDevelopment'],
                'featureTags': [x.get_text() for x in soup.select('info-features-tags')]
            }

            if property_data['propertyType'] == 'piso':
                property_data['floor'] = soup.select('.info-features > span')[2].select_one('span').text.strip().lower() or "no info"
            else:
                property_data['floor'] = "does not apply"

            property_data_df = pd.DataFrame(property_data, index=0)
            
            # create or concat data to idealista_dataset
            if hasattr(self,'dataset'):
                self.dataset = pd.concat([self.dataset, property_data_df])
            else:
                try:
                    self.dataset = pd.concat(pd.read_csv('./idealista_dataset.csv'), property_data_df)
                except:
                    self.dataset = property_data_df.copy()

            property_data_df.to_csv('idealista_dataset.csv', mode='a', index=False, header=False)

            self.properties_links_df.at[row.Index,'done'] = True


            print(f'Property {row.Index+1} of {len(self.properties_links_df.shape[0])}')

    def full_scrape(self, starting_url:str) -> None:
        self.get_areas_df(starting_url)
        
        print('Found '+self.areas_df['n_houses'].sum()+' houses')

        self.generate_properties_links_df()

        self.properties_links_df = self.properties_links_df.drop_duplicates()

        self.get_properties_data()



scraper = idealista_scraper()

url = 'https://www.idealista.com/venta-viviendas/madrid-madrid/'

scraper.generate_properties_links_df()


scraper.full_scrape(url)

idealista_dataset = scraper.dataset.copy()

r = requests.get('https://www.idealista.com/venta-viviendas/madrid-madrid/mapa',headers=scraper.headers)
soup = BeautifulSoup(r,"html.parser")
location_id_mapper = [{x.get('data-location-id'):x.find('a').text} for x in soup.select('.breadcrumb-dropdown-subitem-element-list')]

idealista_dataset['area'] = idealista_dataset['area'].apply(lambda x: "-".join(x.split("-")[:8]))
idealista_dataset['area'] = idealista_dataset['area'].map(location_id_mapper)
idealista_dataset['price_m2'] = idealista_dataset['price'] / idealista_dataset['size']

# Idealista often suffer of fraudolent announces so i aggregate 
# the price_m2 per area using median as it is more robust than mean
price_m2_per_area = idealista_dataset[['area','price_m2']].groupby(['area']).median().reset_index()
price_m2_per_area = [{key:value} for key,value in price_m2_per_area.values]

idealista_dataset['area_price'] = idealista_dataset['area'].map(price_m2_per_area).astype(float)
