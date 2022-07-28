from latest_user_agents import get_random_user_agent
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ProxyError,HTTPError,ConnectTimeout
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
        self.proxies = []
        self.scraping_explicit_wait_time = 5
        self.IDEALISTA_HOSTNAME = 'https://www.idealista.com'

    def get_free_proxies(self) -> None:
        '''
        This function scrapes the free-proxy-list.net website looking for
        free proxies.

        Args:
        - headers: the user agent needed to avoid 403 error when accessing the free-proxy-list.net website

        Returns: a pandas dataframe with the free proxies' table
        '''
        endpoint = 'https://api.proxyscrape.com/v2/'
        params = {
            'request':'displayproxies',
            'protocol':'http',
            'timeout':10000,
            'country':'gb,es,pt,it,fr,de,nl', # or all
            'ssl':'all',
            'anonymity':'all'
        }

        response = requests.get(endpoint,params=params)
        
        self.proxies = response.text.split('\r\n')[:-1]
    
    def wait_time(self) -> None:
        '''
        Wrapper for time.sleep function to wait an amount of seconds that
        is based on the class property "scraping_explicit_wait_time" multiplied
        for a random number (from 0 to 1) generated with numpy

        No arguments

        '''
        time.sleep(self.scraping_explicit_wait_time*np.random.rand())

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
        # First try with real IP
        response = requests.get(url,headers=self.headers)
        if response.status_code == 200:
            return response
        
        # Rotating through proxies  
        if len(self.proxies) == 0: 
            self.get_free_proxies()
            print(f'found {len(self.proxies)} proxies')

        while (self.proxies): # as long as there are proxies
            proxy = self.proxies[0]
            print(f'Trying to connect to {url} from ip {proxy}')
            dict_proxy = {
                'http':proxy,
                'https':proxy
            }

            try:
                response = requests.get(url, headers=self.headers, proxies=dict_proxy, timeout=timeout)
                response.raise_for_status()
                return response
            except ProxyError:
                print(f'Connection to proxy {proxy} has failed. Moving to next proxy.')
            except HTTPError:
                print(f'Response code: {response.status_code}. Descr: {response.reason}')
            except (TimeoutError,ConnectTimeout):
                current_user_agent = self.headers['User-Agent']
                print(f'Connection failed to {url} using proxy {proxy} and user_agent {current_user_agent}')
            
            # if the proxy does not work, removes it
            self.proxies.remove(proxy)
            print(f'{len(self.proxies)} proxies left')
            #user_agent = get_random_user_agent()
            #self.headers['User-Agent'] = user_agent

        return response

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
            print(f'recursion over: {area_url.geturl()}')
            areas = self.links_from_breadcrumb(area_url.geturl())
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

        
    def generate_houses_df(self) -> None:
        '''
        Has to be run after get_areas_df as it needs areas_df to work.
        It iterates through each area stored in areas_df DataFrame, and extract
        houses' links from each page of the area (district or subdistrict)

        It creates a new class property "houses_list"

        '''

        if not hasattr(self,'areas_df'):
            raise 'Please run get_areas_df first'
            

        self.houses_list = []
        for area in self.areas_df[self.areas_df['done']==False].itertuples():
            path = f'{area.area_url}pagina-{str(area.page)}.htm?ordenado-por=fecha-publicacion-desc'
            
            while True:
                response = self.proxy_requests(self.IDEALISTA_HOSTNAME+path)
                soup = BeautifulSoup(response.text,'lxml')
                thisPageLinks = [[area.area_name,self.IDEALISTA_HOSTNAME+x.get('href')] for x in soup.select('a.item-link')]
                self.houses_list.extend(thisPageLinks)

                # if there is a next page
                next_page = soup.select_one('.pagination .next a')
                if next_page:
                    # done with this page
                    path = self.IDEALISTA_HOSTNAME+next_page.get('href')
                    # storing next page on area_df in case this breaks
                    self.areas_df['page'].iloc[area.index] = area.page+1
                else:
                    # done with this area
                    break
                
                self.wait_time()

            self.areas_df['done'].iloc[area.index] = True
        
    
    def get_houses_data(self) -> pd.DataFrame:
        '''

        '''

        if not hasattr(self,'houses_list'):
            raise 'Cannot find properties\' links'


        for area_name, link in self.houses_list:
            response = self.proxy_requests(link)
            soup = BeautifulSoup(response,'lxml')
            
            utag_script = [x for x in soup.select('script') if 'utag_data' in x.get_text()]
            utag_data = json.loads(str(utag_script[0]).split(';')[0].split(' ')[7])
            house_data = {
                'id':utag_data['ad']['id'],
                'area':area_name,
                'link':link,
                'price':utag_data['ad']['price'],
                'size':utag_data['ad']['characteristics']['constructedArea'],
                'hasParking':utag_data['ad']['characteristics'].get('hasParking',0), # if does not exist, get 0
                'roomNumber':utag_data['ad']['characteristics']['roomNumber'],
                'bathNumber':utag_data['ad']['characteristics']['bathNumber'],
                'hasSwimmingPool':utag_data['ad']['characteristics'].get('hasSwimmingPool',0), # if does not exist, get 0
                'hasTerrace':utag_data['ad']['characteristics'].get('hasTerrace',0), # if does not exist, get 0
                'hasGarden':utag_data['ad']['characteristics'].get('hasGarden',0), # if does not exist, get 0
                'hasLift':utag_data['ad']['characteristics'].get('hasLift',0), # if does not exist, get 0
                'isGoodCondition':utag_data['ad']['condition']['isGoodCondition'],
                'isNeedsRenovating':utag_data['ad']['condition']['isNeedsRenovating'],
                'isNewDevelopment':utag_data['ad']['condition']['isNewDevelopment']
            }
            
            if hasattr(self,'dataset'):
                self.dataset = self.dataset.append(house_data, ignore_index=True)
            else:
                self.dataset = pd.DataFrame(house_data, index=0)

            self.dataset.to_csv('idealista_dataset.csv',mode='a')

            print(f'Property {self.houses_list.index([area_name,link])+1} of {len(self.houses_list)}')


    def scrape(self, starting_url:str) -> None:
        self.get_areas_df(starting_url)
        
        print('Found '+self.areas_df['n_houses'].sum()+' houses')

        self.generate_houses_list()


        






scraper = idealista_scraper()

url = 'https://www.idealista.com/venta-viviendas/madrid-madrid/'

areas_list = scraper.get_areas_list(url)





