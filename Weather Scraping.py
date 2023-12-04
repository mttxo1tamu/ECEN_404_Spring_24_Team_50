"""
ECEN 403 - 904
Group # 50
Ernesto Fuentes
Code derived with help from Nicholas Tran
Subsystem - Data Scrapping
"""
import threading
import multiprocessing
from urllib3.exceptions import ConnectTimeoutError
from queue import Queue, Empty
import concurrent.futures
import json
import os
from bs4 import BeautifulSoup




from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.keys import Keys
import requests, sys, re
import pandas as pd

# URL with wunderground weather information
url = 'https://www.wunderground.com/history/daily/us/tx/fort-worth/KFTW/date/2023-11-4'

# Commands related to the webdriver (not sure what they do, but I can guess):
bi = FirefoxBinary(r'C:\Program Files (x86)\Mozilla Firefox\\firefox.exe')
br = webdriver.Firefox()

# This starts an instance of Firefox at the specified URL:
br.get(url)

# I understand that at this point the data is in html format and can be
# extracted with BeautifulSoup:
soup = BeautifulSoup(br.page_source, 'lxml')

# Close the firefox instance started before:
br.quit()

# I'm only interested in the tables contained on the page:
tables = soup.find_all('table')

# Write all the tables into csv files:
for i in range(len(tables)):
    out_file = open('underground11_14' + str(i + 1) + '.csv', 'wb')
    table = tables[i]

    # ---- Write the table header: ----
    table_head = table.findAll('th')
    output_head = []
    for head in table_head:
        output_head.append(head.text.strip())

    # Some cleaning and formatting of the text before writing:
    header = '"' + '"      "'.join(output_head) + '"'
    header = re.sub('\s', '', header) + '\n'
    out_file.write(header.encode(encoding='UTF-8'))

    # ---- Write the rows: ----
    output_rows = []
    rows = table.findAll('tr')
    for j in range(1, len(rows)):
        table_row = rows[j]
        columns = table_row.findAll('td')
        output_row = []
        for column in columns:
            output_row.append(column.text.strip())

        # Some cleaning and formatting of the text before writing:
        row = '"' + '"        "'.join(output_row) + '"'
        row = re.sub('\s', '', row) + '\n'
        out_file.write(row.encode(encoding='UTF-8'))

    out_file.close()




def get_data(url):
    headers = {
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://www.wunderground.com/',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code > 500:
            if "To discuss automated access to WU data please contact" in response.text:
                print("Page %s was blocked by WU. Please try using better proxies\n"%url)
            else:
                print("Page %s must have been blocked by WU as the status code was %d"%(url,response.status_code))
    # Pass the HTML of the page and create  




                # URL with wunderground weather information
        #url = 'https://www.wunderground.com/history/daily/us/tx/fort-worth/KFTW/date/2023-11-4'

        # Commands related to the webdriver (not sure what they do, but I can guess):
        bi = FirefoxBinary(r'C:\Program Files (x86)\Mozilla Firefox\\firefox.exe')
        br = webdriver.Firefox()

        # This starts an instance of Firefox at the specified URL:
        br.get(url)

        # I understand that at this point the data is in html format and can be
        # extracted with BeautifulSoup:
        soup = BeautifulSoup(br.page_source, 'lxml')

        # Close the firefox instance started before:
        br.quit()

        # I'm only interested in the tables contained on the page:
        tables = soup.find_all('table')

        # Write all the tables into csv files:
        for i in range(len(tables)):
            out_file = open('underground11_14' + str(i + 1) + '.csv', 'wb')
            table = tables[i]

            # ---- Write the table header: ----
            table_head = table.findAll('th')
            output_head = []
            for head in table_head:
                output_head.append(head.text.strip())

            # Some cleaning and formatting of the text before writing:
            header = '"' + '"      "'.join(output_head) + '"'
            header = re.sub('\s', '', header) + '\n'
            out_file.write(header.encode(encoding='UTF-8'))

            # ---- Write the rows: ----
            output_rows = []
            rows = table.findAll('tr')
            for j in range(1, len(rows)):
                table_row = rows[j]
                columns = table_row.findAll('td')
                output_row = []
                for column in columns:
                    output_row.append(column.text.strip())

                # Some cleaning and formatting of the text before writing:
                row = '"' + '"        "'.join(output_row) + '"'
                row = re.sub('\s', '', row) + '\n'
                out_file.write(row.encode(encoding='UTF-8'))

            out_file.close()
    except requests.exceptions.ConnectionError:
        print("Connection Refused Error:", url)
        return None
    except ConnectTimeoutError:
        print("Error Connection Timed Out:", url)
        return None

def add_data(products):
    with open('Vector Database\data.txt', 'a') as file:
        file.write(json.dumps(products)+'\n')
        file.close()

# Create worker threads (will die when main exits)
NUMBER_OF_THREADS = 8
def create_workers():
    for _ in range(NUMBER_OF_THREADS):
        t = threading.Thread(target=work)
        t.daemon = True
        t.start()

# Do the next job in the queue
queue = Queue()
def work():
    while True:
            try:
                print("\n Name of the current executing process: ", multiprocessing.current_process().name, '\n')
 
            except Empty:
                return
            except Exception as e:
                print(e)
                continue

def main():
    #f = open('Vector Database\crawled.txt')
    # create multithreading go through file of urls
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    for url in f:
        try:
            pool.submit(get_data('https://www.wunderground.com/history/daily/us/tx/fort-worth/KFTW/date/2023-11-4'))
            print("Thread working ", 'https://www.wunderground.com/history/daily/us/tx/fort-worth/KFTW/date/2023-11-4')
        except Exception as e:
            print(str(e))
            return set()
    pool.shutdown(wait=True)
    #f.close()

#main()
#create_workers()

get_data('https://www.wunderground.com/history/daily/us/tx/fort-worth/KFTW/date/2023-11-4')
