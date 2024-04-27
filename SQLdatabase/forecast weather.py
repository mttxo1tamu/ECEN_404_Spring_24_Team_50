from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

########################################   
# Imports required for SQL
from sqlalchemy import create_engine, MetaData, delete, Table, Column
import sqlalchemy
from sqlalchemy.engine import URL
import pypyodbc
import pandas as pd

###################################
# declares the variables and defines them for the server connections, along with the table names that are going to be assigned
SERVER_NAME = ''
DATABASE_NAME = ''
TABLE_NAME = ''

# makes the connection to the database with the connection string; has the driver, server name, database name, id, and password
connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Uid={''};
    Pwd={''};
"""

###################################
connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
engine = create_engine(connection_url, module=pypyodbc)

################################### Clearing the table of previous content
metadata = MetaData()
metadata.reflect(bind=engine)

weather_forecasts = metadata.tables['weather_forecasts']
conn = engine.connect()

clear_table = weather_forecasts.delete()
conn.execute(clear_table)
conn.commit()

###################################

def remove_units(df):
    for column in df.columns:
        if column not in ['Conditions', 'Time', 'Direction']:
            # Remove units and keep decimals
            df[column] = df[column].astype(str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
    return df

station_array = ['https://www.wunderground.com/hourly/us/tx/houston/29.75,-95.36/date/',
                 'https://www.wunderground.com/hourly/us/tx/tyler/KTYR/date/',
                 'https://www.wunderground.com/hourly/us/tx/lubbock/date/',
                 'https://www.wunderground.com/hourly/us/tx/dallas/date/',
                 'https://www.wunderground.com/hourly/us/tx/mcallen/date/',
                 'https://www.wunderground.com/hourly/us/tx/austin/date/',
                 'https://www.wunderground.com/hourly/us/tx/abilene/date/',
                 'https://www.wunderground.com/hourly/us/tx/midland/date/']
region_array = ['COAST','EAST','WEST','NORTH','NORTH_C','SOUTHERN','SOUTH_C','FAR_WEST']

options = Options()
options.headless = True
driver = webdriver.Firefox(options=options)

current_date = datetime.now().date() + timedelta(days=1)  # Start from the day after the current dat

# Stop 15 days later
end_date = datetime.now().date() + timedelta(days=16)



for i in range(8):
    region_df = pd.DataFrame()  # Initialize an empty dataframe for the region
    current_date = datetime.now().date() + timedelta(days=1)  # Start from the day after the current date
    for j in range(14):  # Ensure current_date doesn't exceed end_date
        attempts = 8
        while attempts > 0:
            try:
                url = f"{station_array[i]}{current_date}"
                print(url)
                print(region_array[i])
                print(current_date)
                driver.get(url)
                tables = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
                for table in tables:
                    newTable = pd.read_html(table.get_attribute('outerHTML'))
                    html_content = table.get_attribute('outerHTML')
                    with StringIO(html_content) as html_buffer:
                        newTable = pd.read_html(html_buffer, header=0)  # Set header=0
                    if newTable:
                        # Filter out rows with no data
                        filtered_table = newTable[0].dropna(how='all')

                        # Assign current date to index
                        filtered_table.index = [current_date] * len(filtered_table)
                        
                        # Apply the function to remove units
                        #filtered_table = remove_units(filtered_table.fillna(''))

                        # Convert the 'Time' column to military time
                        filtered_table['Time'] = pd.to_datetime(filtered_table['Time'], format='%I:%M %p')

                        #Convert 'Time' column to datetime, keeping only time part
                        filtered_table['Time'] = pd.to_datetime(filtered_table['Time']).dt.time

                        #Assign 'current_date' to the "Time" column
                        filtered_table['Time'] = [datetime.combine(current_date, time) for time in filtered_table['Time']]

                        # Splice the 'Wind' column to extract the content after the second whitespace
                        filtered_table['Direction'] = filtered_table['Wind'].str.split(n=2).str[-1]

                        # Apply the function to remove units
                        filtered_table = remove_units(filtered_table.fillna(''))

                        # Add a new column 'Location' based on the region_array
                        filtered_table['Location'] = region_array[i]  # Use the current value of i as index for region_array

                        # Concatenate to region_df
                        region_df = pd.concat([region_df, filtered_table])  

                current_date += timedelta(days=1)
                if(current_date == end_date):
                    attempts = 0

            except Exception as e:
                print(f"Error scraping data for {current_date}: {e}")
                attempts -= 1
                if attempts == 0:
                    print(f"Failed to scrape data for {current_date} after multiple attempts.")
                    break  # Break out of the inner loop when attempts are exhausted
                else:
                    continue  # Continue to the next iteration of the inner loop if attempts remain
        break  # Break out of the outer loop after completing the date range
        
    print(f"All dataframes for region {region_array[i]} concatenated into one dataframe:")
    print(region_df)  # Print or do further processing with the concatenated dataframe
    print(region_df.dtypes)
    #add to SQL
    ####################################################################
    connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
    engine = create_engine(connection_url, module=pypyodbc)

    ###########################################################
    if 'Unnamed: 0' in region_df.columns:
        region_df.drop(columns=['Unnamed: 0'], inplace=True)

    print('inserting data into the database...')
    region_df.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
    print('data insertion complete')
    #region_df.to_csv('testing_task_scheduler.csv')

driver.quit()

