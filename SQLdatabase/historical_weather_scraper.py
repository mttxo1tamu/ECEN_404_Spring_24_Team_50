from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime, timedelta
import threading
from io import StringIO

import time


########################################   
# Imports required for SQL
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pypyodbc
import pandas as pd

###################################
# declares the variables and defines them for the server connections, along with the table names that are going to be assigned
SERVER_NAME = 'tcp:tecafs.database.windows.net,1433'
DATABASE_NAME = 'TecafsSqlDatabase'
TABLE_NAME = 'hist_data'

# makes the connection to the database with the connection string; has the driver, server name, database name, id, and password
connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Uid={'tecafs2023'};
    Pwd={'Capstone50'};
"""

###################################
connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
engine = create_engine(connection_url, module=pypyodbc)

###################################

# Function to scrape data for a given date range
def scrape_data(start_date, end_date, result_list):
    try:
        # Temporary list to store DataFrame for this thread
        thread_data = []
        
        # does the scraping while opening
        while start_date <= end_date:
            attempts = 8  # Number of retry attempts
            while attempts > 0:
                try:
                    url = f"{station_array[i]}{start_date}"  # assigns the url with appropriate date
                    options = Options()
                    options.headless = True
                    driver = webdriver.Firefox(options=options)
                    driver.get(url)
                    tables = WebDriverWait(driver, 40).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))  # gets total table count on page, currently waits 15 seconds
                    last_table = tables[-1]
                    
                    # Read the last table only
                    html_content = last_table.get_attribute('outerHTML')
                    with StringIO(html_content) as html_buffer:
                        newTable = pd.read_html(html_buffer, header=0)  # Set header=0
                    if newTable:
                        # Filter out rows with no data
                        filtered_table = newTable[0].dropna(how='all')
                        # Assign current date to index
                        filtered_table.index = [start_date] * len(filtered_table)
                        
                        # Apply the function to remove units
                        filtered_table = remove_units(filtered_table.fillna(''))

                        # Convert the 'Time' column to military time
                        filtered_table['Time'] = pd.to_datetime(filtered_table['Time'], format='%I:%M %p')


                        # Filter the rows where the minutes part of the time is equal to 53
                        filtered_table = filtered_table[(filtered_table['Time'].dt.minute == 53) | (filtered_table['Time'].dt.minute == 52)]



                        # Round the times to the nearest hour
                        filtered_table['Time'] = filtered_table['Time'].dt.round('H').dt.strftime('%H:%M')
                        
                        # Create a mask for rows with time 0:00
                        zero_am_mask = filtered_table['Time'] == '00:00'
                        # Extract rows with time 0:00
                        zero_am_rows = filtered_table[zero_am_mask]
                        # Remove the rows with time 0:00 from the DataFrame
                        filtered_table = filtered_table[~zero_am_mask]

                        # Increment the date of rows with time 0:00
                        zero_am_rows.index = zero_am_rows.index + timedelta(days=1)
                        # Concatenate the modified DataFrame
                        filtered_table = pd.concat([filtered_table, zero_am_rows])

                        # Create a mask for rows with time 1:00
                        one_am_mask = filtered_table['Time'] == '01:00'
                        # Extract rows with time 1:00
                        one_am_rows = filtered_table[one_am_mask]
                        # Remove the rows with time 1:00 from the DataFrame
                        filtered_table = filtered_table[~one_am_mask]
                        # Increment the date of rows with time 1:00
                        one_am_rows.index = one_am_rows.index + timedelta(days=1)
                        # Concatenate the modified DataFrame
                        filtered_table = pd.concat([filtered_table, one_am_rows])

                        # Append to temporary list
                        thread_data.append(filtered_table)
                        
                        break  # Exit retry loop if successful
                except Exception as e:
                    print(f"Error scraping data for {start_date}: {e}")
                    attempts -= 1
                    if attempts == 0:
                        print(f"Failed to scrape data for {start_date} after multiple attempts.")
                finally:
                    driver.quit()

            start_date += timedelta(days=1)  # changes the date by one day forward
        
        # Append the temporary list to the result list
        result_list.extend(thread_data)
    except Exception as e:
        print(f"Error scraping data: {e}")



def remove_units(df):
    for column in df.columns:
        if column not in ['Condition', 'Wind', 'Time']:
            # Remove units and keep decimals
            df[column] = df[column].astype(str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
    return df

#open csv, and convert to df for the power data############################################

# base url, currently houston, will pull from text file containing the cities, will have to change weather station accordingly as well, will also be listed in a txt file in respective order to city names
#base_url = 'https://www.wunderground.com/history/daily/us/tx/houston/KHOU/date/' #houston, coast
#base_url = 'https://www.wunderground.com/history/daily/us/tx/tyler/KTYR/date/' #tyler, east
#base_url = 'https://www.wunderground.com/history/daily/us/tx/abilene/KABI/date/'  #Abilene, west 
#base_url = 'https://www.wunderground.com/history/daily/us/tx/lubbock/KLBB/date/' #lubbock, north
#base_url = 'https://www.wunderground.com/history/daily/us/tx/dallas/KDAL/date/' #dallas, north central
#base_url = 'https://www.wunderground.com/history/daily/us/tx/mcallen/KMFE/date/' #mccallen, southern
#base_url = 'https://www.wunderground.com/history/daily/us/tx/austin/KAUS/date/' #austin, south central
#base_url = 'https://www.wunderground.com/history/daily/us/tx/midland/KMAF/date/' #midland, west

#station arrays, along with the region array
station_array = ['https://www.wunderground.com/history/daily/us/tx/houston/KHOU/date/','https://www.wunderground.com/history/daily/us/tx/tyler/KTYR/date/','https://www.wunderground.com/history/daily/us/tx/abilene/KABI/date/','https://www.wunderground.com/history/daily/us/tx/lubbock/KLBB/date/','https://www.wunderground.com/history/daily/us/tx/dallas/KDAL/date/','https://www.wunderground.com/history/daily/us/tx/mcallen/KMFE/date/','https://www.wunderground.com/history/daily/us/tx/austin/KAUS/date/','https://www.wunderground.com/history/daily/us/tx/midland/KMAF/date/']
#3rd in station, 'https://www.wunderground.com/history/daily/us/tx/abilene/KABI/date/',
region_array = ['COAST','EAST','WEST','NORTH','NORTH_C','SOUTHERN','SOUTH_C','FAR_WEST']
#3rd in region, 'WEST',

#importing the power data into a dataframe from the csv file
power_df = pd.read_csv(r"C:\Users\netof\OneDrive\Desktop\capstone\ERCOTData2002_2023_2.0.csv")

# Remove any trailing whitespace and unwanted characters
power_df['Hour_End'] = power_df['Hour_End'].str.replace(' DST', '')

# Convert the 'Hour_End' column to datetime format
power_df['Hour_End'] = pd.to_datetime(power_df['Hour_End'], format="%m/%d/%Y %H:%M")

#print(power_df)
#print(power_df.dtypes)
# date ranges
start_date = datetime(2019, 1, 1)  # change accordingly
end_date = datetime(2020, 12, 31)  # change accordingly
total_days = (end_date - start_date).days
threads_count = 12
delta = total_days // threads_count

# Split the date range into approximately equal parts
date_ranges = [(start_date + i * timedelta(days=delta), start_date + (i + 1) * timedelta(days=delta) - timedelta(days=1)) for i in range(threads_count)]

# Adjust the last date range
date_ranges[-1] = (date_ranges[-1][0], end_date)

#time for the script to run
start_time = time.time()

#start the loop here to go through all the regions rather than just one, put everything after this line into the loop
for i in range(8):
    # Create and start threads
    threads = []
    result_list = []  # List to store the DataFrames from each thread
    for start, end in date_ranges:
        thread = threading.Thread(target=scrape_data, args=(start, end, result_list))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(result_list)


    # Sort the DataFrame by index (date) to arrange rows from earliest to latest date
    final_df = final_df.sort_index()

    # Convert the 'Time' column to military time
    #final_df['Time'] = final_df['Time'].apply(lambda x: datetime.strptime(x, '%I:%M %p').strftime('%H:%M'))

    # Reorder the time within each day from the earliest to the latest
    final_df = final_df.groupby(level=0).apply(lambda x: x.sort_values(by='Time'))

    # Reset index before applying move_early_rows function
    final_df = final_df.reset_index(level=0,drop = True)

    # Apply the function to the DataFrame
    final_df = remove_units(final_df)

    
    # Sort the DataFrame by index (date) and time
    final_df = final_df.sort_index()

#######################################################################################

    # Convert the 'Time' column to datetime format
    final_df['Time'] = pd.to_datetime(final_df['Time'], format='%H:%M')

    # Extract hours and minutes separately
    hours = final_df['Time'].dt.hour
    minutes = final_df['Time'].dt.minute

    # Add the time components to the index (date)
    final_df['Time'] = final_df.index + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')

    # Add a new column 'Location' based on the region_array
    final_df['Location'] = region_array[i]  # Use the current value of i as index for region_array


    ######################### Trying to combine the power data into this same dataframe that already includes the weather
    #final_df['Power'] = final_df['Time'].apply(lambda x: power_df.loc[power_df['Hour_End'] == x, region_array[i]].values[0])
    final_df['Power'] = final_df['Time'].apply(lambda x: power_df.loc[power_df['Hour_End'] == x, region_array[i]].values[0] if not power_df.loc[power_df['Hour_End'] == x, region_array[i]].empty else None)



#########################################################################################
    # Print the DataFrame
    #print(final_df)
    #print(i)
    
    final_df.to_csv('final_dataframe_' + region_array[i] +'2019_2020.csv')
    #final_df.to_csv('coast_weather_more.csv')
    
    #add to SQL
    ####################################################################
    connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
    engine = create_engine(connection_url, module=pypyodbc)

    ###########################################################
    if 'Unnamed: 0' in final_df.columns:
        final_df.drop(columns=['Unnamed: 0'], inplace=True)

    print('inserting data into the database...')
    final_df.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
    print('data insertion complete')

    #############################################################
    #printing the types for the columns of the dataframe
    #print(final_df.dtypes)

    #print(final_df)

#to determine total runtime of the script
end_time = time.time()
elapsed_time = end_time - start_time
print('total elapsed time for the program to run,probably seconds, is as follows')
print(elapsed_time)


#print("final df")
#print(final_df)

# Convert the final DataFrame to a CSV file named 'coast_weather.csv'
#final_df.to_csv('coast_weather_more.csv')
