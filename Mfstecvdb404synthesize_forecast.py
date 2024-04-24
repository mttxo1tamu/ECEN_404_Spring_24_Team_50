##Automated daily forecasting script from current weather forecast


import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import time as time
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from sqlalchemy import create_engine, MetaData, delete, Table, Column
import sqlalchemy
from sqlalchemy.engine import URL
import pypyodbc
import docker


start_time = time.time()

plt.rcParams['figure.figsize'] = [12, 5] #plot size 10"w x 5"h

# set up embedding model

#initialize df for access outside of loop
df = pd.DataFrame()


###################################
# connect to SQL
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

connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
engine = create_engine(connection_url, module=pypyodbc)  

###################################

#clear all existing data in the sql table before writing new forecasts
metadata = MetaData()
metadata.reflect(bind=engine)

synthesize_forecast = metadata.tables['synthesize_forecast']

conn = engine.connect()

clear_table = synthesize_forecast.delete()
conn.execute(clear_table)
conn.commit()

###################################


locationslist = ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C', 'SOUTHERN', 'SOUTH_C', 'WEST']

#loop to create forecast for each location
for i in range(8):
    location = locationslist[i]

    location_start_time = time.time()

    # Define the SQL query for current location
    sql_query = f"""
    SELECT *
    FROM VDB_daily_forecast
    WHERE location = '{location}'
    """
    
    # Execute the query with user input as parameters
    df = pd.read_sql_query(sql_query, engine)

    ##################################

    ML_sql_query = f"""
    SELECT *
    FROM ML_forecast
    WHERE location = '{location}'
    """
    
    # Execute the query with user input as parameters
    mldf = pd.read_sql_query(ML_sql_query, engine)

    ##################################

    df = df.sort_values(by=['location', 'time']) #sorts by datetime within each location
    df = df.reset_index(drop=True)

    mldf = mldf.sort_values(by=['location', 'time']) #sorts by datetime within each location
    mldf = mldf.reset_index(drop=True)

    df['ml_power'] = mldf['power']

    df['forecast power'] = df.apply(lambda row: ((row['forecast power'] + row['ml_power'])/2), axis=1)
    
    
    #apply savitsky-golay filter for smoothing
    #df['forecast_savgol'] = df[['forecast_power']].apply(lambda x: savgol_filter(x,18,3))
    
    ##################################
    
    location_end_time = time.time()
    location_elapsed_time = location_end_time - location_start_time
    
    # Write the modified DataFrame back to a new CSV file
    data_folder = Path("/mnt/c/Users/wcans/milvus_compose/ERCOT_Hourly_Load_2022/")
    output_csv_file_path = data_folder / 'dailyforecast_out.csv'  # Replace with the desired output path
    #not enabled at the moment
    #df.to_csv(output_csv_file_path, index=False)

    '''
    #print data
    #create and print plots
    df.plot(x="date", y=["forecast_power", "power", "forecast_savgol"], kind="line")
    # Display the plot
    plt.title('Forecast Power vs Actual Power Over Time')
    plt.xlabel('Date')
    plt.ylabel('Power (megawatt-hours)')
    plt.legend(['Forecast Power', 'Power', 'SavGol Forecast'])
    plt.savefig('power_figure.png')
    plt.show(block=True)
    
    
    df.plot(x='date', y=['percent_error', 'savgol_percent_error'], kind='line')
    # Display the plot
    plt.title('Percent Error')
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.legend(['Forecast Error', 'SavGol Error'])
    plt.savefig('error_figure.png')
    plt.show(block=True)
    
    print('Mean Average Percent Error - base forecast: ', mean_percent_error)
    print('Mean Average Percent Error - savgol forecast: ', savgol_mean_percent_error)'''
    print(f"Location Time: {location_elapsed_time} seconds")
    print('-------------------------------')


    df = df.drop(columns=['ml_power', 'conditions', 'temperature', 'feels like', 'precip', 'amount', 'cloud cover', 'dew point', 'humidity', 'wind', 'pressure', 'direction']) #'time_diff', 'missing_hours'
    
    #print(df)
    
    #Add to SQL
    df.to_sql('synthesize_forecast', con=engine, if_exists='append', index=False)


end_time = time.time()
elapsed_time = end_time - start_time
print("Added all forecasts to SQL")
print(f"Total Generation Time: {elapsed_time} seconds")


