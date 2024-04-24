##Automated daily forecasting script from current weather forecast


import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pymilvus
import time as time
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from sqlalchemy import create_engine, MetaData, delete, Table, Column
import sqlalchemy
from sqlalchemy.engine import URL
import pypyodbc
import docker


start_time = time.time()

plt.rcParams['figure.figsize'] = [12, 5] #plot size 10"w x 5"h

# set up embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#initialize df for access outside of loop
df = pd.DataFrame()

####################################

#open docker and run milvus
client = docker.from_env()

container1 = client.containers.get("1d3c8ee64db3")
container2 = client.containers.get("ecd05052c85a")
container3 = client.containers.get("fc26d9202b9b")

container1.start()
container2.start()
container3.start()

time.sleep(60) #wait 1 min to ensure vdb has launched

####################################
# connect to milvus vdb
connections.connect(
  alias="default",
  user='root',
  password='tecafs',
  host='localhost',
  port='19530'
)

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

VDB_daily_forecast = metadata.tables['VDB_daily_forecast']

conn = engine.connect()

clear_table = VDB_daily_forecast.delete()
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
    FROM weather_forecasts
    WHERE Location = '{location}'
    """
    
    # Execute the query with user input as parameters
    df = pd.read_sql_query(sql_query, engine)

    ##################################
    
    #fix df formatting
    #df = df.drop(columns=['dew point', 'wind', 'wind gust', 'pressure', 'precip.'])
    df = df.rename(columns={"temp.": "temperature"})
    df['year'] = df['time'].dt.year.astype(int)
    df['date'] = df['time'].dt.strftime('%m-%d')
    df['hour'] = df['time'].dt.strftime('%H:%M')
    

    df = df.sort_values(by=['location', 'time']) #sorts by datetime within each location
    df = df.reset_index(drop=True)
    
    #reorder columns to match vdb
    #df = df[['vector', 'location', 'time', 'year', 'date', 'hour', 'temperature', 'humidity', 'wind speed', 'condition', 'power']]
        
    ###################################

    #define embedding function
    def get_embedding(prevec):
        text = prevec
        embedding = model.encode(text, show_progress_bar = False)
        return embedding
    
    # Create the prevec conglomeration by combining values from other columns and embed it into 'vector' column 
    df['vector'] = df.apply(lambda row: get_embedding(f"{row['year']}/{row['date']}/{row['hour']}/{row['hour']}/{row['hour']}/{row['temperature']}/{row['temperature']}/{row['humidity']}/{row['wind']}/{row['conditions']}"), axis=1)

    ##################################
    
    collection = Collection("historical_data_21_22")  #get existing milvus collection
    collection.load()  #load collection
    
    def search(text, hour, date, year, location, top_k = 10):
        # AUTOINDEX does not require any search params, chosen to specify metric
        search_params = {
          "metric_type": "L2",
          
        }
        
        #do similarity search
        results = collection.search(
          data = [text],
          anns_field='vector',
          param=search_params,
          limit = top_k,  # Limit to top_k results per search
          expr='location=="{0}"'.format(location),
          output_fields=['index', 'location', 'year', 'date', 'hour', 'temperature', 'humidity', 'wind_speed', 'condition', 'power']
        )
        avg_power = 0
        
        #sum the top_k power values
        for hit in  results[0]:
            #print (hit.entity.get(['location', 'year', 'date', 'hour', 'temperature', 'humidity', 'wind_speed', 'condition', 'power']))
            avg_power += (hit.entity.get('power')*(1.015**(year-hit.entity.get('year')))) #normalize to current year equivalent - 1.5% increase per year
            
        
        #calculate average power, adjust to tune for best results
        month = int(date.split('-')[0])
        hr = int(hour.split(':')[0])
        
        yearly_amp = (-1000*math.sin((math.pi/6)*(month-1)+1))+(300*math.sin((math.pi/3)*(month-1)+1))+1700
        
        #avg_power= (-(yearly_amp)*math.sin((math.pi/12)*hr))+(avg_power/top_k)
        #avg_power= (-500*math.sin((math.pi/12)*hr))+(avg_power/top_k)
        avg_power= (avg_power/top_k)
        return avg_power

        #normalization percentage variable throughout year on bell curve - 0-5%
        #amplitude boost variable throughout year on bell curve (start above 0 boost, increase to max boost in august), also as a function of current value

    ##################################

    df['forecast_power'] = df.apply(lambda row: search(row['vector'], row['hour'], row['date'], row['year'], row['location']), axis=1)  
    
    ##################################
    
    #apply savitsky-golay filter for smoothing
    df['forecast_savgol'] = df[['forecast_power']].apply(lambda x: savgol_filter(x,18,3))
    
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
    print(f"Forecast Generation Time: {location_elapsed_time} seconds")
    print('-------------------------------')

    '''
    #check for missing hours in the forecast
    expected_diff = pd.Timedelta(hours=1)
    rollover_diff = pd.Timedelta(hours=23)
    
    #convert time to datetime
    df['time'] = pd.to_datetime(df['time'], format='%H:%M')
    
    # Calculate the actual difference between each consecutive datetime
    time_diff = df['time'].diff()
    
    # Find the rows where the difference is not as expected
    missing_hours = df[(time_diff != expected_diff) & (time_diff != rollover_diff)]
    missing_hours['time'] = missing_hours['time'].dt.strftime('%H:%M')
    
    # Print a notification if there are missing hours
    if not missing_hours.empty:
      print("Notification: Missing hour(s) detected between the following timestamps:")
      print(missing_hours[['date', 'time']])
    else:
      print("No missing hours detected.")
    
    #convert time back to strings
    df['time'] = df['time'].dt.strftime('%H:%M')
    '''
    df = df.drop(columns=['vector']) #'time_diff', 'missing_hours'
    
    #print(df)
    
    #Format forecast for insertion
    df = df.drop(columns=['forecast_power'])
    
    new_column_name = 'forecast power'
    df.rename(columns={'forecast_savgol': new_column_name}, inplace=True)
    
    #Add to SQL
    df.to_sql('VDB_daily_forecast', con=engine, if_exists='append', index=False)


end_time = time.time()
elapsed_time = end_time - start_time
print("Added all forecasts to SQL")
print(f"Total Generation Time: {elapsed_time} seconds")


