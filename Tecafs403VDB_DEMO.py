import pandas as pd
import numpy as np
import pymilvus
import time as time
import matplotlib.pyplot as plt
from pathlib import Path
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

# connect to milvus vdb
connections.connect(
  alias="default",
  user='root',
  password='tecafs',
  host='localhost',
  port='19530'
)

# set up embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#while loop allows user to run multiple times
run_again = 'y'
while(run_again == 'y' or run_again == 'yes' or run_again == 'Y' or run_again == 'Yes'):

  # Read in the CSV file
  data_folder = Path("/mnt/c/Users/wcans/milvus_compose/ERCOT_Hourly_Load_2022/")
  csv_file_path = data_folder / "jan23demodata.csv"  # Path to raw 2023 data (pre forecasting)
  forecast_df = pd.read_csv(csv_file_path) #create dataframe

  start_day = int(input('Howdy! Data is available for January 2023. Please enter the number of the day to begin forecasting from (1-31): '))
  num_days = int(input('How many days would you like to forecast? '))
  
  if((32-start_day) < num_days):
      num_days = 32-start_day
  
  #find indices to start and end at
  end_day = start_day+num_days-1
  start_hour = 24*(start_day-1)
  end_hour = (24*end_day)-1

  #filter and set as new dataframe
  filtered_df = pd.DataFrame(forecast_df.iloc[start_hour:(end_hour+1)])

  #define embedding function
  def get_embedding(prevec):
      text = [prevec]
      embedding = model.encode(text)
      return embedding

  #format the prevector data conglomeration as year/date/time/temp/precip/humidity, then get embeddings
  filtered_df['vector'] = filtered_df.apply(lambda row: get_embedding(f"{row['year']}/{row['date']}/{row['time']}/{row['temp']}/{row['precip']}/{row['humidity']}"), axis=1)
  filtered_df['vector'] = filtered_df['vector'].apply(lambda row: row[0])  #fix formatting of vector

  collection = Collection("small_historical_data")  #get existing milvus collection
  collection.load()  #load collection

  def search(text, top_k = 10):
      # AUTOINDEX does not require any search params, chosen to specify metric
      search_params = {
          "metric_type": "L2"
      }

      #do similarity search
      results = collection.search(
          data = [text],
          anns_field='vector',
          param=search_params,
          limit = top_k,  # Limit to top_k results per search
          ##output_fields=['original_question', 'answer']  # Include the original question and answer in the result
          output_fields=['index', 'year', 'date', 'time', 'temp', 'precip', 'humidity', 'power']
      )
      avg_power = 0
      
      #sum the top_k power values
      for hit in  results[0]:
          avg_power += hit.entity.get('power')

      #calculate average power, adjust to tune for best results
      avg_power= 0.98*(avg_power/top_k)
      return avg_power
      
  filtered_df['forecast_power'] = filtered_df.apply(lambda row: search(row['vector']), axis=1)  

  #Calculate percent error for each row
  filtered_df['percent_error'] = ((abs(filtered_df['forecast_power'] - filtered_df['power'])) / filtered_df['power']) * 100
  
  #Calculate overall percent error
  mean_percent_error = filtered_df['percent_error'].mean()

  # Write the modified DataFrame back to a new CSV file
  output_csv_file_path = data_folder / '5daydemoforecast1_out.csv'  # Replace with the desired output path
  #not enabled at the moment
  filtered_df.to_csv(output_csv_file_path, index=False)

  #print data
  #create and print plots
  filtered_df.plot(x="time", y=["power", "forecast_power"], kind="line")
  # Display the plot
  plt.title('Forecast Power vs Actual Power Over Time')
  plt.xlabel('Time')
  plt.ylabel('Power (megawatt-hours)')
  plt.legend(['Power', 'Forecast Power'])
  plt.savefig('power_figure.png')


  filtered_df.plot(x='time', y=['percent_error'], kind='line')
  # Display the plot
  plt.title('Percent Error')
  plt.xlabel('Time')
  plt.ylabel('Percentage')
  plt.savefig('error_figure.png')



  print('Mean Average Percent Error: ', mean_percent_error)
  print('-------------------------------')

  filtered_df = filtered_df.drop(columns=['vector'])
  print(filtered_df)
  
  # User Interface
  run_again = input('Would you like to run again? (y/n)')
  

