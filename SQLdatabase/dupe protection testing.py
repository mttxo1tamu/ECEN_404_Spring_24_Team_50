########################################   
# Imports required for SQL
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pypyodbc
import pandas as pd

###################################
# declares the variables and defines them for the server connections, along with the table names that are going to be assigned
SERVER_NAME = ''
DATABASE_NAME = ''
TABLE_NAME = 'historical_data'

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

###################################
#testing the duplication portion here, should output a mesage if it fails

# Load DataFrame
# Assuming df is your DataFrame, Dropping the unnamed

uploading_df = pd.read_csv(r"C:\final_dataframe_WEST_fixed.csv")
#print(uploading_df)

if 'Unnamed: 0' in uploading_df.columns:
        uploading_df.drop(columns=['Unnamed: 0'], inplace=True)

# Convert 'time' column in DataFrame to datetime64[ns] type
uploading_df['Time'] = pd.to_datetime(uploading_df['Time'])

columnname = 'Location'
firstcolumn = uploading_df.pop(columnname)
uploading_df.insert(0, columnname, firstcolumn)

columnstoadddecimal = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Wind Gust']

uploading_df[columnstoadddecimal] = uploading_df[columnstoadddecimal].apply(lambda x: x / 1.0)

uploading_df.columns = uploading_df.columns.str.lower()

# Query the SQL database to fetch the rows
sql_query = "SELECT * FROM historical_data"
sql_df = pd.read_sql_query(sql_query, engine)
print('sql_df')
print(sql_df.dtypes)
print(sql_df)
print('df')
print(uploading_df.dtypes)
print(uploading_df)

# Resetting index for both DataFrames
uploading_df_reset = uploading_df.reset_index(drop=True)
sql_df_reset = sql_df.reset_index(drop=True)
count = 0
# Resetting the index of both dataframes
uploading_df_reset = uploading_df.reset_index(drop=True)
sql_df_reset = sql_df.reset_index(drop=True)

# Merging DataFrames
merged_df = pd.merge(uploading_df_reset, sql_df_reset, on=['location', 'time', 'temperature', 'dew point', 'humidity', 'wind', 'wind speed', 'wind gust', 'pressure', 'precip.', 'condition', 'power'])
print(merged_df)

# Check if any rows are matched
if not merged_df.empty:
    print("Found the following amount of matching rows")
else:
    print("No matching rows found.")

count = len(merged_df)
print(count)

# Find the indices of rows in uploading_df_reset that are not present in merged_df
indices_to_remove = ~uploading_df_reset.index.isin(merged_df.index)

# Filter uploading_df_reset to keep only the rows not present in merged_df
uploading_df_remaining = uploading_df_reset[indices_to_remove]

# Print or further process uploading_df_remaining as needed
print('The following dataframe is what remains from what is to be uploaded after finding matching row information')
print(uploading_df_remaining)
