"""
ECEN 404 - 901
Group # 50
Ernesto Fuentes
Code derived with help from Nicholas Tran
Subsystem - Data Scrapping
"""
#####################   
# Imports required
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pypyodbc
import pandas as pd

###################################

# declares the variables and defines them for the server connections, along with the table names that are going to be assigned
SERVER_NAME = ''
DATABASE_NAME = ''
TABLE_NAME = 'clean_data'

###################################

# makes the connection to the database with the connection string; has the driver, server name, database name, id, and password
connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Uid={''};
    Pwd={''};
"""
###################################

# attempted connection string, didn't work though

#connection_string1 = pypyodbc.connect("Driver={ODBC Driver 18 for SQL Server};Server=tcp:tecafs.database.windows.net,1433;Database=TecafsSqlDatabase;Uid=tecafs2023;Pwd={Capstone50};")

###################################


connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': connection_string})
engine = create_engine(connection_url, module=pypyodbc)


excel_file = pd.read_excel(r"C:\Users\netof\OneDrive\Desktop\capstone\2013_ercot_hourly_load_data.xls", sheet_name=None)
for sheet_name, df_data in excel_file.items():
    print(f'Loading worksheet {sheet_name}...')
    #{'fail', 'replace', 'append'}
    df_data.to_sql(TABLE_NAME, engine, if_exists='append', index=False)

###################################
