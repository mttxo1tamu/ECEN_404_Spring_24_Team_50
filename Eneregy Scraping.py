from requests import get  # to make GET request
import pandas as pd
import csv
def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

#download("https://www.ercot.com/files/docs/2016/01/07/native_load_2015.xls", "2015_ercot_hourly_load_data.xls")
def splicing(file_name):
    df = pd.read_csv(file_name)
    df[['Time','Temperature', 'DewPoint', 'Humidity', 'Wind', 'WindSpeed', 'WindGust', 'Pressure', 'Precip.', 'Condition']] = df['Time"Temperature"DewPoint"Humidity"Wind"WindSpeed"WindGust"Pressure"Precip."Condition'].str.split('"', expand=True)
    df.drop(['Time"Temperature"DewPoint"Humidity"Wind"WindSpeed"WindGust"Pressure"Precip."Condition'], axis=1, inplace=True)
    df.to_csv(file_name, index=False)
def removechar(file_name1):

    input_file = open(file_name1, 'r')
    output_file = open('output.csv', 'w')

    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    for row in reader:
        # replace $ with empty string in each value
        row = [value.replace("Â", '') for value in row]
        row = [value.replace("°", '') for value in row]
        row = [value.replace("%", '') for value in row]
        #row = [value.replace("F", '') for value in row]
        row = [value.replace("mph", '') for value in row]
        #row = [value.replace("in", '') for value in row]


        # keep only the first and last columns
        #row = [row[0]] + [row[-1]]
        writer.writerow(row)

    input_file.close()
    output_file.close()
#download("https://www.ercot.com/files/docs/2016/01/07/native_load_2015.xls", "2015_ercot_hourly_load_data.xls")
#splicing('underground11_142.csv')
removechar('underground11_142.csv')

