from env import user, password, host
import os
import pandas as pd

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_telco_data():
    sql_query = 'SELECT * FROM customers'
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco_df.csv')
    return df

def get_telco_data(cached=False):
    '''
    This function reads in telco data from Codeup database if cached == False
    or if cached == True reads in telco df from a csv file, returns df
    '''
    if cached or os.path.isfile('telco_df.csv') == False:
        df = new_telco_data()
    else:
        df = pd.read_csv('telco_df.csv', index_col=0)
    return df