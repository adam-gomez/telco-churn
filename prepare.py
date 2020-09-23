import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def telco_split(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco_data(df):
    '''
    This function reads a dataframe of telco data and
    returns prepped train, validate, and test dfs
    '''
  
    # drop rows where tenure = 0
    df = df[df.tenure != 0]
    
    # change total_charges type from object to float
    df['total_charges'] = pd.to_numeric(df.total_charges)

    # convert gender into numeric data
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})

    # convert partner into numeric data
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})

    # convert dependents into numeric data
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})

    # convert phone_service into numeric data
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})

    # convert multiple_lines into numeric data
    df['multiple_lines_encoded'] = df.multiple_lines.map({'Yes': 1, 'No': 0, 'No phone service': 0})

    # convert online_security into numeric data
    df['online_security_encoded'] = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    # convert online_backup into numeric data
    df['online_backup_encoded'] = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 0})    

    # convert device_protection into numeric data
    df['device_protection_encoded'] = df.device_protection.map({'Yes': 1, 'No': 0, 'No internet service': 0})      
    
    # convert tech_support into numeric data
    df['tech_support_encoded'] = df.tech_support.map({'Yes': 1, 'No': 0, 'No internet service': 0})     
    
    # convert streaming_tv into numeric data
    df['streaming_tv_encoded'] = df.streaming_tv.map({'Yes': 1, 'No': 0, 'No internet service': 0})        

    # convert streaming_movies into numeric data
    df['streaming_movies_encoded'] = df.streaming_movies.map({'Yes': 1, 'No': 0, 'No internet service': 0}) 

    # convert paperless_billing into numeric data
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0}) 

    # convert churn into numeric data
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0}) 

    # -- New Features ---------------------------------------------------------------- 

    # encode number_phone_lines by utilizing information from multiple_lines
    df['number_phone_lines'] = df.multiple_lines.map({'Yes' : 2, 'No': 1, 'No phone service': 0})
    
    # encode number_relationships by utilizing information from dependents_encoded and partner_encoded
    df['number_relationships'] = df['dependents_encoded'] + df['partner_encoded']

    # encode number_streaming_services by utilizing information from streaming_tv_encoded and streaming_movies_encoded
    df['number_streaming_services'] = df['streaming_tv_encoded'] + df['streaming_movies_encoded']

    # encode number_online_services by utilizing information from online_security_encoded and online_backup_encoded
    df['number_online_services'] = df['online_security_encoded'] + df['online_backup_encoded']

    # encode tenure in years (rounded down) by utilizing information from tenure (currently stored in months)
    df['yearly_tenure'] = df.tenure.apply(lambda x: math.floor(x/12))

    # -- Breaking up Features ---------------------------------------------------------

    # encode internet_service_type into dummy columns, drop "no internet" column
    telco_dummies = pd.get_dummies(df.internet_service_type_id)
    telco_dummies = telco_dummies.rename(columns={1 : "dsl_encoded", 2: "fiber_optic_encoded"})
    telco_dummies.drop(columns=3, inplace = True)

    # join dummy columns back to df
    df = pd.concat([df, telco_dummies], axis=1)

    # encode has_internet_encoded
    df['has_internet'] = df.internet_service_type_id.apply(lambda x: 0 if x == 3 else 1)

    # encode payment_type_id into dummy columns
    telco_dummies = pd.get_dummies(df.payment_type_id)
    telco_dummies = telco_dummies.rename(columns={1: "electronic_check_encoded", 2: "mailed_check_encoded", 3: "bank_transfer_encoded", 4: "credit_card_encoded"})

    # join dummy columns back to df
    df = pd.concat([df, telco_dummies], axis=1)

    # drop the payment_type_id column
    #df.drop(columns=['payment_type_id'], inplace = True)

    # encode contract_type_id into dummy columns
    telco_dummies = pd.get_dummies(df.contract_type_id)
    telco_dummies = telco_dummies.rename(columns={1: "month_to_month_encoded", 2: "one_year_contract_encoded", 3: "two_year_contract_encoded"})

    # join dummy columns back to df
    df = pd.concat([df, telco_dummies], axis=1)

    # split data into train, validate, test dfs stratify churn
    train, validate, test = telco_split(df)
    
    return train, validate, test



# -- Creating a DataFrame for predictions after modeling is complete --
def prep_telco_data_prediction(df):
    '''
    This function reads a dataframe of telco data and
    returns prepped train, validate, and test dfs
    '''
  
    # drop rows where tenure = 0
    df = df[df.tenure != 0]
    
    # change total_charges type from object to float
    df['total_charges'] = pd.to_numeric(df.total_charges)

    # convert gender into numeric data
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})

    # convert partner into numeric data
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})

    # convert dependents into numeric data
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})

    # convert phone_service into numeric data
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})

    # convert multiple_lines into numeric data
    df['multiple_lines_encoded'] = df.multiple_lines.map({'Yes': 1, 'No': 0, 'No phone service': 0})

    # convert online_security into numeric data
    df['online_security_encoded'] = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    # convert online_backup into numeric data
    df['online_backup_encoded'] = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 0})    

    # convert device_protection into numeric data
    df['device_protection_encoded'] = df.device_protection.map({'Yes': 1, 'No': 0, 'No internet service': 0})      
    
    # convert tech_support into numeric data
    df['tech_support_encoded'] = df.tech_support.map({'Yes': 1, 'No': 0, 'No internet service': 0})     
    
    # convert streaming_tv into numeric data
    df['streaming_tv_encoded'] = df.streaming_tv.map({'Yes': 1, 'No': 0, 'No internet service': 0})        

    # convert streaming_movies into numeric data
    df['streaming_movies_encoded'] = df.streaming_movies.map({'Yes': 1, 'No': 0, 'No internet service': 0}) 

    # convert paperless_billing into numeric data
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0}) 

    # convert churn into numeric data
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0}) 

    # -- New Features ---------------------------------------------------------------- 

    # encode number_phone_lines by utilizing information from multiple_lines
    df['number_phone_lines'] = df.multiple_lines.map({'Yes' : 2, 'No': 1, 'No phone service': 0})
    
    # encode number_relationships by utilizing information from dependents_encoded and partner_encoded
    df['number_relationships'] = df['dependents_encoded'] + df['partner_encoded']

    # encode number_streaming_services by utilizing information from streaming_tv_encoded and streaming_movies_encoded
    df['number_streaming_services'] = df['streaming_tv_encoded'] + df['streaming_movies_encoded']

    # encode number_online_services by utilizing information from online_security_encoded and online_backup_encoded
    df['number_online_services'] = df['online_security_encoded'] + df['online_backup_encoded']

    # encode tenure in years (rounded down) by utilizing information from tenure (currently stored in months)
    df['yearly_tenure'] = df.tenure.apply(lambda x: math.floor(x/12))

    # -- Breaking up Features ---------------------------------------------------------

    # encode internet_service_type into dummy columns, drop "no internet" column
    telco_dummies = pd.get_dummies(df.internet_service_type_id)
    telco_dummies = telco_dummies.rename(columns={1 : "dsl_encoded", 2: "fiber_optic_encoded"})
    telco_dummies.drop(columns=3, inplace = True)

    # join dummy columns back to df
    df = pd.concat([df, telco_dummies], axis=1)

    # encode has_internet_encoded
    df['has_internet'] = df.internet_service_type_id.apply(lambda x: 0 if x == 3 else 1)

    # encode payment_type_id into dummy columns
    telco_dummies = pd.get_dummies(df.payment_type_id)
    telco_dummies = telco_dummies.rename(columns={1: "electronic_check_encoded", 2: "mailed_check_encoded", 3: "bank_transfer_encoded", 4: "credit_card_encoded"})

    # join dummy columns back to df
    df = pd.concat([df, telco_dummies], axis=1)

    # drop the payment_type_id column
    #df.drop(columns=['payment_type_id'], inplace = True)

    # encode contract_type_id into dummy columns
    telco_dummies = pd.get_dummies(df.contract_type_id)
    telco_dummies = telco_dummies.rename(columns={1: "month_to_month_encoded", 2: "one_year_contract_encoded", 3: "two_year_contract_encoded"})

    # join dummy columns back to df
    df = pd.concat([df, telco_dummies], axis=1)
    
    return df