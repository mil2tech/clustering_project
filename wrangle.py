from pandas import DataFrame
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import acquire

def prep_familyhome17():
    df = acquire.get_zillow_data()
    # Code to fill nulls in on both fields
    df["fireplacecnt"].fillna( 0 , inplace = True)
    df['fireplaceflag'] = np.where(df.fireplacecnt > 0, 1 , 0)
    # fill in nulls of fields related to pool and hottub/spa
    df["hashottuborspa"].fillna( 0 , inplace = True)
    df["pooltypeid7"].fillna( 0 , inplace = True)
    df["pooltypeid10"].fillna( 0 , inplace = True)
    df["pooltypeid2"].fillna( 0 , inplace = True)
        # create new column for pool cont since there is an discripency in pool count column with this condition
    def conditions(df):
        if (df['pooltypeid7'] > 0) or (df['pooltypeid2'] > 0) :
            return 1
        else:
            return 0
    df['haspool'] = df.apply(conditions, axis=1)
    # fill in columns related to garage
    df["garagecarcnt"].fillna( 0 , inplace = True)
    df["garagetotalsqft"].fillna( 0 , inplace = True)
    # code fixing columns
    df['taxdelinquencyflag'].fillna( 'N' , inplace = True)
    df['yearsdeliquent'] = 16 - df.taxdelinquencyyear
    df['yearsdeliquent'] = df['yearsdeliquent'].replace([-83.00], 17)
    df['yearsdeliquent'].fillna( 0 , inplace = True)
    # fix latitude and longitude
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000
    # Code to fill nulls in
    df["basementsqft"].fillna( 0 , inplace = True)
    df['home_age'] = 2017 - df.yearbuilt
    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100
    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt


    # drop observations with nulls in column
    #df = df[pd.notnull(df['yearbuilt'])]
    # Droped columns
    #df = df.drop(columns=['hashottuborspa', 'poolcnt','calculatedbathnbr', 'finishedsquarefeet12','fullbathcnt', 'propertycountylandusecode', 'propertylandusetypeid', 'regionidcounty', 'regionidcity','propertyzoningdesc', 'roomcnt', 'rawcensustractandblock','censustractandblock'])
    # convert data types
    #df['bedroomcnt'] = df.bedroomcnt.astype(int)
    #df['yearbuilt'] = df.yearbuilt.astype(int)
   #df['fips'] = df.fips.astype(int)
    #df['regionidzip'] = df.regionidzip.astype(int) 
    #df['garagecarcnt'] = df.garagecarcnt.astype(int)
    #df['fireplacecnt'] = df.fireplacecnt.astype(int)
    #df.rename(columns = {'yearbuilt':'year', 'taxvaluedollarcnt':'home_value', 'calculatedfinishedsquarefeet':'squarefeet', 'lotsizesquarefeet':'lot_size', 'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms' }, inplace = True)
    #df['fips_encoded'] = df.fips.map({6059: 2, 6037: 1, 6111: 3,})
   
    return df




# Scale the data samples

def scale_data(train, validate, test):
    
    scale_columns = [ ]
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    rbs = RobustScaler()
    
    rbs.fit(train[scale_columns])
    
    train_scaled[scale_columns] = rbs.transform(train[scale_columns])
    validate_scaled[scale_columns] = rbs.transform(validate[scale_columns])
    test_scaled[scale_columns] = rbs.transform(test[scale_columns])
    
    return train_scaled, validate_scaled, test_scaled