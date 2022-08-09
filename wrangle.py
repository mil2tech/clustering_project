from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    # Rename columns
    df.rename(columns = {'fireplaceflag':'hasfireplace', 'pooltypeid7':'pool_wo_spa_hottub', 'pooltypeid10':'has_spa_hottub', 'pooltypeid2':'pool_w_spa_hottub' }, inplace = True)
    # fill in columns related to garage
    df["garagecarcnt"].fillna( 0 , inplace = True)
    df["garagetotalsqft"].fillna( 0 , inplace = True)
    # code fixing columns
    df['taxdelinquencyflag'].fillna( 'N' , inplace = True)
    df['years_taxdeliquent'] = 16 - df.taxdelinquencyyear
    df['years_taxdeliquent'] = df['years_taxdeliquent'].replace([-83.00], 17)
    df['years_taxdeliquent'].fillna( 0 , inplace = True)
    # fix latitude and longitude
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000
    # Code to fill nulls in
    df["basementsqft"].fillna( 0 , inplace = True)
    # create age column
    df['home_age'] = 2017 - df.yearbuilt
    df['home_age_bin'] = pd.cut(df.home_age, bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, .60, .666, .733, .8, .866, .933])
    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100
    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    # bin calcualatedfinishedsquarefeet
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000], labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'home_age_bin': 'float64', 'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})
    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt
    # Map values in columns
    df['fips_encoded'] = df.fips.map({6059: 2, 6037: 1, 6111: 3})
    df['taxdelinquencyflag'] = df.taxdelinquencyflag.map({"Y": 1, "N": 0})
    # Rename columns
    df.rename(columns = { 'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms' , 'yearbuilt':'year'}, inplace = True)

    def get_counties():
        '''
        This function will create dummy variables out of the original fips column. 
        And return a dataframe with all of the original columns except regionidcounty.
        We will keep fips column for data validation after making changes. 
        New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
        The fips ids are renamed to be the name of the county each represents. 
        '''
        # create dummy vars of fips id
        county_df = pd.get_dummies(df.fips)
        # rename columns by actual county name
        county_df.columns = ['LA', 'Orange', 'Ventura']
        # concatenate the dataframe with the 3 county columns to the original dataframe
        df_dummies = pd.concat([df, county_df], axis = 1)
        # drop regionidcounty and fips columns
        #df_dummies = df_dummies.drop(columns = ['regionidcounty'])
        return df_dummies
    
    df = get_counties()    
    #Define function to drop columns/rows based on proportion of nulls
    def null_dropper(df, prop_required_column, prop_required_row):
    
        prop_null_column = 1 - prop_required_column
    
        for col in list(df.columns):
        
            null_sum = df[col].isna().sum()
            null_pct = null_sum / df.shape[0]
        
            if null_pct > prop_null_column:
                df.drop(columns=col, inplace=True)
            
        row_threshold = int(prop_required_row * df.shape[1])
    
        df.dropna(axis=0, thresh=row_threshold, inplace=True)

        return df

    #Execute my function 
    df = null_dropper(df, 0.75, 0.75)

    # Drop remaining rows with null values
    df = df.dropna()

    # Drop Necessary columns
    df = df.drop(columns=['taxamount','assessmentyear','propertylandusedesc','garagetotalsqft', 'hashottuborspa', 'calculatedbathnbr', 'finishedsquarefeet12','fullbathcnt', 'propertycountylandusecode', 'propertylandusetypeid', 'regionidcounty', 'regionidcity', 'roomcnt', 'rawcensustractandblock','censustractandblock'])



    # drop observations with nulls in column
    #df = df[pd.notnull(df['yearbuilt'])]
    # Dropped columns
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

def remove_outliers():
    import wrangle
    df = wrangle.prep_familyhome17()
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathrooms <= 7) & (df.bedrooms <= 7) & 
               (df.regionidzip < 100000) & 
               (df.bathrooms > 0) & 
               (df.bedrooms > 0) & 
               (df.acres < 20) &
               (df.calculatedfinishedsquarefeet < 10000) & 
               (df.taxrate < 10) &
               (df.garagecarcnt <= 5)
              )]

def subsets():
    import wrangle
    df = wrangle.remove_outliers()
    df_la, df_ventura, df_orange = df[df.LA == 1].drop(columns = ['parcelid',  'taxvaluedollarcnt', 'fips', 'fips_encoded',
                                        'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'year', 
                                        'lotsizesquarefeet', 'regionidzip', 'transactiondate',
                                        'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                                        'LA', 'Ventura', 'Orange']) ,df[df.Ventura == 1].drop(columns = ['parcelid',  'taxvaluedollarcnt', 'fips', 'fips_encoded',
                                        'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'year', 
                                        'lotsizesquarefeet', 'regionidzip', 'transactiondate',
                                        'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                                        'LA', 'Ventura', 'Orange']) , df[df.Orange == 1].drop(columns = ['parcelid',  'taxvaluedollarcnt', 'fips', 'fips_encoded',
                                        'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'year', 
                                        'lotsizesquarefeet', 'regionidzip', 'transactiondate',
                                        'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                                        'LA', 'Ventura', 'Orange'])
    return df_la, df_ventura, df_orange



def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train , validate , & test 
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions


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