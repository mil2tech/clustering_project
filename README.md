# About the Project

## Project Goal

###  The goal of this project is to identify the drivers that determine the logerror of price of single family properties by constructing an ML Regression model with features from clustering.

## Project Description

### The cost of living and inflation is on the rise. The price of rent is also increasing and there is higher demand to purchase a home more than ever. Buyers want to find the home that works best for them. Buyers are using Zillow during their home buying journey to look for their potential new home. We will analayze data of homes that were purchased in the year 2017 from Zillow. The data will be helpful to determine factors that the logerror of a home. 

### Logerror is the following:

### Logerror = log(Zillow home estimate) - log(home value)

## Initial Questions

- Does the size of the lot and the square footage of the home impact the logerror?

- Does the county where the home is located impact the logerror?

- Does luxuary features like fireplace or pool impact the logerror of a home?

- Does the home being tax deliquent impact the log error?

## Data Dictionary

| Feature                        | Description                                                                                                            |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 'airconditioningtypeid'        |  Type of cooling system present in the home (if any)                                                                   |
| 'architecturalstyletypeid'     |  Architectural style of the home (i.e. ranch, colonial, split-level, etc…)                                             |
| 'basementsqft'                 |  Finished living area below or partially below ground level                                                            |
| 'bathroomcnt'                  |  Number of bathrooms in home including fractional bathrooms                                                            |
| 'bedroomcnt'                   |  Number of bedrooms in home                                                                                            |
| 'buildingqualitytypeid'        |  Overall assessment of condition of the building from best (lowest) to worst (highest)                                 |
| 'buildingclasstypeid'          | The building framing type (steel frame, wood frame, concrete/brick)                                                    |
| 'calculatedbathnbr'            |  Number of bathrooms in home including fractional bathroom                                                             |
| 'decktypeid'                   | Type of deck (if any) present on parcel                                                                                |
| 'threequarterbathnbr'          |  Number of 3/4 bathrooms in house (shower + sink + toilet)                                                             |
| 'finishedfloor1squarefeet'     |  Size of the finished living area on the first (entry) floor of the home                                               |
| 'calculatedfinishedsquarefeet' |  Calculated total finished living area of the home                                                                     |
| 'finishedsquarefeet6'          | Base unfinished and finished area                                                                                      |
| 'finishedsquarefeet12'         | Finished living area                                                                                                   |
| 'finishedsquarefeet13'         | Perimeter  living area                                                                                                 |
| 'finishedsquarefeet15'         | Total area                                                                                                             |
| 'finishedsquarefeet50'         |  Size of the finished living area on the first (entry) floor of the home                                               |
| 'fips'                         |  Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details   |
| 'fireplacecnt'                 |  Number of fireplaces in a home (if any)                                                                               |
| 'fireplaceflag'                |  Is a fireplace present in this home                                                                                   |
| 'fullbathcnt'                  |  Number of full bathrooms (sink, shower + bathtub, and toilet) present in home                                         |
| 'garagecarcnt'                 |  Total number of garages on the lot including an attached garage                                                       |
| 'garagetotalsqft'              |  Total number of square feet of all garages on lot including an attached garage                                        |
| 'hashottuborspa'               |  Does the home have a hot tub or spa                                                                                   |
| 'heatingorsystemtypeid'        |  Type of home heating system                                                                                           |
| 'latitude'                     |  Latitude of the middle of the parcel multiplied by 10e6                                                               |
| 'longitude'                    |  Longitude of the middle of the parcel multiplied by 10e6                                                              |
| 'lotsizesquarefeet'            |  Area of the lot in square feet                                                                                        |
| 'numberofstories'              |  Number of stories or levels the home has                                                                              |
| 'parcelid'                     |  Unique identifier for parcels (lots)                                                                                  |
| 'poolcnt'                      |  Number of pools on the lot (if any)                                                                                   |
| 'poolsizesum'                  |  Total square footage of all pools on property                                                                         |
| 'pooltypeid10'                 |  Spa or Hot Tub                                                                                                        |
| 'pooltypeid2'                  |  Pool with Spa/Hot Tub                                                                                                 |
| 'pooltypeid7'                  |  Pool without hot tub                                                                                                  |
| 'propertycountylandusecode'    |  County land use code i.e. it's zoning at the county level                                                             |
| 'propertylandusetypeid'        |  Type of land use the property is zoned for                                                                            |
| 'propertyzoningdesc'           |  Description of the allowed land uses (zoning) for that property                                                       |
| 'rawcensustractandblock'       |  Census tract and block ID combined - also contains blockgroup assignment by extension                                 |
| 'censustractandblock'          |  Census tract and block ID combined - also contains blockgroup assignment by extension                                 |
| 'regionidcounty'               | County in which the property is located                                                                                |
| 'regionidcity'                 |  City in which the property is located (if any)                                                                        |
| 'regionidzip'                  |  Zip code in which the property is located                                                                             |
| 'regionidneighborhood'         | Neighborhood in which the property is located                                                                          |
| 'roomcnt'                      |  Total number of rooms in the principal residence                                                                      |
| 'storytypeid'                  |  Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.).  See tab for details. |
| 'typeconstructiontypeid'       |  What type of construction material was used to construct the home                                                     |
| 'unitcnt'                      |  Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)                                    |
| 'yardbuildingsqft17'           | Patio in  yard                                                                                                         |
| 'yardbuildingsqft26'           | Storage shed/building in yard                                                                                          |
| 'yearbuilt'                    |  The Year the principal residence was built                                                                            |
| 'taxvaluedollarcnt'            | The total tax assessed value of the parcel                                                                             |
| 'structuretaxvaluedollarcnt'   | The assessed value of the built structure on the parcel                                                                |
| 'landtaxvaluedollarcnt'        | The assessed value of the land area of the parcel                                                                      |
| 'taxamount'                    | The total property tax assessed for that assessment year                                                               |
| 'assessmentyear'               | The year of the property tax assessment                                                                                |
| 'taxdelinquencyflag'           | Property taxes for this parcel are past due as of 2015                                                                 |
| 'taxdelinquencyyear'           | Year for which the unpaid propert taxes were due                                                                       |



## Steps to Reproduce

### The Plan
- Acquire the data
- Wrangle (prepare and clean) the data
- Split the data into three samples: `train`, `validate`, and `test`
- Conduct exploration of `train` sample and find relationships in data by plotting visuals and statistical testing
- Scale the `train` sample to normalize the values prior to modeling. Will transform `validate` and `test` samples also
- Explore the data
- Generate clusters from scaled `train` sample
- Generate model predictions by fitting models to only the scaled `train` and `validate` samples to avoid data leakage into `test` sample
- Evaluate the performance of the models with root mean square error (RMSE) calculation and pick the best performing model
- Evaluate the performance of best model.
- Draw conclusion


### Acquistion of zillow family home data

To acquire the zillow home data, I used the zillow_db in our mySQL server using the query below.

'''

    SELECT prop. *,
    predictions_2017.logerror,
    predictions_2017.transactiondate,
    air.airconditioningdesc,
    arch.architecturalstyledesc,
    build.buildingclassdesc,
    heat.heatingorsystemdesc,
    land.propertylandusedesc,
    story.storydesc,
    type.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
                SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
    AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
    LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
    LEFT JOIN storytype story USING(storytypeid)
    LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
    WHERE propertylandusedesc IN ('Single Family Residential' , 'Mobile Home', 'Manufactured, Modular, Prefabricated Homes', 'Patio Home', 'Bungalow', 'Planned Unit Development') 
    AND transactiondate <= '2017-12-31'
    AND prop.longitude IS NOT NULL
    AND prop.latitude IS NOT NULL;

'''

- Should acquire raw dataset that has 54395 rows and 67 columns
- The query will bring null values

### Preparation of data

To clean the data I did the following things in order

'''
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
   
    return df
'''
Removed outliers in bed, bath, zip, square feet, acres, garagecarcnt& tax rate
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
'''


### Split the data

To split the data

- Split the refined dataframe into 3 samples using the train_test_split function from sklearn.model_selection using a random seed of "123"

### Scale the `train` sample and transform other samples using MinMax Scaler

-  Create function with columns to scale with code scale_columns = [ 'squarefeet', 'lot_size', 'bedrooms', 'bathrooms','garagecarcnt', 'fips_encoded', 'home_age' ]
-   made copy of each sample train_scaled = train.copy() , validate_scaled = validate.copy() , test_scaled = test.copy()
-  imported  sklearn.preprocessing.MinMaxScaler() as rbs
-  fit scaler object to `train` with code rbs.fit(train[scale_columns])
- transformed other samples with code :
    -  train_scaled[scale_columns] = rbs.transform(train[scale_columns])
    - validate_scaled[scale_columns] = rbs.transform(validate[scale_columns])
    - test_scaled[scale_columns] = rbs.transform(test[scale_columns])


### Make X_train and y_train

- Create x_train by using scaled train sample using code:
     - 

- Create y-train by using target variable only
     

### Modeling the data and evaluating which model performs the best on scaled `train` and `validate` samples
- from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor 
- Create prediction DataFrames with the actual home value of `train`  and  home value baseline prediction by using the mean of the `home_value` in `train` sample
- Create prediction DataFrames with the actual home value of `validate` and  home value baseline prediction by using the mean of the `home_value` in `train` sample
- Create model object and fit model to `x_train` and ` y_train`
- Use the model object to make prediction on both scaled `train` and `validate` samples by omitting columns 'home_value', 'fips', 'year'.
- Add model predictions to both prediction dataframes created earlier
- Evaluate the performce of the models by using root mean square error (RMSE) on the predictions using similar code:
    - from scipy import stats
    - from math import sqrt
    - from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
    - train_s_rmse = sqrt(mean_squared_error(home_value_mean['home_value'], home_value_mean.baseline))

    - def calculate_rmse(t):
        return sqrt(mean_squared_error(home_value_mean['home_value'], t))

    - print('Train baseline RMSE: {}.'.format(train_s_rmse))
    - home_value_mean.apply(calculate_rmse).sort_values()
- Review the results and pick the model with the lowest RMSE to run on scaled `test` sample

### Running best model on scaled `test` sample

- Create prediction DataFrame with the actual home value of `train`  and  home value baseline prediction by using the mean of the `home_value` in `train` sample
- Use the model object to make prediction on  scaled `test`  samples by omitting columns 'home_value', 'fips', 'year'.
- Add model predictions to both prediction dataframes created earlier
- Evaluate the performce of the models by using root mean square error (RMSE) on the predictions using similar code:
    - from scipy import stats
    - from math import sqrt
    - from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
    - train_s_rmse = sqrt(mean_squared_error(home_value_mean['home_value'], home_value_mean.baseline))

    - def calculate_rmse(t):
        return sqrt(mean_squared_error(home_value_mean['home_value'], t))

    - print('Train baseline RMSE: {}.'.format(train_s_rmse))
    - home_value_mean.apply(calculate_rmse).sort_values()
- Review the results of model performance on scaled `test` sample

### Conclusion

- Summary of the project
    - 

- Recommendation
    - 
    - 

- Next Steps
    - 
    - 

