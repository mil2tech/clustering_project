import pandas as pd
import numpy as np
import os

import env

from env import host, user, password

def get_db_url(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


    
def get_zillow_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = '''
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
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    df = df.drop(columns='id')
    return df

