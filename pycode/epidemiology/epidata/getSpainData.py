############################################################################################################
#                                                                                                          #
#        IMPORTANT NOTE: WHEN USING THIS DATA, WE HAVE TO CITE https://github.com/datadista/datasets       #
#                                                                                                          #
#                                                                                                          #
#        DO NOT USE DATA FROM THE FOLLOWING REGIONS SINCE THE COLUMNS HOSPITALIZED AND ICU                 #
#        ARE NOT CORRECTLY SUMMED TO TOTAL NUMBERS ! THE SAME APPLIES TO ALL AGE DATA AT THE MOMENT !      #
#                                                                                                          #    
#               HOSPITALIZED                                   ICU                                         #
#               Castilla La Mancha (until 2020-04-11)          Castilla La Mancha (hasta 2020-04-12)       #
#               Comunidad Valenciana (hasta 2020-04-08)        Castilla y León (hasta 2020-04-17)          #
#               Madrid (hasta 2020-04-26)                      Comunidad Valenciana (hasta 2020-04-08)     #
#               Castilla y León (hasta 2020-04-06)             Galicia (hasta 2020-04-29)                  #
#               Madrid (hasta 2020-04-26)                                                                  #           
#                                                                                                          #
############################################################################################################


import os
import pandas
import numpy as np

from epidemiology.epidata import outputDict as od
from epidemiology.epidata import getDataIntoPandasDataFrame as gd
from epidemiology.epidata import defaultDict as dd


def get_spain_data(read_data=dd.defaultDict['read_data'],
                   make_plot=dd.defaultDict['make_plot'],
                   out_form=dd.defaultDict['out_form'],
                   out_folder=dd.defaultDict['out_folder']):

   directory = os.path.join(out_folder, 'Spain/')

   if not os.path.exists(directory):
      os.makedirs(directory)

   AgesJSONData = os.path.join(directory ,'raw_spain_all_age.json')
   StatJSONData = os.path.join(directory ,'raw_spain_all_state.json')

   if(read_data):
      # if once dowloaded just read json file

      #### ages' data
      df_age = pandas.read_json(AgesJSONData)

      print("Read from local. Available columns:", df_age.columns)

      #### states' data
      df_state = pandas.read_json(StatJSONData)

      print("Read from local. Available columns:", df_state.columns)

   else:
  
      # Get data:
      # https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/nacional_covid19_rango_edad.csv
      df_age = gd.loadCsv('nacional_covid19_rango_edad',
                          apiUrl='https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/')
      
      if df_age.empty != True:
         # standardization of column titles from Spanish to English
         # the stupid character in front of 'fecha' is correct here. There is a bug in the original file.
         df_age.rename({'fecha': 'Date', 'rango_edad': 'Age10', 'sexo': 'Gender', 'casos_confirmados': 'Confirmed',
               'hospitalizados': 'Hospitalized', 'ingresos_uci': 'ICU', 'fallecidos' : 'Deaths'}, axis=1, inplace=True) 

         print("Read Spanish age data from online. Available columns:", df_age.columns)

         # translate column gender from Spanish to English and standardize
         df_age.loc[df_age.Gender=='ambos', ['Gender']] = 'both'
         df_age.loc[df_age.Gender=='mujeres', ['Gender']] = 'female'
         df_age.loc[df_age.Gender=='hombres', ['Gender']] = 'male'
         df_age.loc[df_age.Age10=='80 y +', ['Age10']] = '80+'
         df_age.loc[df_age.Age10=='90 y +', ['Age10']] = '90+'
         df_age.loc[df_age.Age10=='Total', ['Age10']] = 'all'

         # Correct Timestamps:
         df_age['Date'] = df_age['Date'].astype( 'datetime64[ns]' ).dt.tz_localize('Europe/Berlin')

         # output data to not always download it
         df_age.to_json(AgesJSONData)

      # Get data:
      # https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_datos_isciii.csv
      df_state = gd.loadCsv('ccaa_covid19_datos_isciii',
                            apiUrl='https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/')
      
      if df_state.empty != True:
         # standardization of column titles from Spanish to English
         df_state.rename({'Fecha': 'Date', 'cod_ine': 'ID_State', 'CCAA': 'State', 'Casos': 'Confirmed_total', 'PCR+': 'Confirmed_PCR', 
               'TestAc+': 'Confirmed_AB', 'Hospitalizados': 'Hospitalized', 'UCI': 'ICU', 'Fallecidos' : 'Deaths', 'Recuperados' : 'Recovered'}, axis=1, inplace=True)

         print("Read Spanish states data from online. Available columns:", df_state.columns)

         # fill empty cells (nan values) with zero
         df_state.replace(np.nan, 0, inplace=True)

         # remove special characters
         df_state.loc[df_state.State=="Andalucía", ['State']] = "Andalucia"
         df_state.loc[df_state.State=="Castilla y León", ['State']] = "Castilla y Leon"
         df_state.loc[df_state.State=="Cataluña", ['State']] = "Cataluna"
         df_state.loc[df_state.State=="País Vasco", ['State']] = "Pais Vasco"
         df_state.loc[df_state.State=="Aragón", ['State']] = "Aragon"

         # Correct Timestamps:
         df_state['Date'] = df_state['Date'].astype( 'datetime64[ns]' ).dt.tz_localize('Europe/Berlin')

         # output data to not always download it
         df_state.to_json(StatJSONData)
         

   # Preparation for plotting/output age specific data:

   outForm = od.outForm[out_form][0]
   outFormEnd = od.outForm[out_form][1]
   outFormSpec = od.outForm[out_form][2]

   # only consider men AND women (through information on gender away)
   df_age = df_age.loc[df_age.Gender=='both']

   # write file for all age groups summed together
   df_agesum = df_age.loc[df_age.Age10=='all']
      # call to df_ageall.to_json("all_age.json", orient='records')
   getattr(df_agesum, outForm)(os.path.join(directory, "spain") + outFormEnd, **outFormSpec)

   # write file with information on all age groups separately
   # age_groups = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+']
   df_agesep = df_age.loc[df_age.Age10!='all']
      # call to df_ageall.to_json("all_age.json", orient='records')
   getattr(df_agesep, outForm)(os.path.join(directory ,"spain_all_age") + outFormEnd, **outFormSpec)

   # Preparation for plotting/output:

   PCR_ONLY = False # if pcr only is used 
   # dff = df_state['state'].unique()


   # if PCR only is used, the number of confirmed cases is the number of people being tested positive with PCR test
   # otherwise, the number of positive antibody tests is also taken into account
   if PCR_ONLY:
      df_state.loc[df_state.Confirmed_total==0, ['Confirmed_total']] = df_state.Confirmed_PCR
   else:
      df_state.loc[df_state.Confirmed_total==0, ['Confirmed_total']] = df_state.Confirmed_PCR + df_state.Confirmed_AB

   states_array = df_state.State.unique()

   # output json
   # call df_state.to_json("spain_all_state.json", orient='records'), or to hdf5 alternatively
   getattr(df_state, outForm)(os.path.join(directory ,"spain_all_state") + outFormEnd, **outFormSpec)


def main():

    [read_data, make_plot, out_form, out_folder] = gd.cli('Download of spain data')
    get_spain_data(read_data, make_plot, out_form, out_folder)


if __name__ == "__main__":

    main()
