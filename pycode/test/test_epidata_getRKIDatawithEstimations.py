import unittest
from pyfakefs import fake_filesystem_unittest
import numpy as np
from datetime import date, timedelta

import os
import pandas as pd

from epidemiology.epidata import getRKIDatawithEstimations as grdwd
from epidemiology.epidata import getDataIntoPandasDataFrame as gd
from epidemiology.epidata import defaultDict as dd
from unittest.mock import patch, call

class TestGetRKIDatawithEstimations(fake_filesystem_unittest.TestCase):

    path = '/home/RKIEstimationData'

    # Notice data is not realistic
    str_whole_country_Germany_jh = \
("""[{"CountryRegion":"Germany","Date":"2020-01-22","Confirmed":0,"Recovered":0.0,"Deaths":0.0},\
{"CountryRegion":"Germany","Date":"2020-01-23","Confirmed":0,"Recovered":0.0,"Deaths":0.0},\
{"CountryRegion":"Germany","Date":"2020-01-24","Confirmed":0,"Recovered":0.0,"Deaths":0.0},\
{"CountryRegion":"Germany","Date":"2020-01-25","Confirmed":0,"Recovered":0.0,"Deaths":0.0},\
{"CountryRegion":"Germany","Date":"2020-01-26","Confirmed":0,"Recovered":1.0,"Deaths":1.0},\
{"CountryRegion":"Germany","Date":"2020-01-27","Confirmed":5,"Recovered":2.0,"Deaths":1.0},\
{"CountryRegion":"Germany","Date":"2020-01-28","Confirmed":6,"Recovered":3.0,"Deaths":1.0},\
{"CountryRegion":"Germany","Date":"2020-01-29","Confirmed":7,"Recovered":3.0,"Deaths":1.0},\
{"CountryRegion":"Germany","Date":"2020-01-30","Confirmed":8,"Recovered":3.0,"Deaths":1.0},\
{"CountryRegion":"Germany","Date":"2020-01-31","Confirmed":9,"Recovered":3.0,"Deaths":2.0}]""")

    str_all_germany_rki = ("""[{"Date":1577836800000,"Confirmed":21,"Deaths":0,"Recovered":21},
{"Date":1577923200000,"Confirmed":23,"Deaths":0,"Recovered":23},
{"Date":1578009600000,"Confirmed":28,"Deaths":0,"Recovered":28},
{"Date":1578096000000,"Confirmed":32,"Deaths":0,"Recovered":32},
{"Date":1578268800000,"Confirmed":33,"Deaths":0,"Recovered":33},
{"Date":1578441600000,"Confirmed":34,"Deaths":0,"Recovered":34},
{"Date":1578528000000,"Confirmed":35,"Deaths":0,"Recovered":35},
{"Date":1578614400000,"Confirmed":37,"Deaths":0,"Recovered":37},
{"Date":1578700800000,"Confirmed":40,"Deaths":0,"Recovered":40},
{"Date":1578873600000,"Confirmed":41,"Deaths":0,"Recovered":41},
{"Date":1578960000000,"Confirmed":42,"Deaths":0,"Recovered":42},
{"Date":1579046400000,"Confirmed":43,"Deaths":0,"Recovered":43},
{"Date":1579132800000,"Confirmed":44,"Deaths":0,"Recovered":44},
{"Date":1579219200000,"Confirmed":47,"Deaths":1,"Recovered":46},
{"Date":1579305600000,"Confirmed":50,"Deaths":1,"Recovered":49},
{"Date":1579392000000,"Confirmed":51,"Deaths":1,"Recovered":50},
{"Date":1579478400000,"Confirmed":52,"Deaths":1,"Recovered":51},
{"Date":1579564800000,"Confirmed":53,"Deaths":1,"Recovered":52},
{"Date":1579651200000,"Confirmed":57,"Deaths":1,"Recovered":56},
{"Date":1579737600000,"Confirmed":59,"Deaths":1,"Recovered":58},
{"Date":1579824000000,"Confirmed":60,"Deaths":1,"Recovered":59},
{"Date":1579910400000,"Confirmed":62,"Deaths":1,"Recovered":61},
{"Date":1579996800000,"Confirmed":63,"Deaths":1,"Recovered":62},
{"Date":1580083200000,"Confirmed":68,"Deaths":1,"Recovered":67},
{"Date":1580169600000,"Confirmed":70,"Deaths":1,"Recovered":69},
{"Date":1580256000000,"Confirmed":73,"Deaths":1,"Recovered":72},
{"Date":1580342400000,"Confirmed":84,"Deaths":1,"Recovered":83},
{"Date":1580428800000,"Confirmed":96,"Deaths":1,"Recovered":95}]""")

    str_all_gender_rki = (""" [{"Gender": "female", "Date": 1577836800000, "Confirmed": 7, "Deaths": 0, "Recovered": 7},
{"Gender": "female", "Date": 1577923200000, "Confirmed": 9, "Deaths": 0, "Recovered": 9},
{"Gender": "female", "Date": 1578009600000, "Confirmed": 13, "Deaths": 0, "Recovered": 13},
{"Gender": "female", "Date": 1578096000000, "Confirmed": 16, "Deaths": 0, "Recovered": 16},
{"Gender": "female", "Date": 1578441600000, "Confirmed": 17, "Deaths": 0, "Recovered": 17},
{"Gender": "female", "Date": 1578614400000, "Confirmed": 18, "Deaths": 0, "Recovered": 18},
{"Gender": "female", "Date": 1578700800000, "Confirmed": 20, "Deaths": 0, "Recovered": 20},
{"Gender": "female", "Date": 1578873600000, "Confirmed": 21, "Deaths": 0, "Recovered": 21},
{"Gender": "female", "Date": 1579046400000, "Confirmed": 22, "Deaths": 0, "Recovered": 22},
{"Gender": "female", "Date": 1579132800000, "Confirmed": 23, "Deaths": 0, "Recovered": 23},
{"Gender": "female", "Date": 1579219200000, "Confirmed": 25, "Deaths": 1, "Recovered": 24},
{"Gender": "female", "Date": 1579392000000, "Confirmed": 26, "Deaths": 1, "Recovered": 25},
{"Gender": "female", "Date": 1579737600000, "Confirmed": 27, "Deaths": 1, "Recovered": 26},
{"Gender": "female", "Date": 1580083200000, "Confirmed": 30, "Deaths": 1, "Recovered": 29},
{"Gender": "female", "Date": 1580256000000, "Confirmed": 33, "Deaths": 1, "Recovered": 32},
{"Gender": "female", "Date": 1580342400000, "Confirmed": 38, "Deaths": 1, "Recovered": 37},
{"Gender": "female", "Date": 1580428800000, "Confirmed": 43, "Deaths": 1, "Recovered": 42},
{"Gender":"male","Date":1577836800000,"Confirmed":14,"Deaths":0,"Recovered":14},
{"Gender":"male","Date":1578009600000,"Confirmed":15,"Deaths":0,"Recovered":15},
{"Gender":"male","Date":1578096000000,"Confirmed":16,"Deaths":0,"Recovered":16},
{"Gender":"male","Date":1578268800000,"Confirmed":17,"Deaths":0,"Recovered":17},
{"Gender":"male","Date":1578528000000,"Confirmed":18,"Deaths":0,"Recovered":18},
{"Gender":"male","Date":1578614400000,"Confirmed":19,"Deaths":0,"Recovered":19},
{"Gender":"male","Date":1578700800000,"Confirmed":20,"Deaths":0,"Recovered":20},
{"Gender":"male","Date":1578960000000,"Confirmed":21,"Deaths":0,"Recovered":21},
{"Gender":"male","Date":1579219200000,"Confirmed":22,"Deaths":0,"Recovered":22},
{"Gender":"male","Date":1579305600000,"Confirmed":25,"Deaths":0,"Recovered":25},
{"Gender":"male","Date":1579478400000,"Confirmed":26,"Deaths":0,"Recovered":26},
{"Gender":"male","Date":1579564800000,"Confirmed":27,"Deaths":0,"Recovered":27},
{"Gender":"male","Date":1579651200000,"Confirmed":31,"Deaths":0,"Recovered":31},
{"Gender":"male","Date":1579737600000,"Confirmed":32,"Deaths":0,"Recovered":32},
{"Gender":"male","Date":1579824000000,"Confirmed":33,"Deaths":0,"Recovered":33},
{"Gender":"male","Date":1579910400000,"Confirmed":35,"Deaths":0,"Recovered":35},
{"Gender":"male","Date":1579996800000,"Confirmed":36,"Deaths":0,"Recovered":36},
{"Gender":"male","Date":1580083200000,"Confirmed":38,"Deaths":0,"Recovered":38},
{"Gender":"male","Date":1580169600000,"Confirmed":40,"Deaths":0,"Recovered":40},
{"Gender":"male","Date":1580342400000,"Confirmed":46,"Deaths":0,"Recovered":46},
{"Gender":"male","Date":1580428800000,"Confirmed":53,"Deaths":0,"Recovered":53}]""")

    rki_files_to_change = ["all_germany_rki", "all_gender_rki", "all_age_rki",
                           "all_state_rki", "all_state_gender_rki", "all_state_age_rki",
                           "all_county_rki", "all_county_gender_rki", "all_county_age_rki"]

    def setUp(self):
        self.setUpPyfakefs()


    def write_rki_data(self, out_folder):

        for file_to_change in self.rki_files_to_change:

            file_rki = file_to_change + ".json"
            file_rki_with_path = os.path.join(out_folder, file_rki)

            if file_to_change == "all_gender_rki":
                with open(file_rki_with_path, 'w') as f:
                    f.write(self.str_all_gender_rki)
            else:
                with open(file_rki_with_path, 'w') as f:
                    f.write(self.str_all_germany_rki )

    def write_jh_data(self, out_folder):
        file_jh = "whole_country_Germany_jh.json"
        file_jh_with_path = os.path.join(out_folder, file_jh)

        with open(file_jh_with_path, 'w') as f:
            f.write(self.str_whole_country_Germany_jh)

    def test_get_rki_data_with_estimations(self):

        [read_data, make_plot, out_form, out_folder] \
            = [True, False, "json", self.path, ]

        # write files which should be read in by program

        directory = os.path.join(out_folder, 'Germany/')
        gd.check_dir(directory)

        self.write_rki_data(directory)
        self.write_jh_data(directory)

        # check if expected files are written
        self.assertEqual(len(os.listdir(self.path)), 1)
        self.assertEqual(len(os.listdir(directory)), 1+len(self.rki_files_to_change))

        grdwd.get_rki_data_with_estimations(read_data, out_form, out_folder, make_plot)

        # check if expected files are written
        self.assertEqual(len(os.listdir(self.path)), 1)
        self.assertEqual(len(os.listdir(directory)), 1+ 2*len(self.rki_files_to_change))

        f_read = os.path.join(directory, "all_germany_rki_estimated.json")
        df = pd.read_json(f_read)

        confirmed = dd.EngEng['confirmed']
        recovered = dd.EngEng['recovered']
        deaths = dd.EngEng['deaths']
        date = dd.EngEng['date']
        recovered_estimated = recovered + "_estimated"
        deaths_estimated = deaths + "_estimated"

        data_list = df.columns.values.tolist()

        self.assertEqual(data_list, [date, confirmed, deaths, recovered, recovered_estimated, deaths_estimated])

        self.assertEqual(df[(df[date] == "2020-01-30")][recovered_estimated].item(), np.round(84 * 3. / 8.))
        self.assertEqual(df[(df[date] == "2020-01-31")][recovered_estimated].item(), np.round(96 * 3. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-30")][deaths_estimated].item(), np.round(84 * 1. / 8.))
        self.assertEqual(df[(df[date] == "2020-01-31")][deaths_estimated].item(), np.round(96 * 2. / 9.))

        # gender specific data
        gender = dd.EngEng['gender']

        f_read = os.path.join(directory, "all_gender_rki_estimated.json")
        df = pd.read_json(f_read)

        data_list = df.columns.values.tolist()

        self.assertEqual(data_list, [gender, date, confirmed, deaths, recovered, recovered_estimated, deaths_estimated])

        self.assertEqual(df[(df[date] == "2020-01-28") & (df[gender] == "male")][recovered_estimated].item(),
                         np.round(40 * 3. / 6.))
        self.assertEqual(df[(df[date] == "2020-01-29") & (df[gender] == "female")][recovered_estimated].item(),
                         np.round(33 * 3. / 7.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "male")][recovered_estimated].item(),
                         np.round(53 * 3. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "female")][recovered_estimated].item(),
                         np.round(43 * 3. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-28") & (df[gender] == "male")][deaths_estimated].item(),
                         np.round(40 * 1. / 6.))
        self.assertEqual(df[(df[date] == "2020-01-29") & (df[gender] == "female")][deaths_estimated].item(),
                         np.round(33 * 1. / 7.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "male")][deaths_estimated].item(),
                         np.round(53 * 2. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "female")][deaths_estimated].item(),
                         np.round(43 * 2. / 9.))

    @patch('epidemiology.epidata.getRKIDatawithEstimations.grd.get_rki_data')
    @patch('epidemiology.epidata.getRKIDatawithEstimations.gjd.get_jh_data')
    def test_get_rki_data_with_estimations_download(self,mock_get_jh_data, mock_get_rki_data):

        [read_data, make_plot, out_form, out_folder] \
            = [False, False, "json", self.path, ]

        directory = os.path.join(out_folder, 'Germany/')
        gd.check_dir(directory)


        mock_get_rki_data.side_effect =self.write_rki_data(directory)
        mock_get_jh_data.side_effect = self.write_jh_data(directory)

        # write files which should be read in by program

        directory = os.path.join(out_folder, 'Germany/')
        gd.check_dir(directory)

        rki_files_to_change = ["all_germany_rki", "all_gender_rki", "all_age_rki",
                               "all_state_rki", "all_state_gender_rki", "all_state_age_rki",
                               "all_county_rki", "all_county_gender_rki", "all_county_age_rki"]

        for file_to_change in rki_files_to_change:

            file_rki = file_to_change + ".json"
            file_rki_with_path = os.path.join(directory, file_rki)

            if file_to_change == "all_gender_rki":
                with open(file_rki_with_path, 'w') as f:
                    f.write(self.str_all_gender_rki)
            else:
                with open(file_rki_with_path, 'w') as f:
                    f.write(self.str_all_germany_rki)

        file_jh = "whole_country_Germany_jh.json"
        file_jh_with_path = os.path.join(directory, file_jh)

        with open(file_jh_with_path, 'w') as f:
            f.write(self.str_whole_country_Germany_jh)

        # check if expected files are written
        self.assertEqual(len(os.listdir(self.path)), 1)
        self.assertEqual(len(os.listdir(directory)), 1+len(rki_files_to_change))

        grdwd.get_rki_data_with_estimations(read_data, out_form, out_folder, make_plot)

        # check if expected files are written
        self.assertEqual(len(os.listdir(self.path)), 1)
        self.assertEqual(len(os.listdir(directory)), 1+ 2*len(rki_files_to_change))

        confirmed = dd.EngEng['confirmed']
        recovered = dd.EngEng['recovered']
        deaths = dd.EngEng['deaths']
        date = dd.EngEng['date']
        recovered_estimated = recovered + "_estimated"
        deaths_estimated = deaths + "_estimated"

        f_read = os.path.join(directory, "all_germany_rki_estimated.json")
        df = pd.read_json(f_read)

        data_list = df.columns.values.tolist()

        self.assertEqual(data_list, [date, confirmed, deaths, recovered, recovered_estimated, deaths_estimated])

        self.assertEqual(df[(df[date] == "2020-01-30")][recovered_estimated].item(), np.round(84 * 3. / 8.))
        self.assertEqual(df[(df[date] == "2020-01-31")][recovered_estimated].item(), np.round(96 * 3. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-30")][deaths_estimated].item(), np.round(84 * 1. / 8.))
        self.assertEqual(df[(df[date] == "2020-01-31")][deaths_estimated].item(), np.round(96 * 2. / 9.))

        # gender specific data
        gender = dd.EngEng['gender']

        f_read = os.path.join(directory, "all_gender_rki_estimated.json")
        df = pd.read_json(f_read)

        data_list = df.columns.values.tolist()

        self.assertEqual(data_list, [gender, date, confirmed, deaths, recovered, recovered_estimated, deaths_estimated])

        self.assertEqual(df[(df[date] == "2020-01-28") & (df[gender] == "male")][recovered_estimated].item(),
                         np.round(40 * 3. / 6.))
        self.assertEqual(df[(df[date] == "2020-01-29") & (df[gender] == "female")][recovered_estimated].item(),
                         np.round(33 * 3. / 7.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "male")][recovered_estimated].item(),
                         np.round(53 * 3. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "female")][recovered_estimated].item(),
                         np.round(43 * 3. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-28") & (df[gender] == "male")][deaths_estimated].item(),
                         np.round(40 * 1. / 6.))
        self.assertEqual(df[(df[date] == "2020-01-29") & (df[gender] == "female")][deaths_estimated].item(),
                         np.round(33 * 1. / 7.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "male")][deaths_estimated].item(),
                         np.round(53 * 2. / 9.))
        self.assertEqual(df[(df[date] == "2020-01-31") & (df[gender] == "female")][deaths_estimated].item(),
                         np.round(43 * 2. / 9.))


if __name__ == '__main__':
    unittest.main()