# -*- coding: utf-8 -*-
# from preprocessing.preprocess import 
from preprocessing.linear_interpolation import run_linear_interpolation

df_gps = run_linear_interpolation("2020-11-1_gps.csv", save = True, return_df= True)


