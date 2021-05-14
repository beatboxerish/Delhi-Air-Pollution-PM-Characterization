# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:28:06 2021

@author: Ishan Nangia
"""

import os

### changing directory to import modules
# print(os.getcwd())
# os.chdir("Desktop/Rijurekha Sen/interpol/Final/preprocessing/")
# print(os.getcwd())
# print(os.listdir("../../data/"))

### filling missing GPS points for all the data
# from linear_interpolation import run_linear_interpolation
# for month in ["nov", "dec", "jan"]:
#     os.mkdir("../../data/" + month + "_new_gps/")
#     files = os.listdir("../../data/" + month+ "_sep/")
#     # print(files)
#     for file in files:
#         if "gps" in file:
#             df_gps = run_linear_interpolation(file, save_df = True, return_df = True,\
#                                               data_dir = "../../data/"+ month + "_sep/",
#                                               results_dir = "../../data/"+ month + "_new_gps/")
            
#     print("Done month", month)

### renaming the linear interpolation files
# for month in ["nov", "dec", "jan"]:
    # files = os.listdir("../../data/" + month + "_new_gps/")
    # for file in files:
    #     os.rename(src = "../../data/" + month + "_new_gps/" + file,
    #               dst = "../../data/" + month + "_new_gps/" + file[10:])    
            
    # print("Done month", month)
