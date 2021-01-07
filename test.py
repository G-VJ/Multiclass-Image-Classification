import os 
import datetime
import csv
from PIL import Image
import numpy as np
from components.Model import Model
from components.components import read_file, write_csv_file, predict_tagged
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

if __name__ == "__main__":
    #print()
    #get input path for file containg path of image to predict/analyse
    f_name= input("Enter file name with full path:")              
    #"get all the path from input file
    print("Reading File contating path")
    test_path = read_file(f_name)
    if len(test_path) == 6:
    #prdiction already tagged files
        print("6 different path found. Predicting labeled files")
        pred_data = predict_tagged(test_path)
        data_type='labeled'
    else:
    #prediction new image place at a single folder.
        print( len(test_path), "path found expecte 6 different path")

    #wirte the prediction to input file
    print("writing data to file")
    #print(pred_data)
    write_csv_file(pred_data, data_type)