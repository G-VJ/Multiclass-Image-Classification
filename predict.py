import os 
import datetime
import csv
from PIL import Image
import numpy as np
from components.Model import Model
from components.components import read_file, write_csv_file, predict_new
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
if __name__ == "__main__":
    #print()
    #get input path for file containg path of image to predict/analyse
    f_name= input("Enter file name with full path:")              
    #"get all the path from input file
    print("Reading File contating path")
    test_path = read_file(f_name) 
    if len(test_path) == 1:
    #prediction new image place at a single folder.
        print("1 path found. Predicting New files")
        pred_data = predict_new(test_path)
        data_type='New'
    else:
        print("Please provide a file contantainig single path")
    #wirte the prediction to input file
    print("writing data to file")
    write_csv_file(pred_data, data_type)