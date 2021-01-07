import os 
from components.Model import Model
from components.components import read_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

if __name__ == "__main__":
    #print()
    #get input path for file containg path of image to be used for training
    f_name= input("Enter file name with full path:")              
    #"get all the path from input file
    print("Reading File contating path")
    path_list = read_file(f_name)
    
    if len(path_list) == 1:
    #Trai model using already tagged files
        model = Model()
        path = path_list[0]
        model.model_train(path)
    else:
        print(len(path_list)," path found. While expecting 1 path only. Exiting...")
        exit()