import os 
import datetime
import csv
from PIL import Image
import numpy as np
from components.Model import Model

lables = ['Buildings', 'Sea', 'Glacier', 'Mountain', 'Forest', 'Street' ]
log_file_name = './log/test_log.txt' 

def read_file(file_name):
    test_file  = open(file_name, "r") 
    ln = test_file.readlines()
    test_file.close()
    path=[]
    for x in ln:
        path.append(x.replace("\n", "").strip())
    return path

def write_csv_file(pred_data, data_type):
    time = datetime.datetime.now().strftime('_%d%m%Y_%H%M%S')
    file_name = 'Prediction_'+ data_type+time + ".csv"
    file_name = "./test_scripts/"+file_name
    with open(file_name, 'w', newline='') as prd_file:
        writer = csv.writer(prd_file)
        if data_type == 'New':
            writer.writerow(["File", "Predicted Class", 'Buildings', 'Sea', 'Glacier', 'Mountain', 'Forest', 'Street'])
        else:
            writer.writerow(['File', 'Original Class', 'Predicted Class', 'Buildings', 'Sea', 'Glacier', 'Mountain', 'Forest', 'Street'])
        writer.writerows(pred_data)
    file_name= file_name.strip()
    print("Result exported to:",file_name)
    

def predict_tagged(test_path):
    pred_data = []
    model = Model()
    model.load_model()
    dr_count = 0
    test_correct = 0
    total = 0    
    for p in test_path:
        for dr in os.listdir(p):
            extension = dr.split(".")[-1] in ("jpg", "jpeg", "png")
            if extension:            
                file = p+"/"+dr
                print("Predicting file ", file)
                img = Image.open(file)
                pred_result = model.predict_img(img)
                pred_value = np.argmax(pred_result, axis = 1)
                pred_index = pred_value[0]
                pred_class = lables[pred_index]
                """
                ln = []
                ln.append(file)
                ln.append(p.split('/')[-1])
                ln.append(pred_class)
                ln.append(round(pred_result[0][pred_index]*100, 2))
                """
                ln = []
                ln.append(file)
                ln.append(p.split('/')[-1])
                ln.append(pred_class)
                for i in range(0,len(pred_result[0])):
                    confidence = f"{pred_result[0][i]*100:0.2f} %"
                    ln.append(confidence)                
                pred_data.append(ln)
                if pred_value == dr_count:
                    test_correct = test_correct + 1
                total = total + 1
        dr_count = dr_count + 1
    score = round((test_correct/total)*100.0, 2)
    score_str =  "Test score "+ str(score) +"%"
    print(score_str)
    log_file  = open(log_file_name, "a") 
    log_file.write(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S ')) 
    log_file.write(score_str)
    log_file.write('\n')
    log_file.close()     
    return (pred_data)

def predict_new(test_path):
    pred_data = []
    model = Model()
    model.load_model()

    for p in test_path:
        for dr in os.listdir(p):
            extension = dr.split(".")[-1] in ("jpg", "jpeg", "png")
            if extension:
                file = p+"/"+dr
                print("Predicting file ", file)
                img = Image.open(file)
                pred_result = model.predict_img(img)
                pred_value = np.argmax(pred_result, axis = 1)
                pred_index = pred_value[0]
                pred_class = lables[pred_index]
                """
                ln = []
                ln.append(file)
                ln.append(pred_class)
                ln.append(round(pred_result[0][pred_index]*100,2))
                pred_data.append(ln)
                """
                ln = []
                ln.append(file)
                ln.append(pred_class)
                for i in range(0,len(pred_result[0])):
                    confidence = f"{pred_result[0][i]*100:0.2f} %"
                    ln.append(confidence)
                pred_data.append(ln)
    return (pred_data)