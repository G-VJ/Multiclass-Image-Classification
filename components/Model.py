import os
import cv2
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import vgg16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

img_size = 150
batch_size = 24
epochs = 10
learn_rate = 0.01
momentum = 0.6
folder_2_class = {'buildings': 0, 'sea': 1, 'glacier':2, 'mountain':3, 'forest':4, 'street':5}
log_file_name = './log/train_log.txt'
class Model:
    
    def predict_img(self, image_ip):
        img = image_ip.resize((img_size,img_size))
        pred_img = np.expand_dims(img, axis=0)
        pred_img = pred_img / 255.0
        #predict the image 
        pred_result = self.model.predict(pred_img)
        return pred_result
    
    def __init__(self):
        self.model = ''
        
         
    def load_model(self):
        self.model = tf.keras.models.load_model('./saved_model/transfer_lrn_model_02012021_131644')      
    
    def load_data(self, train_path):
        train_data = []
        train_label = []

        for ex in os.listdir(train_path):
            for i in range(0,len(os.listdir(train_path+ex+'/'))):
                path = train_path+ex+'/'+os.listdir(train_path + ex)[i]
                extension = path.split(".")[-1] in ("jpg", "jpeg", "png")
                if extension:
                    img = load_img(path, target_size=(img_size, img_size))
                    img_arr = img_to_array(img)
                    train_data.append(img_arr)
                    train_label.append(folder_2_class.get(ex))
        print("Train data and Label:", len(train_data))
        
        #shfulling the data
        train_data, train_label = shuffle(train_data, train_label)
        
        #normalizer and reshape data
        train_data=np.array(train_data)/255.0
        train_data=np.reshape(train_data,(train_data.shape[0],img_size,img_size,3))
        train_label=np.array(train_label)
        train_label = to_categorical(train_label)
        
        #split data
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2)
        return X_train, X_test, y_train, y_test
    
    def image_datagen_fnl(self, path):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=0.2        
            )
        
        train_path = path +'/seg_train'
        train_genrator = train_datagen.flow_from_directory(
            train_path,
            subset="training",
            target_size=(img_size, img_size),
            shuffle=True,
            batch_size=batch_size)
        
        validation_generator = train_datagen.flow_from_directory(
            train_path,
            subset="validation",
            target_size=(img_size, img_size),
            batch_size=batch_size)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_path = path +'/seg_test'
        test_genrator = test_datagen.flow_from_directory(
            test_path,
            target_size=(img_size, img_size),
            shuffle=True,        
            batch_size=batch_size)    
        
        return train_genrator, test_genrator, validation_generator
    
    def model_create_optmiser_tnfrlr(self, learn_rate, momentum):
        base_model = vgg16.VGG16( include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        base_model.trainable = False
        for layer in base_model.layers:
            if ('block5'in layer.name):   # or ('block4'in layer.name):
                layer.trainable = True 
        
        x_op = base_model.output
        x = tf.keras.layers.Flatten()(x_op)
        output = tf.keras.layers.Dense(6, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.inputs, outputs=output) 
        
        #optimizer
        optimizer = tf.keras.optimizers.SGD(lr=learn_rate, momentum=momentum)      
        # model compile
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    def model_train(self, path):
        #create model
        model_transfer = self.model_create_optmiser_tnfrlr(learn_rate=learn_rate, momentum=momentum )
        
        print("Path:", path)
        #get data gen
        #train_datagen , test_datagen, val_datagen = self.image_datagen_fnl(path)
        print("Reading and processing data")
        X_train, X_test, y_train, y_test = self.load_data(path)
        
        #train model
        """
        history = model_transfer.fit(train_datagen,
                          epochs=epochs,
                          validation_data=val_datagen)
        """
        print("training model")
        history = model_transfer.fit(X_train,
                                     y_train,
                                     epochs=epochs,
                                     validation_split=0.2)
        
        scoreTest = model_transfer.evaluate(X_test, y_test, verbose=0)
        print("Test loss, Test accuracy:", scoreTest)
        
        time = datetime.datetime.now().strftime('_%d%m%Y_%H%M%S')
        model_name = "transfer_lrn_model" + time
        model_full_path = "./saved_model/" + model_name
        model_transfer.save(model_full_path)
        print("model saved at", model_full_path) 

        log_file  = open(log_file_name, "a") 
        log_file.write(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')) 
        log_file.write(' Test loss: ') 
        log_file.write(str(round(scoreTest[0],4)))
        log_file.write(' Test accuracy: ')
        log_file.write(str(round(scoreTest[1]*100,2)))
        log_file.write('% Model Name: ')
        log_file.write(model_name)
        log_file.write('\n')
        log_file.close() 