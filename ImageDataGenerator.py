import numpy as np
import keras
import pandas as pd
import cv2
import os.path
import os
import subprocess 
from joblib import Parallel, delayed


class ImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dir_, item_ids, image_ids, labels, df, title_seq, des_seq,cont_features, batch_size=32, dim=(160,160), n_channels=3, shuffle=True, is_train=True):
        'Initialization'
        self.dir = dir_
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.image_ids = image_ids
        self.item_ids = item_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.is_train = is_train
        self.df = df
        self.title_seq = title_seq
        self.des_seq = des_seq
        self.cont_features= cont_features
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.item_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size > len(self.item_ids):

            indexes = self.indexes[index*self.batch_size:]

        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        item_ids_temp = [self.item_ids[k] for k in indexes]

        image_X = self.__data_generation(item_ids_temp)
        df = self.df.iloc[indexes]
        
        des_seq = self.des_seq[indexes]
        title_seq = self.title_seq[indexes]
        
        
        user_type = df.user_type.values
        category_name = df.category_name.values
        parent_category_name = df.parent_category_name.values
        param_1 = df.param_1.values
        param_2 = df.param_2.values
        param_3 = df.param_3.values
        region = df.region.values
        city = df.city.values
        image_top_1 = df.image_top_1.values
        price_p = df.price_p.values
        item_seq_number_cat = df.item_seq_number_cat.values
        continuous_inputs = [df[feat].values for feat in self.cont_features ]
        
        if not self.is_train:
            return [des_seq, title_seq, user_type , category_name, parent_category_name, param_1, param_2, param_3,
                   region,city, image_top_1, price_p, item_seq_number_cat, image_X, *continuous_inputs]
        
        
        y = self.labels[indexes]
        return [des_seq, title_seq, user_type , category_name, parent_category_name, param_1, param_2, param_3,
                   region,city, image_top_1, price_p, item_seq_number_cat, image_X, *continuous_inputs], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.item_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _load_image(self, item_id):
        image_id = self.image_ids[item_id]
        try:
            fname = f'{self.dir}/{image_id}.jpg'
            img = cv2.imread(fname)
            img = cv2.resize(img, self.dim, interpolation = cv2.INTER_LINEAR)
            return img
        except cv2.error as e:
            return np.zeros([*self.dim, self.n_channels])
        except:
            return np.zeros([*self.dim, self.n_channels])
        

    
    def __data_generation(self, item_ids_temp):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((len(item_ids_temp), *self.dim, self.n_channels))

        # Generate data
        for i, item_id in enumerate(item_ids_temp):
            image_id = self.image_ids[item_id]
           
            fname = f'{self.dir}/{image_id}.jpg'
            if os.path.isfile(fname):
                img = cv2.imread(fname)
                try:
                    img = cv2.resize(img, self.dim, interpolation = cv2.INTER_LINEAR)
                except cv2.error as e:
                    img = np.zeros([*self.dim, self.n_channels])
            else: 
                img = np.zeros([*self.dim, self.n_channels])

            
            X[i,] = img


        return X
    
    