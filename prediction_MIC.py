import re
import random
import pandas as pd
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(gpus[0], True)
import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import numpy as np
import pickle
import tensorflow.keras.backend as K
from time import time
from matplotlib import pyplot as plt

################## PARAMETERS ##################
Fold = 3
np.random.seed(3)

main_dir = "./KlebsiellaPneumoniae/"
data_map_file = main_dir + "data_map.csv"
kmc_dir = main_dir + "params/counts_data_8/"

LIST_ANTIBIOTICS = ['Cefazolin', 'Aztreonam', 'Ceftriaxone', 'Piperacillin/Tazobactam', 'Gentamicin', 'Amikacin', 'Levofloxacin', 'Ciprofloxacin', 'Ampicillin', 'Tetracycline', 'Nitrofurantoin', 'Ceftazidime', 'Cefoxitin', 'Imipenem', 'Meropenem', 'Tobramycin', 'Ampicillin/Sulbactam', 'Trimethoprim/Sulfamethoxazole', 'Cefuroxime sodium', 'Cefepime']

ANTIBIOTIC = LIST_ANTIBIOTICS[18]
print(ANTIBIOTIC)
ANTIBIOTIC_MODEL_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + ".h5"
print("ANTIBIOTIC_MODEL_FILE: ", ANTIBIOTIC_MODEL_FILE)

TEST_RESULT_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + "_test.csv"
VAL_RESULT_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + "_val.csv"
print("TEST_RESULT_FILE: ", TEST_RESULT_FILE)
print("VAL_RESULT_FILE: ", VAL_RESULT_FILE)

#################################################

def convertMIC(s):
    new_s = re.sub('\>([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))*2), s)
    new_s = re.sub('\<([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))/2), new_s)
    new_s = re.sub('\<=([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    new_s = re.sub('^([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    return float(new_s)
    
def convertKmers(filename):
    feature = np.zeros((DIMENSION_X, DIMENSION_Y), dtype=np.float32)
    current_kmers_list = set(list())
    #print('{:0.5f}'.format(filename))
    with open(kmc_dir + '{:0.5f}'.format(filename) , 'r') as data_file:
        for index, line in enumerate(data_file.read().splitlines()):
            kmer_counts = int(line.split("\t")[-1])
            kmer = line.split("\t")[0]
            current_kmers_list.add(kmer)
            kmer_index = all_8mers_dictionary[kmer]
            for i, el in enumerate(feature[kmer_index]):
                if kmer_counts == 1:
                    feature[kmer_index][1] = 1
                elif i == math.ceil(math.log(kmer_counts, BASE_NUMBER)):
                    feature[kmer_index][i] = 1
                # if math.ceil(math.log(kmer_counts, BASE_NUMBER)) == 511:
                #     print(kmer_counts)
            #print(kmer)
            #print(kmer_index)
            #print(kmer_counts)
            #print(feature[kmer_index])
    return feature

def create_fold(NB_SAMPLES, FOLD): 
    NB_SAMPLES_PER_FOLD = int(NB_SAMPLES/10)
    #print("NB_SAMPLES_PER_FOLD: ", NB_SAMPLES_PER_FOLD)
    
    lst_index = np.arange(NB_SAMPLES)
    #print(lst_index)
    np.random.shuffle(lst_index)
    #print(lst_index)

    test_index = lst_index[NB_SAMPLES_PER_FOLD * (FOLD-1) * 2:NB_SAMPLES_PER_FOLD * (FOLD-1) * 2 + NB_SAMPLES_PER_FOLD]
    val_index = lst_index[NB_SAMPLES_PER_FOLD * (FOLD-1) * 2 + NB_SAMPLES_PER_FOLD:NB_SAMPLES_PER_FOLD * FOLD * 2 ]
    train_index = np.concatenate([lst_index[:NB_SAMPLES_PER_FOLD * (FOLD-1) * 2], lst_index[NB_SAMPLES_PER_FOLD * FOLD * 2:]])
    
    return train_index, val_index, test_index

# + Input: file data_map.csv
# + Output: List [{'PATRIC ID': 573.1323, 'MIC': 4.0}, {'PATRIC ID': 573.13391, 'MIC': 4.0},...] cua mot ANTIBIOTIC
def getListData(ANTIBIOTIC):
    df = pd.read_csv(data_map_file)
    #print(df)
    df_filtered = df[df['Antibiotic'].eq(ANTIBIOTIC)]
    #print(df_filtered)

    NB_SAMPLES = len(df_filtered)
    #print("NB_SAMPLES: ", NB_SAMPLES)

    # Xac dinh MIC
    df_filtered['MIC'] = df_filtered['Actual MIC'].apply(convertMIC)
    #print(df_filtered)

    lst_data = []
    for index, row in df_filtered.iterrows():
       #print(row['PATRIC ID'], row['MIC'])
       lst_data.append({'PATRIC ID':row['PATRIC ID'], 'MIC': row['MIC']})

    print("lst_data: ", len(lst_data))
    #print(lst_data[:10])
    return lst_data

######## DATA LOADER ###########################
class DataGenerator(keras.utils.all_utils.Sequence):
    def __init__(self, lst_data, batch_size=8, shuffle=True):
        self.lst_data = lst_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end() 
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.lst_data) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        lst_data_temp = [self.lst_data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(lst_data_temp)

        return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.lst_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
   
    def __data_generation(self, lst_data_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, DIMENSION_X, DIMENSION_Y))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, item in enumerate(lst_data_temp):
            # Store sample
            X[i,] = convertKmers(item['PATRIC ID'])

            # Store class
            y[i] = item['MIC']

        return X, y
    
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3,
                    activation='relu',
                    padding='same',
                    input_shape=(DIMENSION_X, DIMENSION_Y, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, kernel_size=3,
                    padding='same',
                    activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.BatchNormalization(),

        # tf.keras.layers.Conv2D(256, kernel_size=3,
        #             activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Conv2D(256, kernel_size=3,
        #             activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        # tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(
                    optimizer='adam',
                    #optimizer=Adam(lr=1e-3),
                    loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"))
    return model

def train(lstTrain, lstVal):    
    training_generator = DataGenerator(lstTrain)
    validation_generator = DataGenerator(lstVal)
    
    # Train model on dataset
    model = get_compiled_model()
    print(model.summary())
    
    #early_stopping = EarlyStopping(patience=100, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=100, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, 
        verbose=1, mode='auto', epsilon=0.0001, cooldown=1, min_lr=1e-8)
    
    tensorboard = TensorBoard(log_dir='logs\\{}'.format(time()))
    #history = model.fit_generator(generator=training_generator,     
    history = model.fit(training_generator, 
                    validation_data=validation_generator,
                    epochs=400,
                    #epochs=1,
                    use_multiprocessing=True,
                    workers=4, 
                    #callbacks=[tensorboard]
                    callbacks=[tensorboard, early_stopping, reduce_lr]
                    )
    model.save(ANTIBIOTIC_MODEL_FILE)

def test(lstTest, fileResult):
    print("\n#### TEST PROCESS ")    
    
    print("Number of test samples: ", len(lstTest))
    
    model = tf.keras.models.load_model(ANTIBIOTIC_MODEL_FILE)
    print(model.summary())
    nbCorrect = 0
    f_res = open(fileResult, "w")
    f_res.write("PATRIC_ID, MIC, Predict MIC")
    for test_sample in lstTest:
        print("\n test_sample: ", test_sample)
        data_sample = convertKmers(test_sample['PATRIC ID'])
        target_sample = test_sample["MIC"]
        
        #print("data_sample: ", data_sample)
        #print("data_sample: ", data_sample.shape)
        #print("target_sample: ", target_sample)
        
        data_sample = tf.reshape(data_sample, [1, DIMENSION_X, DIMENSION_Y, 1])
        #print("data_sample: ", data_sample.shape)        
        res = model.predict(data_sample)
        #print("res: ", res)
        MIC_res = res[0][0]
        print("MIC_res: ", MIC_res)
        if (MIC_res >= target_sample / 2 ) and MIC_res <= target_sample * 2:
            print("TRUE")
            nbCorrect += 1
        else:
            print("FALSE")
        f_res.write("\n" + str(test_sample['PATRIC ID']) + "," + str(test_sample["MIC"]) + "," + str(MIC_res))
        
        #break
    
    print("Test ACC: ", nbCorrect, " / ", len(lstTest), " = ", nbCorrect / len(lstTest))
    f_res.write("\nTest ACC: " + str(nbCorrect) + " / " + str(len(lstTest)) + " = " + str( nbCorrect / len(lstTest)))
    
    f_res.close()
    
############### MAIN PROGRAM ##############
if __name__ == "__main__":    
    with open (main_dir + 'params/all_8mers', 'rb') as fp:
        all_8mers = pickle.load(fp)
    all_8mers_list = list(all_8mers)
    all_8mers_list.sort()
    all_8mers_dictionary = { kmer : i for (i, kmer) in enumerate(all_8mers_list) }

    lstData = getListData(ANTIBIOTIC)
    NB_SAMPLES = len(lstData)
    
    BASE_NUMBER = 2
    DIMENSION_X = len(all_8mers)
    DIMENSION_Y = 14

    print("DIMENSION_X: ", DIMENSION_X, "  DIMENSION_Y: ", DIMENSION_Y)    
    
    train_index, val_index, test_index = create_fold(NB_SAMPLES, Fold)    
    
    lstTrain = [lstData[i] for i in train_index]
    lstVal = [lstData[i] for i in val_index]
    lstTest = [lstData[i] for i in test_index]
        
    print("lstTrain: ", len(lstTrain))
    print("lstVal: ", len(lstVal))
    print("lstTest: ", len(lstTest))
    
    train(lstTrain, lstVal)    
    
    test(lstTest, TEST_RESULT_FILE)    
    

    




