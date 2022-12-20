import re
import random
import pandas as pd
import math
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(gpus[0], True)
import keras
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import numpy as np
import pickle
import tensorflow.keras.backend as K
from time import time
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


################## PARAMETERS ##################
Fold = 4
np.random.seed(3)

MY_RANDOM_STATE = 3

main_dir = "./KlebsiellaPneumoniae/"
data_map_file = main_dir + "data_map.csv"
kmc_dir = main_dir + "params/counts_data_8/"

LIST_ANTIBIOTICS = [
    ('Amikacin', 64, 16), ('Ampicillin', 32, 8), 
    ('Ampicillin/Sulbactam', 32, 8), ('Aztreonam', 16, 4), 
    ('Cefazolin', 32, 16), ('Cefepime', 16, 2), 
    ('Cefoxitin', 32, 8), ('Ceftazidime', 16, 4), 
    ('Ceftriaxone', 4, 0.5), ('Cefuroxime sodium', 32, 8), 
    ('Ciprofloxacin', 4, 1), ('Gentamicin', 16, 4), 
    ('Imipenem', 4, 1), ('Levofloxacin', 8, 2), 
    ('Meropenem', 4, 1), ('Nitrofurantoin', 128, 32), 
    ('Piperacillin/Tazobactam', 128, 16), ('Tetracycline', 16, 4), 
    ('Tobramycin', 16, 4), ('Trimethoprim/Sulfamethoxazole', 4, 2) 
    ]
LIST_THRESHOLD_RESISTANCE = []
LIST_THRESHODL_SUSCEPTIBLE = []

ANTIBIOTIC_INDEX = 0
ANTIBIOTIC = LIST_ANTIBIOTICS[ANTIBIOTIC_INDEX][0]
THRESHOLD_RESISTANT = LIST_ANTIBIOTICS[ANTIBIOTIC_INDEX][1]
THRESHOLD_SUSCEPTIBLE = LIST_ANTIBIOTICS[ANTIBIOTIC_INDEX][2]

print(ANTIBIOTIC, THRESHOLD_RESISTANT, THRESHOLD_SUSCEPTIBLE)

#sys.exit(0)
ANTIBIOTIC_MODEL_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + ".h5"
print("ANTIBIOTIC_MODEL_FILE: ", ANTIBIOTIC_MODEL_FILE)

OLD_MODEL_FILE = "./models_EarlyStopping/" + ANTIBIOTIC_MODEL_FILE
print("OLD_MODEL_FILE: ", OLD_MODEL_FILE)

TEST_RESULT_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + "_test.csv"
VAL_RESULT_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + "_val.csv"
print("TEST_RESULT_FILE: ", TEST_RESULT_FILE)
print("VAL_RESULT_FILE: ", VAL_RESULT_FILE)

HISTORY_FILE = ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_fold" + str(Fold) + "_history.pkl"
print("HISTORY_FILE: ", HISTORY_FILE)

#################################################
# Positive: highly resistant (MIC >8mg/liter) 
# Negative: low/intermediately resistant (MIC ≤ 8mg/liter). 
def getClassMIC(mic):
    if mic <= 8: return 0
    else: return 1

def sig(x):
    return 1/(1 + np.exp(-x))
 
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

def train_test_10_folds(): 
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=MY_RANDOM_STATE)    
    
    fold_index = 1
    for train_index, test_index in kf.split(lst_data, lst_label):
        train_data = [lst_data[idx] for idx in train_index]        
        train_label = [lst_label[idx] for idx in train_index]
        
        test_data = [lst_data[idx] for idx in test_index]        
        test_label = [lst_label[idx] for idx in test_index]
        
        if fold_index == Fold: 
            #train_test(train_data, train_label, test_data, test_label)        
        
            trainAdditionalEpochs(train_data, train_label, test_data, test_label, OLD_MODEL_FILE, 1)
                        
    
        fold_index = fold_index + 1
        #break
        
def train_test(train_data, train_label, test_data, test_label):
    print("Train_data: ", sum(train_label), "/", len(train_data))
    print("Test_data: ", sum(test_label), "/", len(test_data))
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_label, test_size=0.1, random_state=42)
    
    train(X_train, X_val, y_train, y_val)    
    
    test(test_data, test_label, TEST_RESULT_FILE)

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
    lst_label = []
    nbResistant = 0
    nbSusceptible = 0
    for index, row in df_filtered.iterrows():
       #print(row['PATRIC ID'], row['MIC'])       
        if float(row['MIC']) >= THRESHOLD_RESISTANT: 
            lst_data.append({'PATRIC ID':row['PATRIC ID'], 'MIC': row['MIC']})
            lst_label.append(1)
            nbResistant = nbResistant + 1
        if float(row['MIC']) <= THRESHOLD_SUSCEPTIBLE: 
            lst_data.append({'PATRIC ID':row['PATRIC ID'], 'MIC': row['MIC']})
            lst_label.append(0)
            nbSusceptible = nbSusceptible + 1
            
    print("lst_data: ", len(lst_data))
    #print(lst_data[:10])
    print("nbResistant: ", nbResistant, " nbSusceptible: ", nbSusceptible)
    
    #print(sum(lst_label))
    return lst_data, lst_label

######## DATA LOADER ###########################
class DataGenerator(keras.utils.all_utils.Sequence):
    def __init__(self, lst_data, lst_label, batch_size=8, shuffle=True):
        self.lst_data = lst_data
        self.lst_label = lst_label
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
        lst_label_temp = [self.lst_label[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(lst_data_temp, lst_label_temp)

        return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.lst_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
   
    def __data_generation(self, lst_data_temp, lst_label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, DIMENSION_X, DIMENSION_Y))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, item in enumerate(lst_data_temp):
            #print(i, item)
            # Store sample
            X[i,] = convertKmers(item['PATRIC ID'])

            # Store class
            y[i] = lst_label_temp[i]

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
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64),
        #tf.keras.layers.Dense(2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),        
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(1, activation='linear')
        #tf.keras.layers.Dense(1, activation='softmax')
    ])

    model.compile(
                    #optimizer='adam',
                    optimizer=Adam(lr=1e-3),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    #loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
                    #metrics=['accuracy'],                    
                    )
    return model

def trainAdditionalEpochs(X_trainval, y_trainval, test_data, test_label, OLD_MODEL_FILE, nEpochs):
    print("OLD_MODEL_FILE: ", OLD_MODEL_FILE)
    
    train_val_generator = DataGenerator(X_trainval, y_trainval)
    model = tf.keras.models.load_model(OLD_MODEL_FILE)
    print(model.summary())
    
    # To get learning rate
    print(K.get_value(model.optimizer.lr))
    # To set learning rate
    K.set_value(model.optimizer.lr, 0.0001)
    
    # Train one epoch
    history = model.fit(train_val_generator, epochs=nEpochs, workers=4)
    model.save(ANTIBIOTIC_MODEL_FILE)
    
    test(test_data, test_label, TEST_RESULT_FILE)    
    
def train(X_train, X_val, y_train, y_val):  
    print("\n X_train: ", len(X_train))
    print("X_val: ", len(X_val))
    training_generator = DataGenerator(X_train, y_train)
    validation_generator = DataGenerator(X_val, y_val)
    
    # Train model on dataset
    model = get_compiled_model()
    print(model.summary())
    
    early_stopping = EarlyStopping(patience=50, verbose=1)
    #early_stopping = EarlyStopping(monitor='val_accuracy', patience=300, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, 
        verbose=1, mode='auto', epsilon=0.0001, cooldown=1, min_lr=1e-8)
    
    tensorboard = TensorBoard(log_dir='logs\\{}'.format(time()))
    
    #Number of samples:  , Class 0:  1320, Class 1:  103
    class_weight = {0: 1., 1: 14.}
    #history = model.fit_generator(generator=training_generator,     
    history = model.fit(training_generator, 
                    validation_data=validation_generator,
                    class_weight=class_weight, 
                    epochs=400,
                    #epochs=150,
                    #epochs=1,
                    use_multiprocessing=True,
                    workers=4, 
                    #callbacks=[tensorboard]
                    #callbacks=[tensorboard, early_stopping, reduce_lr]
                    callbacks=[tensorboard, early_stopping]
                    )
        
    model.save(ANTIBIOTIC_MODEL_FILE)
        
    outfile = open(HISTORY_FILE,'wb')
    pickle.dump(history.history,outfile)
    outfile.close()
    
#    infile = open(HISTORY_FILE,'rb')
#    new_dict = pickle.load(infile)
#    print(new_dict)
#    infile.close()
    print_history(HISTORY_FILE)

def print_history(history_file):
    infile = open(HISTORY_FILE,'rb')
    history = pickle.load(infile)
    print(history)
    infile.close()
    
    print(history.keys())

    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')
    #plt.savefig('acc.png', bbox_inches='tight')
    #plt.show()

    file_loss = "loss.png"
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig(file_loss, bbox_inches='tight')
    
def test(test_data, test_label, fileResult):
    print("\n#### TEST PROCESS ")    
    
    print("Number of test samples: ", len(test_data))
    
    model = tf.keras.models.load_model(ANTIBIOTIC_MODEL_FILE)
    print(model.summary())
    nbCorrect = 0
    f_res = open(fileResult, "w")
    f_res.write("PATRIC_ID, MIC, Reference, Hypothese")
    lst_ref = np.zeros(len(test_data))
    lst_hyp_prob = np.zeros(len(test_data))
    idx = 0
    for test_sample in test_data:
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
        
        hyp_prob = sig(res[0][0])
        if hyp_prob > 0.5: hyp = 1
        else: hyp = 0
        
        #print("hyp prob: ", hyp_prob, " hyp: ", hyp)
        
        ref = test_label[idx]
        #print("ref: ", ref)
        if hyp == ref: nbCorrect = nbCorrect + 1
        
        lst_ref[idx] = ref
        lst_hyp_prob[idx] = hyp_prob
        
        f_res.write("\n" + str(test_sample['PATRIC ID']) + "," + str(test_sample["MIC"]) + "," + str(ref) + "," + str(hyp_prob))
        
        idx = idx + 1
        #break
    
    acc, auc_roc, auc_pr, kappa, sensitivity, specificity, mcc, F1 = process_result(lst_hyp_prob, lst_ref)
    
    results = '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "," + '{0:.4f}'.format(F1) + "\n"
    f_res.write("\nAccuracy,AUC-ROC,AUC-PR,Kappa,Sensitivity,Specificity,MCC,F1\n")
    #results = '{0:.4f},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{5:.4f},{6.4f},{6.4f}'.format(acc, auc_roc, auc_pr, kappa, sensitivity, specificity, mcc, F1) + "\n"
    f_res.write(results)
        
    #print("Test ACC: ", nbCorrect, " / ", len(lstTest), " = ", nbCorrect / len(lstTest))
    #f_res.write("\nTest ACC: " + str(nbCorrect) + " / " + str(len(lstTest)) + " = " + str( nbCorrect / len(lstTest)))
    
    f_res.close()

# File result : PATRIC_ID, MIC, Predict MIC
def process_result_file(fileResult, threshold = 0.5):
    f_res = open(fileResult)
    lines = f_res.readlines()
    #print(lines)
    lst_ref = np.zeros(len(lines) - 2)
    lst_hyp_prob = np.zeros(len(lines) - 2)
    idx = 0
    for line in lines[1:-2]:
        print(line.strip())
        #lst_ref[idx] = getClassMIC(float(line.strip().split(",")[1]))
        #lst_hyp_prob[idx] = float(line.strip().split(",")[2])
        
        lst_ref[idx] = float(line.strip().split(",")[2])
        lst_hyp_prob[idx] = float(line.strip().split(",")[3])
        
        idx = idx + 1
    f_res.close()
    
    #print(lst_ref)
    #print(lst_hyp_prob)
    
    acc, auc_roc, auc_precision_recall, kappa, sensitivity, specificity, mcc, F1 = process_result(lst_hyp_prob, lst_ref, threshold)
    
    results = '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_precision_recall) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "," + '{0:.4f}'.format(F1) + "\n"
    
    print("acc, auc_roc, auc_precision_recall, kappa, sensitivity, specificity, mcc, F1")
    print(results)
    
# Tinh accuracy, sensitivity, specificity
# Note that in binary classification, recall of the positive class is also known as "sensitivity"; recall of the negative class is "specificity".
def process_result(test_scores, test_labels, threshold = 0.5):
    #print("test_scores: ", test_scores)
    print("test_scores: ", test_scores.shape)
    #print("test_labels: ", test_labels)
    print("test_labels: ", test_labels.shape)
    
    test_hyp = []
    for score in test_scores:
        if score > threshold: test_hyp.append(1)
        else: test_hyp.append(0)
    
    conf_mat = confusion_matrix(test_labels, test_hyp)
    print("confusion_matrix: ", conf_mat.shape, "\n", conf_mat)
    
    acc = accuracy_score(test_labels, test_hyp)
    print("Accuracy: ", acc)
    
    target_names = ['class 0 : Negative', 'class 1: Positive']
    print(classification_report(test_labels, test_hyp, target_names=target_names))
    
    specificity = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    sensitivity = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
    tp = conf_mat[1][1]
    tn = conf_mat[0][0]
    fp = conf_mat[1][0]
    fn = conf_mat[0][1]
    mcc = (tp * tn - fp * fn ) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    F1 = 2*tp / (2*tp + fp + fn)

    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("mcc: ", mcc)
    print("F1: ", F1)
    
    # Tinh Kappa
    #p0 = (tp + tn) / (tp + fn + tn + fp)
    #pe = (tp + fn) * (tp + fp) * (tn + fn) * (tn + fp) / (tp + fn + tn + fp) / (tp + fn + tn + fp)
    #kappa = (p0 - pe) / (1 - pe)
    #print("p0: ", p0, " pe: ", pe)
    #print("Kappa: ", kappa)
    
    kappa = 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn)*(fn + tn))
    print("Kappa: ", kappa)
    
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    auc_roc = auc(fpr,tpr)
    print("Test AUC: ", auc_roc)
    
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)
    auc_precision_recall = auc(recall, precision)
    
    return acc,auc_roc,auc_precision_recall,kappa,sensitivity,specificity,mcc,F1

def count_nb_classes(lstData):
    print(lstData)
    nbClass0 = 0
    nbClass1 = 0
    for item in lstData: 
        label = getClassMIC(item['MIC'])
        if label == 0: nbClass0 = nbClass0 + 1
        if label == 1: nbClass1 = nbClass1 + 1
        
    print("Number of samples: ", len(lstData))
    print("Class 0: ", nbClass0)
    print("Class 1: ", nbClass1)
    
############### MAIN PROGRAM ##############
if __name__ == "__main__":    
    with open (main_dir + 'params/all_8mers', 'rb') as fp:
        all_8mers = pickle.load(fp)
    all_8mers_list = list(all_8mers)
    all_8mers_list.sort()
    all_8mers_dictionary = { kmer : i for (i, kmer) in enumerate(all_8mers_list) }

    lst_data, lst_label = getListData(ANTIBIOTIC)
    
    BASE_NUMBER = 2
    DIMENSION_X = len(all_8mers)
    DIMENSION_Y = 14

    print("DIMENSION_X: ", DIMENSION_X, "  DIMENSION_Y: ", DIMENSION_Y)    
    
    train_test_10_folds()
    
    train(lstTrain, lstVal)    
    
    test(lstTest, TEST_RESULT_FILE)
    
    #print_history(HISTORY_FILE)
    
    # Train more one epoch
    OLD_MODEL_FILE = "./models_EarlyStopping/Imipenem_fold5.h5"
    trainAdditionalOneEpoch(OLD_MODEL_FILE)
    
    test(lstTest, TEST_RESULT_FILE)
    
    process_result_file(TEST_RESULT_FILE, 0.5)
    
    

    

    




