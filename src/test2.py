import librosa
import numpy as np
from keras import models
import os
import argparse
import time#

def load(file_name):
    print("load 실행")#
    wav = file_name
    file_sr = librosa.get_samplerate(wav)
    y, sr = librosa.load(wav, sr=file_sr)
    
    if(sr != 16000):
        print("resampling 진행")
        y = librosa.resample(y, sr, 16000)
    
    return y

def make_patches(y):
    print("make_patches 실행")#
    
    test = []
    for i in range(int((len(y)/1600))-19):
        p = i * 1600
        q = p + 32000
        split = y[p:q]
        
        max_mine = np.max(split)
        ratio = 0.46202388 / max_mine
        split = split * ratio
        
        mfcc = librosa.feature.mfcc(split, sr=16000)
        test.append(mfcc)
    test = np.array(test)
    test_X = np.expand_dims(test, -1)
    
    return test_X

def predict(test_audio, model):
    print("predict 실행")#
    test_X = make_patches(load(test_audio))
    
    Y_pred = model.predict(test_X)
    y_pred = np.argmax(Y_pred,axis=1)
    
    return y_pred

def get_time(fire_predict):
    for i in range(len(fire_predict)):
        k = round(fire_predict[i] * 0.1,2)
        print(str(k) , "초 ~", str((k+2)) , "초 불이야 감지")
            
        
def get_result(y_pred):
    fire_count = 0
    non_count = 0
    fire_predict = []
    temp_predict = []
    
    for i in range(len(y_pred)):
        
        if(y_pred[i] == 0):
            non_count = non_count + 1
        
            if(non_count >= 4):
                fire_count = 0
        
        else :
            fire_count = fire_count + 1
            non_count = 0
            if(fire_count >= 6):
                temp_predict.append(i)
    n = 0
    for i in range(len(temp_predict)):
        if(temp_predict[i] > n):
            fire_predict.append(temp_predict[i])
            n = temp_predict[i] + 20
    
    if(fire_predict):
        print("갯수",len(fire_predict))
        get_time(fire_predict)
    else:
        print("불이야 감지 못함")

def main():
    start = time.time()#
    print("main 실행")#
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_audio", type=str, help="input audio, wav file", required = True)
    parser.add_argument("--model_dir", type=str, help="model, h5 file", required = True)
    
    args = parser.parse_args()
    test_audio = args.test_audio
    model_dir = args.model_dir
    
    model = models.load_model(model_dir)
    
    get_result(predict(test_audio, model))
    print("time :", time.time() - start)#


if __name__ == "__main__":
    main()