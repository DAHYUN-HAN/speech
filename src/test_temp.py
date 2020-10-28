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
    
    sr = 16000
    test = []
    for i in range(int((len(y)/sr))-1):
        p = i *sr
        q = p + (sr*2)
        split = y[p:q]
        
        max_mine = np.max(split)
        ratio = 0.46202388 / max_mine
        split = split * ratio
        
        mfcc = librosa.feature.mfcc(split, sr=sr)
        test.append(mfcc)
    test = np.array(test)
    test_X = np.expand_dims(test, -1)
    
    return test_X

def get_model(model_dir):
    print("get_model 실행")#
    return models.load_model(model_dir)

def predict(test_audio, model_dir):
    print("predict 실행")#
    test_X = make_patches(load(test_audio))
    model = get_model(model_dir)
    
    Y_pred = model.predict(test_X)
    y_pred = np.argmax(Y_pred,axis=1)
    
    return y_pred

def get_time(y_pred):
    detect = False
    for i in range(len(y_pred)):
        if(y_pred[i] == 1):
            detect = True
            print(str(i) , "초 ~", str((i+2)) , "초 불이야 감지")
            
    if(not detect):
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
    
    get_time(predict(test_audio, model_dir))
    print("time :", time.time() - start)#


if __name__ == "__main__":
    main()