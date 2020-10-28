import librosa
import numpy as np
from keras import models
import os
import argparse

def load(file_name):
    wav = file_name
    file_sr = librosa.get_samplerate(wav)
    y, sr = librosa.load(wav, sr=file_sr)
    
    if(sr != 16000):
        y = librosa.resample(y, sr, 16000)
    
    return y

def make_patches(y):
    
    sr = 16000
    test = []
    for i in range(int((len(y)/sr))-1):
        p = i *sr
        q = p + (sr*2)
        split = y[p:q]
        
#         max_mine = np.max(split)
#         ratio = 0.46202388 / max_mine
#         split = split * ratio
        
        mfcc = librosa.feature.mfcc(split, sr=sr)
        test.append(mfcc)
    test = np.array(test)
    test_X = np.expand_dims(test, -1)
    
    return test_X

def predict(test_audio, model):
    test_X = make_patches(load(test_audio))
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_audio", type=str, help="input audio, wav file", required = True)
    
    args = parser.parse_args()
    test_audio = args.test_audio
    
    model = models.load_model("model/test_model.h5")
    
    get_time(predict(test_audio, model))

if __name__ == "__main__":
    main()