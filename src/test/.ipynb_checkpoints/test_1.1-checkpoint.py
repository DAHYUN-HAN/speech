import librosa
import numpy as np
from keras import models
import os
import keyboard
from tqdm import tqdm
import shutil
import soundfile as sf
import sys

input_audio = ("audio/input_audio/")
output_origin_audio = ("audio/output_origin_audio/")
output_split_audio = ("audio/output_split_audio/")
model_dir = ("model/test_model.h5")

def get_model():
    return models.load_model(model_dir)

def load(file_name):
    wav = file_name
    file_sr = librosa.get_samplerate(wav)
    y, sr = librosa.load(wav, sr=file_sr)
    
    if(sr != 16000):
        print("reampling")
        y = librosa.resample(y, sr, 16000)
    
    return y

def make_patches(y):
    print("mfcc 추출")
    sr = 16000
    test = []
    for i in tqdm(range(int((len(y)/sr))-1)):
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
    
    y = load(input_audio+test_audio)
    
    test_X = make_patches(y)
    
    Y_pred = model.predict(test_X)
    y_pred = np.argmax(Y_pred,axis=1)
    
    detect = get_time(y, y_pred, test_audio)
    
    move_file(test_audio, detect)

def get_time(y, y_pred, test_audio):
    detect = False
    
    print("결과\n",test_audio + "파일")
    
    for i in range(len(y_pred)):
        if(y_pred[i] == 1):
            detect = True
            print("\t"+str(i)+"초 ~ " +str((i+2))+"초 불이야 감지")
            #split_file(y, i, test_audio)
            return 1
            
    if(not detect):
        print("불이야 감지 못함")
        return 0
        
def split_file(y, i, test_audio):
    start = i*16000
    split = y[start:start+32000]
    sf.write(output_split_audio+test_audio[:-4] + " "  + str(i) +"초 ~" + str(i+2) + "초.wav", split, 16000, subtype='PCM_16')
        
def move_file(test_audio, detect):
    if(detect):
        shutil.move(input_audio+test_audio, output_origin_audio + "exist/" + test_audio)
    else:
        shutil.move(input_audio+test_audio, output_origin_audio + "non/" + test_audio)

def main():
    
    model = get_model()
    print("시작")
    while True:
        try:
            if keyboard.is_pressed('q'):
                print('종료')
                break
            
            input_audio_list = os.listdir(input_audio)
            
            if(len(input_audio_list)):
                print("파일확인")
                predict(input_audio_list[0], model)
            
            
        except:
            break

        
if __name__ == "__main__":
    main()
    sys.exit()