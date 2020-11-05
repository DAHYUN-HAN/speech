import librosa
import numpy as np
from keras import models
import os
import keyboard
from tqdm import tqdm
import shutil
import soundfile as sf
import sys
import time

input_audio = ("audio/input_audio/")
output_origin_audio = ("audio/output_origin_audio/")
output_split_audio = ("audio/output_split_audio/")
model_dir = ("model/test_model_3.h5")

def get_model():
    return models.load_model(model_dir)

def load(file_name):
    wav = file_name
    file_sr = librosa.get_samplerate(wav)
    y, sr = librosa.load(wav, sr=file_sr)
    
    if(sr != 16000):
#         print(sr, "reampling")
        y = librosa.resample(y, sr, 16000)
    
    return y

def make_patches(y):
    print("mfcc 추출")
    
    test = []
    for i in tqdm(range(int((len(y)/1600))-19)):
        p = i * 1600
        q = p + 32000
        split = y[p:q]
        
        mfcc = librosa.feature.mfcc(split, sr=16000)
        test.append(mfcc)
    test = np.array(test)
    test_X = np.expand_dims(test, -1)
    
    return test_X

def predict(test_audio, model):
    y = load(input_audio+test_audio)
    
    if(len(y) < 32000):
        detect = -1
        print("2초 이하 파일")
        move_file(test_audio, detect)
        return 0
    
    test_X = make_patches(y)
    
    Y_pred = model.predict(test_X)
    y_pred = np.argmax(Y_pred,axis=1)
    
    detect = get_result(y, y_pred, test_audio)
    
    move_file(test_audio, detect)            
        
def get_result(y, y_pred, test_audio):
    fire_count = 0
    non_count = 0
    pass_count = 21
    fire_predict = []
    
    for i in range(len(y_pred)):
        
        if(y_pred[i] == 1):
            fire_count = fire_count + 1
            
            if(fire_count >= 5):
                non_count = 0
                if(pass_count > 20):
                    pass_count = 0
                    fire_predict.append(i)
        
        else :
            non_count = non_count + 1
            if(non_count >= 5):
                fire_count = 0
                
        pass_count = pass_count + 1
            
    print("결과\n",test_audio + "파일")
    
    if(fire_predict):
        result(y, fire_predict, test_audio)
        return 1
    else:
        print("불이야 감지 못함")
        return 0

def result(y, fire_predict, test_audio):
    for i in range(len(fire_predict)):
        k = round(fire_predict[i] * 0.1,2)
        time = get_time(k)
        print(time[0], "분 ", time[1], "초 ~ ", time[2], "분 ", time[3], "초 불이야 감지")
        split_file(y, k, test_audio, time)

def get_time(k):
    end = k+2
    time = []
    
    time.append(str(int(k / 60)))
    time.append(str(int(k % 60)))
    time.append(str(int(end / 60)))
    time.append(str(int(end % 60)))
    return time
        
def split_file(y, i, test_audio, time):
    path = make_path(test_audio)
    
    start = int(i*16000)
    split = y[start:start+32000]
    file_name = test_audio[:-4] + " " + time[0]+"분 "+time[1]+"초 ~ "+time[2]+"분 "+time[3]+"초.wav"

    sf.write(path+file_name, split, 16000, subtype='PCM_16')

def make_path(test_audio):
    path = output_split_audio + test_audio[:-4]
    if not os.path.isdir(path):                                                           
        os.mkdir(path)
    return path+"/"
        
def move_file(test_audio, detect):
    if(detect == 1):
        shutil.move(input_audio+test_audio, output_origin_audio + "exist/" + test_audio)
    elif(detect == -1):
        shutil.move(input_audio+test_audio, output_origin_audio + "fail/" + test_audio)
    else:
        shutil.move(input_audio+test_audio, output_origin_audio + "non/" + test_audio)

def main():
    
    model = get_model()
    print("시작. 파일 load 가능")
    while True:
        try:
            if keyboard.is_pressed('q'):
                print('종료')
                break
            
            input_audio_list = os.listdir(input_audio)
            
            if(len(input_audio_list)):
#                 print("파일확인")
                start = time.time()
                predict(input_audio_list[0], model)
                print("time :", time.time() - start)
            
        except:
            break

if __name__ == "__main__":
    main()