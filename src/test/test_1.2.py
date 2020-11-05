import librosa
import numpy as np
from keras import models
import os


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
        start = time.time()
        y = librosa.resample(y, sr, 16000)
        print("time :", time.time() - start)
    
    return y

def make_patches(y):
    print("mfcc 추출")
    sr = 16000
    test = []
    
    for i in tqdm(range(int((len(y)/sr))-1)):
        p = i *sr
        q = p + (sr*2)
        split = y[p:q]
        
        mfcc = librosa.feature.mfcc(split, sr=sr)
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
    y_pred = np.argmax(Y_pred >= 0.8,axis=1)
    
    detect = get_result(y, y_pred, test_audio)
    
    move_file(test_audio, detect)

def get_result(y, y_pred, test_audio):
    detect = 0
    
    print("결과\n",test_audio + "파일")
    
    for i in range(len(y_pred)):
        if(y_pred[i] == 1):
            detect = 1
            time = get_time(i)
            print(time[0], "분 ", time[1], "초 ~ ", time[2], "분 ", time[3], "초 불이야 감지")
            split_file(y, i, test_audio, time)
            
    if(not detect):
        print("불이야 감지 못함")
    
    return detect

def get_time(i):
    end = i+2
    time = []
    
    time.append(str(int(i / 60)))
    time.append(str(int(i % 60)))
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
            input_audio_list = os.listdir(input_audio)
            
            if(len(input_audio_list)):
                predict(input_audio_list[0], model)
                
        except KeyboardInterrupt:
            print("종료")
            break
        
if __name__ == "__main__":
    main()
    sys.exit()