import numpy as np
import os
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.engine import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

#TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, LSTM, RepeatVector, Permute, Multiply, 

def get_dataset():
    all_X_ex = np.load("dataset/train_X.npy")
    all_file = np.load("dataset/train_file.npy")
    all_y = np.load("dataset/train_y.npy")
    
    train_X_ex, validation_X_ex, train_y, validation_y, train_file, validation_file =\
    train_test_split(all_X_ex, all_y, all_file, test_size = 0.18, random_state=42)
    
    return train_X_ex, validation_X_ex, train_y, validation_y, train_file, validation_file

def make_model(train_X_ex, learning_rate = 0.00001):
    ip = Input(shape=train_X_ex[0].shape)
    m = Conv2D(64, kernel_size=(4,4), activation='relu')(ip)
    m = MaxPooling2D(pool_size=(4,4))(m)


    m=Flatten()(m)
    m=Dense(32, activation='relu')(m)
    op=Dense(2, activation='softmax')(m)

    model = Model(ip, op)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


def main():
    
    train_X_ex, validation_X_ex, train_y, validation_y, train_file, validation_file = get_dataset()
    model = make_model(train_X_ex)
    
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)

    checkpoint = ModelCheckpoint('model/train_X_ex_best.h5', 
            monitor='val_loss', mode='min', verbose=1, 
            save_best_only=True, save_freq='epoch'
    )
    
    history = model.fit(train_X_ex,
                        train_y,
                        epochs=2,
                        batch_size=32,
                        verbose=1,
                        validation_data=(validation_X_ex, validation_y),
                        callbacks=[checkpoint, earlystopping])
    
    models.save_model(model, 'model/last.h5')
    
        
if __name__ == "__main__":
    main()
    sys.exit()