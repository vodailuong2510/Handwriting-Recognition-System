from tensorflow.keras.utils import to_categorical

def preprocess_data(trainX, trainy, testX, testy):
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    
    return trainX, trainy, testX, testy

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    
    return train_norm, test_norm