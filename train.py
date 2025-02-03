from Recognizer.models import define_model
from Recognizer.utils import load_data
from Recognizer.preprocessing import preprocess_image, prep_pixels
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    
    X_train, y_train, X_test, y_test = preprocess_image(X_train, y_train, X_test, y_test)

    X_train, X_test = prep_pixels(X_train, X_test)

    aug = ImageDataGenerator(rotation_range=0.15, zoom_range=0.15, width_shift_range=0.15, height_shift_range=0.15, horizontal_flip = False)

    model = define_model()
    model.fit(aug.flow(trainX, trainY), epochs=10, batch_size=64, verbose = 1)
    model.save('model.h5')