from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import os

def get_data():
    (trainX, trainy), (testX, testy) = mnist.load_data()

    return (trainX, trainy), (testX, testy)

def show_images(images, num_images=5):
    for i in range(num_images):
        plt.subplot(num_images//5 + 1, 1 + 1)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
    plt.show()

if __name__ == '__main__':
    (trainX, trainy), (testX, testy) = get_data()
    show_images(trainX)