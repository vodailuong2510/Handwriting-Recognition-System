import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot as plt

if __name__ == '__main__':
    def create_dataset_from_folder(folder, label):
        images = []
        labels = []
        def load_image(filename):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            _, bw_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
            bw_image = cv2.resize(bw_image, (28, 28))
            bw_image = bw_image.reshape(28, 28, 1)
            bw_image = bw_image.astype('float32')/255.0
            return bw_image
        for filename in os.listdir(folder):
            img = load_image(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                labels.append(label)
        return images, labels

    test_dataset = []
    label = []
    num_classes = 10

    contents = os.listdir('.\\data')
    folders = [item for item in contents if os.path.isdir(os.path.join('.\\data', item))]
    for folder in folders:
        root_folder = ".\\data\\" + folder
        for class_index in range(0, num_classes):
            class_folder = os.path.join(root_folder, f"{class_index}")
            images, labels = create_dataset_from_folder(class_folder, class_index)
            test_dataset.extend(images)
            label.extend(labels)

    test_dataset = np.array(test_dataset)
    label = to_categorical(label)
    model = load_model(r'.\final_model2.h5')
    model.evaluate(test_dataset, label)

    predicted_labels = np.argmax(model.predict(test_dataset), axis = 1)
    plt.figure(figsize=(28, 28))
    for i in range(0, 60):
        plt.subplot(20, 3, i + 1)
        plt.imshow(test_dataset[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {np.argmax(label[i])}, predict: {predicted_labels[i]}")
        plt.axis('off')
    plt.show()