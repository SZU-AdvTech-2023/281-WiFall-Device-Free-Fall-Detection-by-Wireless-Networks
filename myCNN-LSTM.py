import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


def load_images(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                # Assuming you use a function to load and preprocess images
                image = load_and_preprocess_image(image_path)
                images.append(image)
                labels.append(int(label))  # Assuming folder names are the labels
    return np.array(images), np.array(labels)


def load_and_preprocess_image(image_path):
    # Implement your image loading and preprocessing logic here
    # You might want to use a library like OpenCV or PIL
    # Example (using PIL):
    # from PIL import Image
    # img = Image.open(image_path)
    # img = img.resize((300, 52))  # Resize if needed
    # img = np.array(img) / 255.0  # Normalize pixel values
    # return img
    pass


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(LSTM(100))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    pic_folder = 'path_to_your_pic_folder'

    # Load and preprocess images
    images, labels = load_images(pic_folder)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, to_categorical(labels), test_size=0.2, random_state=42)

    # Build the model
    input_shape = (300, 52, 3)  # Adjust dimensions based on your image properties
    model = build_model(input_shape)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Plot loss during training
    plt.ylim((0, 1))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN and LSTM loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('CNN_LSTM_loss.png', dpi=300)
    plt.show()

    # Save model weights and plot model structure
    model.save_weights('model_weights.h5')
    plot_model(model, to_file='CNN_LSTM_model.png', show_shapes=True)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\nTest Loss:', loss)
    print('Test Accuracy:', accuracy)

    # Make predictions and compute confusion matrix
    y_prediction = model.predict(X_test)
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_prediction, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    print('Confusion Matrix:')
    print(cm)


if __name__ == '__main__':
    main()
