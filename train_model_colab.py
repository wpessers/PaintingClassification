import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Uncomment this to use google colab + drive
# from google.colab import drive


# ================================================================================

#             This is a Google Colaboratory Notebook
#           Will not run locally without modifications
#   https://colab.research.google.com/drive/1BkGBMs619BALk2ZqFC6H2tWsK6r58t1V

#           To run please:
#           1) add following link to your Google Drive in the main directory:
#               https://drive.google.com/drive/folders/1DraXBdiJJrOoLc3AyUJ-GKdDa53hOFb8?usp=sharing
#           2) open Colaboratory link
#           3) run the code
#           4) authenticate with your Google Drive
# ================================================================================

# To avoid PIL errors
Image.MAX_IMAGE_PIXELS = 10000000000000

# Uncomment this to use google colab + drive
# drive.mount('/content/drive')


def main():
    base_dir = r'/content/drive/My Drive/AI/betterpaintings/betterpaintings/'
    train_dir = base_dir + 'Train'
    validation_dir = base_dir + 'Validation'
    test_dir = base_dir + 'Test'

    print('\n==================================================================')
    print('MAKING THE MODEL.')

    variables_for_classification = 4  # number of classes

    model = models.Sequential()

    # Strong pretrained model as base for the custom model
    model.add(Xception(include_top=False, weights='imagenet', input_shape=(499, 499, 3)))
    # dropout layer to avoid overfitting
    model.add(layers.Dropout(0.35))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(variables_for_classification, activation='softmax'))

    print('COMPILING THE MODEL.')
    # loss function for multiclassification: categorical_crossentropy
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.00001),
                  metrics=['acc'])
    model.summary()

    print('\n==================================================================')
    print('DATA PREPROCESSING.\n')

    # large image size to preserve a lot of the image data
    image_target_size = (499, 499)
    # mini-batch Gradient Descent
    batch_size = 16

    print('USING IMAGEDATAGENERATOR TO CONVERT JPGS TO TENSORS.\n')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    realtest_datagen = ImageDataGenerator(rescale=1. / 255)

    print('\n==================================================================')
    print('DATA AUGMENTATION.\n')

    # Data generator with data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest',
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       brightness_range=(0.1, 0.9))

    # Get images for training from the labelled directories
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=image_target_size,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    # Get images for validation from the labelled directories
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            target_size=image_target_size,
                                                            batch_size=batch_size,
                                                            class_mode='categorical')

    test_generator = realtest_datagen.flow_from_directory(test_dir,
                                                          target_size=image_target_size,
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          shuffle=False, )

    print('\n==================================================================')
    print('START TRAINING.\n')

    # class weights to counteract the unbalanced dataset.
    class_weights = [13, 12, 1, 6]

    # This checkpoint makes sure the model doesn't degrade when the validation accuracy decreases.
    checkpoint = ModelCheckpoint(".h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    # This stops training our model early if it goes 20 (patience) epochs without improvements in validation accuracy.
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

    # This reduces the learning rate by 20% when the validation loss doesn't improve for 3 subsequent epochs.
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=0.000000001, verbose=3)

    # The model gets trained with the above declared class weights and callbacks.
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=50,
                                  epochs=60,
                                  shuffle=True,
                                  validation_data=validation_generator,
                                  validation_steps=25,
                                  callbacks=[checkpoint, early, reducelr],
                                  class_weight=class_weights)

    # getting the test data stats
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # plotting the evolution of training/validation accuracy/loss against the epochs
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    # Plotting confusion matrix of predicted compared to true labels
    test_res = model.predict_generator(test_generator)
    test_res = np.argmax(test_res, axis=1)
    print('Confusion Matrix')
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(test_generator.classes, test_res), annot=True, linewidths=1, fmt='d')
    plt.xlabel('predicted')
    plt.ylabel('Truth')
    plt.show()

    # Show the classification report for our model
    target_names = ['Brueghel', 'Mondriaan', 'Picasso', 'Rubens']
    print('Classification Report')
    print(classification_report(test_generator.classes, test_res, target_names=target_names))
    score, acc = model.evaluate_generator(test_generator)
    print("Test score: ", score)
    print("Test accuracy: ", acc)

    # saving the model to disk
    model.save(r'/content/v1.h5')


if __name__ == "__main__":
    main()
