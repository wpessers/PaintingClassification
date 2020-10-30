import fnmatch
import ntpath
import os
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import transform
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

PAINTINGS_DIR = r'paintings'

TEST_DIR = r'paintings/Test'


def main():
    #Load v1 model by default
    change_model('v1')
    done = False
    while not done:
        print_menu()
        input_command = input().split(' ')
        if input_command[0] == 'exit':
            print('Exiting...')
            done = True
        elif input_command[0] == 'modelinfo':
            show_model_info()
        elif input_command[0] == 'loadmodel':
            change_model(input_command[1])
        elif input_command[0] == 'stats':
            show_classification_report()
        elif input_command[0] == 'accuracy':
            show_accuracy()
        elif input_command[0] == 'modelsummary':
            model_summary()
        else:
            for filename in input_command:
                make_prediction(filename)


def show_model_info():
    print('====================================================================================================')
    print('There are 2 different models, v1 and v2.')
    print('The v1 model uses a 244x244 input size, opposed to 499x499 for v2.')
    print('Therefore the v2 model retains much more information about the original data.')
    print('This might make the v2 model more accurate if we do some hyperparameter optimisation as we did for v1.')
    print('====================================================================================================')


def model_summary():
    model.summary()


def print_menu():
    print('====================================================================================================')
    print('1) Enter \'modelinfo\' to view information about the available models')
    print('2) Enter \'loadmodel modelname\' to choose a model e.g. \'loadmodel v1\' (default is v1)')
    print('3) Enter names of specific paintings, seperated by spaces, you want to predict. e.g. \'painting1 painting2 '
          'painting3\'')
    print('4) Enter \'stats\' for a classification report and confusion matrix for the test dataset')
    print('5) Enter \'accuracy\' for the accuracy on the test dataset')
    print('6) Enter \'modelsummary\' for a summary of the chosen model')
    print('7) Enter \'exit\' to exit')
    print('====================================================================================================')


def change_model(model_name):
    global model_path
    global image_target_size
    global model

    if model_name == 'v1':
        image_target_size = (224, 224)
        model_path = r'v1.h5'
    if model_name == 'v2':
        image_target_size = (499, 499)
        model_path = r'v2.h5'
    model = load_model(model_path)


def show_classification_report():
    print('Generating classification report')
    test_generator = load_test_datagenerator()
    test_res = model.predict_generator(test_generator)
    test_res = np.argmax(test_res, axis=1)
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(test_generator.classes, test_res)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, linewidths=1)
    plt.xlabel('predicted')
    plt.ylabel('Truth')
    plt.show()
    target_names = ['Brueghel', 'Mondriaan', 'Picasso', 'Rubens']
    print(classification_report(test_generator.classes, test_res, target_names=target_names))


def load_test_datagenerator():
    batch_size = 16
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                      target_size=image_target_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False, )
    return test_generator


def show_accuracy():
    print('Calculating accuracy on test dataset')
    test_generator = load_test_datagenerator()
    score, acc = model.evaluate_generator(test_generator)
    print("Test accuracy: ", acc)


def make_prediction(image_name):
    image_path = find_image(image_name)
    if image_path is not None:
        image = load_image(image_path)
        prep_image = preprocess_image(image)
        print('Predicting image: ' + ntpath.basename(image_path))
        y_prob = model.predict(prep_image)
        show_prediction(y_prob)
    else:
        print('Command or image not found')


def show_prediction(y_prob):
    target_names = ['Brueghel', 'Mondriaan', 'Picasso', 'Rubens']

    predicted_class = np.argmax(y_prob, axis=1)[0]
    predicted_name = target_names[predicted_class]
    prediction_certainty = y_prob[0][predicted_class]
    print('Predicted ' + predicted_name + ' with condfidence of: ' + str(round(prediction_certainty * 100, 2)) + '%')


def preprocess_image(image):
    prep_image = np.array(image).astype('float32') / 255
    prep_image = transform.resize(prep_image, (224, 224, 3))
    prep_image = np.expand_dims(prep_image, axis=0)
    return prep_image


def load_image(file_name):
    return Image.open(file_name)


def find_image(image_name):
    for root, dirnames, filenames in os.walk(PAINTINGS_DIR):
        for filename in fnmatch.filter(filenames, '*' + image_name + '*'):
            return os.path.join(root, filename)


if __name__ == '__main__':
    main()
