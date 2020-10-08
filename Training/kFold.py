import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from pip._internal.req.req_file import preprocess
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from ExcelUtils import createExcelSheet, writeToFile
from sklearn.utils import shuffle
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.utils.vis_utils import plot_model
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
print(dt_string)
excel_headers = []
excel_dictionary = []
excel_headers.append("Date and Time")
excel_dictionary.append(dt_string)

# Globals
max_num = 1998  # Set to sys.maxsize when running entire data set
max_num_testing = sys.maxsize  # Set to sys.maxsize when running entire data set
max_num_prediction = sys.maxsize  # Set to sys.maxsize when running entire data set
validation_split = 0.2  # A float value between 0 and 1 that determines what percentage of the training
# data is used for validation.
k_fold_num = 5  # A number between 1 and 10 that determines how many times the k-fold classifier
# is trained.
epochs = 20  # A number that dictates how many iterations should be run to train the classifier
batch_size = 128  # The number of items batched together during training.
run_k_fold_validation = True  # Set this to True if you want to run K-Fold validation as well.
input_shape = (100, 100, 3)  # The shape of the images being learned & evaluated.
augmented_multiple = 2  # This uses data augmentation to generate x-many times as much data as there is on file.
use_augmented_data = True  # Determines whether to use data augmentation or not.
patience_num = 3  # Used in the early stopping to determine how quick/slow to react.
use_early_stopping = True  # Determines whether to use early stopping or not.
use_model_checkpoint = True  # Determines whether the classifiers keeps track of the most accurate iteration of itself.
monitor_early_stopping = 'val_loss'
monitor_model_checkpoint = 'val_acc'
use_shuffle = True


# Helper methods
def getPositiveImages(images_dir, max_num, input_shape):
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        positive_images = np.zeros([num_of_images, 3, 100, 100])
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/%s_g_norm.fits' % (os.path.join(root, folder), folder))
            r_img_path = get_pkg_data_filename('%s/%s_r_norm.fits' % (os.path.join(root, folder), folder))
            i_img_path = get_pkg_data_filename('%s/%s_i_norm.fits' % (os.path.join(root, folder), folder))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = [g_data, r_data, i_data]
            positive_images[index] = img_data
            index += 1

            if index >= num_of_images:
                break
        return positive_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2])


def getNegativeImages(images_dir, max_num, input_shape):
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        negative_images = np.zeros([num_of_images, 3, 100, 100])
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/g_norm.fits' % (os.path.join(root, folder)))
            r_img_path = get_pkg_data_filename('%s/r_norm.fits' % (os.path.join(root, folder)))
            i_img_path = get_pkg_data_filename('%s/i_norm.fits' % (os.path.join(root, folder)))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = [g_data, r_data, i_data]
            negative_images[index] = img_data
            index += 1

            if index >= num_of_images:
                break
        return negative_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2])


def getUnseenData(images_dir, max_num, input_shape):
    des_tiles = {}

    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/g_norm.fits' % (os.path.join(root, folder)))
            r_img_path = get_pkg_data_filename('%s/r_norm.fits' % (os.path.join(root, folder)))
            i_img_path = get_pkg_data_filename('%s/i_norm.fits' % (os.path.join(root, folder)))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = np.array([g_data, r_data, i_data]).reshape(input_shape[0], input_shape[1], input_shape[2])
            des_tiles.update({folder: img_data})
            index += 1
            if index >= num_of_images:
                break

        return des_tiles


def makeImageSet(positive_images, negative_images=None, known_des_names=None, neg_des_names=None,
                 shuffle_needed=use_shuffle):
    if negative_images is None:
        negative_images = []
        known_des_names = {}
        neg_des_names = {}

    image_set = []
    label_set = []
    des_names_set = []

    # If there is none in objects for the known_des_names and neg_des_names
    if known_des_names is None and neg_des_names is None:
        for index_none in range(0, len(positive_images)):
            image_set.append(positive_images[index_none])
            label_set.append(1)

        for index_none in range(0, len(negative_images)):
            image_set.append(negative_images[index_none])
            label_set.append(0)

        if shuffle_needed:
            image_set, label_set = shuffle(image_set, label_set)

    else:  # if there is names for des
        for index_des in range(0, len(positive_images)):
            image_set.append(positive_images[index_des])
            label_set.append(1)
            des_names_set.append(known_des_names[index_des])

        for index_des in range(0, len(negative_images)):
            image_set.append(negative_images[index_des])
            label_set.append(0)
            des_names_set.append(neg_des_names[index_des])

        if shuffle_needed:
            image_set, label_set, des_names_set = shuffle(image_set, label_set, des_names_set)

    return np.array(image_set), np.array(label_set), np.array(des_names_set)


def buildClassifier(input_shape=(100, 100, 3)):
    # Initialising the CNN
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    classifier.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    classifier.add(Dropout(0.5))  # added extra Dropout layer
    classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    classifier.add(Dropout(0.5))  # added extra dropout layer
    classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(0.2))  # antes era 0.25
    classifier.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dense(units=1024, activation='relu'))  # added new dense layer
    classifier.add(Dropout(0.2))  # antes era 0.25
    # Step 3 - Flattening
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='relu'))  # added new dense layer
    classifier.add(Dense(units=256, activation='relu'))  # added new dense layer
    # Step 4 - Full connection
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.summary()

    # Compiling the CNN
    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return classifier


def visualiseActivations(img_tensor, base_dir):
    global predicted_class, size
    # Run prediction on that image
    predicted_class = classifier.predict_classes(img_tensor, batch_size=10)
    print("Predicted class is: ", predicted_class)
    # Visualize activations
    layer_outputs = [layer.output for layer in classifier.layers[:12]]
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    layer_names = []
    for layer in classifier.layers[:12]:
        layer_names.append(layer.name)
    images_per_row = 3
    count = 0
    for layer_name, layer_activation in zip(layer_names, activations):
        number_of_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        number_of_columns = number_of_features // images_per_row
        display_grid = np.zeros((size * number_of_columns, images_per_row * size))
        for col in range(number_of_columns):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        activations_figure = plt.figure(figsize=(scale * display_grid.shape[1],
                                                 scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        activations_figure.savefig('%s/%s_Activation_%s.png' % (base_dir, count, layer_name))
        plt.close()

        count += 1


def usingModelsWithOrWithoutAugmentedData(training_data, training_labels, val_data, val_labels):
    model_checkpoint = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor=monitor_model_checkpoint,
                                       save_best_only=True)

    early_stopping = EarlyStopping(monitor=monitor_early_stopping, patience=patience_num)  # original patience =3

    classifier = buildClassifier()
    callbacks_array = []
    if use_early_stopping:
        callbacks_array.append(early_stopping)
    if use_model_checkpoint:
        callbacks_array.append(model_checkpoint)

    print(len(training_data))
    history = classifier.fit(training_data,
                             training_labels,
                             epochs=epochs,
                             validation_data=(val_data, val_labels),
                             callbacks=callbacks_array,
                             batch_size=batch_size
                             # steps_per_epoch=int(len(training_data) / batch_size),
                             )
    return history, classifier


def createAugmentedData(training_data, training_labels):
    complete_training_data_set = []
    complete_training_labels_set = []

    for data in training_data:
        complete_training_data_set.append(data)
    print("Complete Training Data: " + str(len(complete_training_data_set)))

    for label in training_labels:
        complete_training_labels_set.append(label)
    print("Complete Training Label: " + str(len(complete_training_labels_set)))

    # create augmented data
    data_augmented = ImageDataGenerator(featurewise_center=True,
                                        featurewise_std_normalization=True,
                                        rotation_range=90,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True)
    data_augmented.fit(training_data)

    training_data_size = training_data.shape[0]
    aug_counter = 0
    while aug_counter < (augmented_multiple - 1):
        iterator = data_augmented.flow(training_data, training_labels, batch_size=training_data_size)
        # iterator = data_augmented.flow(training_data, training_labels, batch_size=batch_size)
        augmented_data = iterator.next()
        for data in augmented_data[0]:
            complete_training_data_set.append(data)
        for label in augmented_data[1]:
            complete_training_labels_set.append(label)
        aug_counter += 1

    print("Size of All Training Data: " + str(len(complete_training_data_set)))
    print("Size of All Training Labels: " + str(len(complete_training_labels_set)))

    array_training_data = np.array(complete_training_data_set)
    array_training_labels = np.array(complete_training_labels_set)

    print("Shape of complete training data: " + str(array_training_data.shape))
    print("Shape of complete training labels: " + str(array_training_labels.shape))

    return np.array(complete_training_data_set), np.array(complete_training_labels_set)


def savePredictedLenses(des_names_array, predicted_class_probabilities, predicted_lenses_filepath, text_file_path):
    if not os.path.exists(predicted_lenses_filepath):
        os.mkdir('%s/' % predicted_lenses_filepath)
    text_file = open('%s' % text_file_path, "a+")
    text_file.write('Predicted Lenses: \n')
    for lens_index in range(len(predicted_class_probabilities)):
        if predicted_class_probabilities[lens_index] == 1:
            text_file.write("%s \n " % des_names_array[lens_index])

    text_file.write('\n')
    text_file.write('\n')

    text_file.write('No Lenses Predicted: \n')
    for lens_index in range(len(predicted_class_probabilities)):
        if predicted_class_probabilities[lens_index] == 0:
            text_file.write("%s \n " % des_names_array[lens_index])
    text_file.close()


def gettingTrueFalsePositiveNegatives(testing_data, testing_labels, text_file_path,
                                      predicted_lenses_filepath):
    if not os.path.exists(predicted_lenses_filepath):
        os.mkdir('%s/' % predicted_lenses_filepath)

    predicted_data = classifier.predict_classes(testing_data)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(testing_labels,
                                                                                    predicted_data.round()).ravel()
    matrix = (confusion_matrix(testing_labels, predicted_data.round()))
    print(str(matrix) + ' \n ')
    print("True Positive: %s \n" % true_positive)
    print("False Negative: %s \n" % false_negative)
    print("False Positive: %s \n" % false_positive)
    print("True Negative: %s \n" % true_negative)

    text_file = open('%s' % text_file_path, "w")
    text_file.write('Predicted vs True Matrix: \n')
    text_file.write(str(matrix) + " \n ")
    text_file.write("True Negative: %s \n" % str(true_negative))
    text_file.write("False Positive: %s \n" % str(false_positive))
    text_file.write("False Negative: %s \n" % str(false_negative))
    text_file.write("True Positive: %s \n" % str(true_positive))
    text_file.write("\n")
    text_file.close()


def executeKFoldValidation(train_data, train_labels, val_data, val_labels, test_data, test_labels,
                           images_47, labels_47, images_84, labels_84, all_unseen_images, all_unseen_labels):
    if run_k_fold_validation:
        print("In executingKFoldValidation")

        # this is doing it manually:
        kfold = StratifiedKFold(n_splits=k_fold_num, shuffle=True)

        test_scores_list = []
        unseen_47_scores_list = []
        unseen_84_scores_list = []
        all_unseen_scores_list = []

        for train, test in kfold.split(train_data, train_labels):
            # make the model
            model = buildClassifier()
            # fit the model
            model.fit(train_data[train],
                      train_labels[train],
                      epochs=epochs,
                      validation_data=(val_data, val_labels),
                      batch_size=batch_size
                      )

            unseen_47_scores = model.evaluate(images_47, labels_47, batch_size=batch_size)
            unseen_47_scores_list.append(unseen_47_scores[1] * 100)
            unseen_84_scores = model.evaluate(images_84, labels_84, batch_size=batch_size)
            unseen_84_scores_list.append(unseen_84_scores[1] * 100)
            test_scores = model.evaluate(test_data, test_labels, batch_size=batch_size)
            test_scores_list.append(test_scores[1] * 100)
            all_unseen_score = model.evaluate(all_unseen_images, all_unseen_labels, batch_size=batch_size)
            all_unseen_scores_list.append(all_unseen_score[1] * 100)

        test_scores_mean = np.mean(test_scores_list)
        test_scores_std = np.std(test_scores_list)
        unseen_47_mean = np.mean(unseen_47_scores_list)
        unseen_47_std = np.std(unseen_47_scores_list)
        unseen_84_mean = np.mean(unseen_84_scores_list)
        unseen_84_std = np.std(unseen_84_scores_list)
        all_unseen_mean = np.mean(all_unseen_scores_list)
        all_unseen_std = np.std(all_unseen_scores_list)

        print("Test Scores: " + str(test_scores_list))
        print("Test Scores Mean: " + str(test_scores_mean))
        print("Test Scores Std: " + str(test_scores_std))
        print("Unseen 47 Scores: " + str(unseen_47_scores_list))
        print("Unseen 47 Scores Mean: " + str(unseen_47_mean))
        print("Unseen 47 Scores Std: " + str(unseen_47_std))
        print("Unseen 84 Scores: " + str(unseen_84_scores_list))
        print("Unseen 84 Scores Mean: " + str(unseen_84_mean))
        print("Unseen 84 Scores Std: " + str(unseen_84_std))
        print("All Unseen Scores: " + str(all_unseen_scores_list))
        print("All Unseen Scores Mean: " + str(all_unseen_mean))
        print("All Unseen Scores Std: " + str(all_unseen_std))

# __________________________________________________________________________
# MAIN

# Get positive training data
train_pos = getPositiveImages('Training/PositiveAll', max_num, input_shape=input_shape)
print("Train Positive Shape: " + str(train_pos.shape))

# Get negative training data
train_neg = getNegativeImages('Training/Negative', max_num, input_shape=input_shape)
print("Train Negative Shape: " + str(train_neg.shape))

all_training_data, all_training_labels, _ = makeImageSet(train_pos, train_neg, shuffle_needed=use_shuffle)
if use_augmented_data:
    all_training_data, all_training_labels = createAugmentedData(all_training_data, all_training_labels)

training_data, val_data, training_labels, val_labels = train_test_split(all_training_data,
                                                                        all_training_labels,
                                                                        test_size=validation_split,
                                                                        shuffle=True)
# Classifier evaluation
test_pos = getPositiveImages('Testing/PositiveAll', max_num_testing, input_shape)
test_neg = getNegativeImages('Testing/Negative', max_num_testing, input_shape)
testing_data, testing_labels, _ = makeImageSet(test_pos, test_neg, shuffle_needed=True)
print("Testing Data Shape:  " + str(testing_data.shape))
print("Testing Labels Shape: " + str(testing_labels.shape))
print("Got Unseen Testing data")

# Evaluate known 47 with negative 47
known_47_images = getUnseenData('UnseenData/Known47', max_num_prediction, input_shape=input_shape)
negative_47_images = getUnseenData('UnseenData/Negative', 47, input_shape=input_shape)
images_47, labels_47, des_47_names = makeImageSet(list(known_47_images.values()),
                                                  list(negative_47_images.values()),
                                                  list(known_47_images.keys()),
                                                  list(negative_47_images.keys()),
                                                  shuffle_needed=True)
print("47 Data Shape:  " + str(images_47.shape))
print("47 Labels Shape: " + str(labels_47.shape))
print("Got Unseen 47 data")

# Evaluate known 84 with negative 84
known_84_images = getUnseenData('UnseenData/Known84', max_num_prediction, input_shape=input_shape)
negative_84_images = getUnseenData('UnseenData/Negative', 84, input_shape=input_shape)
images_84, labels_84, des_84_names = makeImageSet(list(known_84_images.values()),
                                                  list(negative_84_images.values()),
                                                  list(known_84_images.keys()),
                                                  list(negative_84_images.keys()),
                                                  shuffle_needed=True)
print("84 Data Shape:  " + str(images_84.shape))
print("84 Labels Shape: " + str(labels_84.shape))
print("Got Unseen 84 data")

all_unseen_images = np.concatenate((images_47, images_84))
all_unseen_labels = np.concatenate((labels_47, labels_84))
all_des_names = np.concatenate((des_47_names, des_84_names))
print("All Data Shape: " + str(all_unseen_images.shape))
print("All Data Labels: " + str(all_unseen_labels.shape))

# K fold for training data
executeKFoldValidation(training_data, training_labels, val_data, val_labels, testing_data, testing_labels,
                       images_47, labels_47, images_84, labels_84, all_unseen_images, all_unseen_labels)

# print("Test loss of normal CNN: %s" % scores[0])
# print("Test accuracy of normal CNN: %s" % scores[1])

# add row to excel table
# createExcelSheet('../Results/kerasCNN_Results.csv', excel_headers)
# writeToFile('../Results/kerasCNN_Results.csv', excel_dictionary)
