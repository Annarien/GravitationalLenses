import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from pip._internal.req.req_file import preprocess
from sklearn.model_selection import cross_val_score, train_test_split
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

# Globals
excel_headers = []
excel_dictionary = []
max_num = 10  # Set to sys.maxsize when running entire data set
max_num_testing = 10
# max_num_testing = sys.maxsize  # Set to sys.maxsize when running entire data set
max_num_prediction = sys.maxsize  # Set to sys.maxsize when running entire data set
validation_split = 0.2  # A float value between 0 and 1 that determines what percentage of the training
# data is used for validation.
k_fold_num = 5  # A number between 1 and 10 that determines how many times the k-fold classifier
# is trained.
epochs = 20  # A number that dictates how many iterations should be run to train the classifier
batch_size = 10  # The number of items batched together during training.
run_k_fold_validation = False  # Set this to True if you want to run K-Fold validation as well.
input_shape = (100, 100, 3)  # The shape of the images being learned & evaluated.
augmented_multiplier = 2
use_augmented_data = True
patience_num = 3
use_early_stopping = True
use_model_checkpoint = True
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
print("date and time: ", dt_string)
excel_headers.append("Date and Time")
excel_dictionary.append(dt_string)

if not os.path.exists('../Results/%s/' % dt_string):
    os.mkdir('../Results/%s/' % dt_string)


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
        return positive_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2]), num_of_images


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


def makeImageSet(positive_images, negative_images=None, known_des_names=None, neg_des_names=None, shuffle_needed=False):
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
    # Adding a third convolutional layer
    classifier.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(0.2))  # antes era 0.25
    # Step 3 - Flattening
    classifier.add(Flatten())
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


def executeKFoldValidation(data,
                           labels,
                           excel_headers,
                           excel_dictionary):
    # global k_fold_std
    num_of_epochs = epochs
    classifier_batch_size = batch_size

    if run_k_fold_validation:
        neural_network = KerasClassifier(build_fn=buildClassifier,
                                         epochs=num_of_epochs,
                                         batch_size=classifier_batch_size)
        k_fold_scores = cross_val_score(neural_network, data, labels, scoring='accuracy', cv=k_fold_num)
        score_mean = k_fold_scores.mean() * 100
        print("kFold Scores Mean: " + str(score_mean))
        k_fold_std = k_fold_scores.std()
        print("kFold Scores Std: " + str(k_fold_std))

        excel_headers.append("K-Fold_Mean")
        excel_dictionary.append(score_mean)
        excel_headers.append("K-Fold_Std")
        excel_dictionary.append(k_fold_std)


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


def usingModelsWithOrWithoutAugmentedData(training_data, training_labels, val_data, val_labels, num_of_images):
    model_checkpoint = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor='val_acc',
                                       save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_num)  # original patience =3

    classifier = buildClassifier()
    callbacks_array = []
    if use_early_stopping:
        callbacks_array.append(early_stopping)
    if use_model_checkpoint:
        callbacks_array.append(model_checkpoint)

    if use_augmented_data:
        training_data, training_labels = createAugmentedData(training_data, training_labels, num_of_images)

    history = classifier.fit(training_data,
                             training_labels,
                             epochs=epochs,
                             # batch_size=batch_size,
                             validation_data=(val_data, val_labels),
                             callbacks=callbacks_array,
                             steps_per_epoch=len(training_data) / batch_size)
    return history, classifier


def createAugmentedData(training_data, training_labels, num_of_images):
    # create a copy of training data and training labels
    original_train_data = np.copy(training_data)
    original_train_labels = np.copy(training_labels)

    aug_counter = 0
    complete_training_data_set = original_train_data
    print("Complete Training Data: " +str(complete_training_data_set.shape))
    complete_training_labels_set = original_train_labels
    print("Complete Training Label: " +str(complete_training_labels_set.shape))

    # create augmented data
    data_augmented = ImageDataGenerator(featurewise_center=True,
                                        featurewise_std_normalization=True,
                                        rotation_range=90,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True)

    # create a multiple of augmented data, here 2 sets of augmented data
    while aug_counter < (augmented_multiplier - 1):
        data_augmented.fit(training_data)
        iterator = data_augmented.flow(training_data, training_labels, batch_size=batch_size)

        print(len(iterator))
        for image, label in iterator:

            reshaped_image = image.reshape(image.shape[0], input_shape[0], input_shape[1], input_shape[2])
            # print("reshaped: "+str(reshaped_image.shape))
            print("label.shape[0]: "+str(label.shape[0]))
            reshaped_label = label.reshape(label.shape[0])
            # new_reshaped_label = np.array([reshaped_label])
            print("reshaped label : " + str(reshaped_label.shape))

            print("Complete Training Data Set: " + str(complete_training_data_set.shape))
            print("Reshaped Image: " + str(reshaped_image.shape))
            complete_training_data_set = np.vstack((complete_training_data_set, reshaped_image))

            print("Complete Training Labels Set: " + str(complete_training_labels_set.shape))
            print("Reshaped Label: "+ str(reshaped_label.shape))

            # complete_training_labels_set = np.vstack((complete_training_labels_set, reshaped_label))
            complete_training_labels_set = np.concatenate((complete_training_labels_set, reshaped_label))
            print("Complete Training Labels Set: " + str(complete_training_labels_set.shape))

            # complete_training_labels_set.append(reshaped_label)

        aug_counter += 1

        # complete_training_data_set = np.vstack((complete_training_data_set, reshaped_image))
        # complete_training_labels_set = np.vstack((complete_training_labels_set, label))

    complete_training_data_set = np.vstack((complete_training_data_set, original_train_data))
    complete_training_labels_set = np.vstack((complete_training_labels_set, original_train_labels))

    print("Size of All Training Data: " + str(complete_training_data_set.shape))
    print("Size of All Training Labels: " + str(complete_training_labels_set.shape))

    return complete_training_data_set, complete_training_labels_set


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


# __________________________________________________________________________
# MAIN


# Get positive training data
train_pos, num_of_images = getPositiveImages('Training/Positive3000', max_num, input_shape=input_shape)
print("Train Positive Shape: " + str(train_pos.shape))
excel_headers.append("Train_Positive_Shape")
excel_dictionary.append(train_pos.shape)

# Get negative training data
train_neg = getNegativeImages('Training/Negative', max_num, input_shape=input_shape)
print("Train Negative Shape: " + str(train_neg.shape))
excel_headers.append("Train_Negative_Shape")
excel_dictionary.append(train_neg.shape)

all_training_data, all_training_labels, _ = makeImageSet(train_pos, train_neg)
training_data, val_data, training_labels, val_labels = train_test_split(all_training_data,
                                                                        all_training_labels,
                                                                        test_size=validation_split,
                                                                        shuffle=True)
excel_headers.append("All_Training_Data_Shape")
excel_dictionary.append(all_training_labels.shape)
excel_headers.append("All_Training_Labels_Shape")
excel_dictionary.append(all_training_labels.shape)
excel_headers.append("Training_Data_Shape")
excel_dictionary.append(training_data.shape)
excel_headers.append("Validation_Data_Shape")
excel_dictionary.append(val_data.shape)
excel_headers.append("Training_Labels_Shape")
excel_dictionary.append(training_labels.shape)
excel_headers.append("Validation_Labels_Shape")
excel_dictionary.append(val_labels.shape)
excel_headers.append("Validation_Split")
excel_dictionary.append(validation_split)

history, classifier = usingModelsWithOrWithoutAugmentedData(training_data,
                                                            training_labels,
                                                            val_data,
                                                            val_labels,
                                                            num_of_images)

classifier.load_weights('best_weights.hdf5')
classifier.save_weights('galaxies_cnn.h5')

excel_headers.append("Epochs")
excel_dictionary.append(epochs)
excel_headers.append("Batch_size")
excel_dictionary.append(batch_size)

# K fold for training data
executeKFoldValidation(training_data,
                       training_labels,
                       excel_headers,
                       excel_dictionary)

# Plot run metrics
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
number_of_completed_epochs = range(1, len(acc) + 1)

# Accuracies
train_val_accuracy_figure = plt.figure()
plt.plot(number_of_completed_epochs, acc, label='Training acc')
plt.plot(number_of_completed_epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
# plt.show()
train_val_accuracy_figure.savefig('../Results/%s/TrainingValidationAccuracy.png' % dt_string)
plt.close()

# Losses
train_val_loss_figure = plt.figure()
plt.plot(number_of_completed_epochs, loss, label='Training loss')
plt.plot(number_of_completed_epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# plt.show()
train_val_loss_figure.savefig('../Results/%s/TrainingValidationLoss.png' % dt_string)
plt.close()

# make positive and negative directory
if not os.path.exists('../Results/%s/PositiveResults/' % dt_string):
    os.mkdir('../Results/%s/PositiveResults/' % dt_string)

if not os.path.exists('../Results/%s/NegativeResults/' % dt_string):
    os.mkdir('../Results/%s/NegativeResults/' % dt_string)

# Plot original positive image
img_positive_tensor = getPositiveImages('Training/Positive3000', 1, input_shape=input_shape)
positive_train_figure = plt.figure()
plt.imshow(img_positive_tensor[0])
# plt.show()
print(img_positive_tensor.shape)
positive_train_figure.savefig('../Results/%s/PositiveResults/PositiveTrainingFigure.png' % dt_string)
plt.close()

# Visualise Activations of positive image
visualiseActivations(img_positive_tensor, base_dir='../Results/%s/PositiveResults/' % dt_string)

# Plot original negative image
img_negative_tensor = getNegativeImages('Training/Negative', 1, input_shape=input_shape)
negative_train_figure = plt.figure()
plt.imshow(img_negative_tensor[0])
# plt.show()
print(img_negative_tensor.shape)
negative_train_figure.savefig('../Results/%s/NegativeResults/NegativeTrainingFigure.png' % dt_string)
plt.close()

# Visualise Activations of negative image
visualiseActivations(img_negative_tensor, base_dir='../Results/%s/NegativeResults/' % dt_string)

# Classifier evaluation
test_pos = getPositiveImages('Testing/Positive3000', max_num_testing, input_shape)
test_neg = getNegativeImages('Testing/Negative', max_num_testing, input_shape)
testing_data, testing_labels, _ = makeImageSet(test_pos, test_neg, shuffle_needed=True)
scores = classifier.evaluate(testing_data, testing_labels, batch_size=batch_size)
print("Test loss: %s" % scores[0])
print("Test accuracy: %s" % scores[1])

excel_headers.append("Test_Loss")
excel_dictionary.append(scores[0])
excel_headers.append("Test_Accuracy")
excel_dictionary.append(scores[1])

gettingTrueFalsePositiveNegatives(testing_data,
                                  testing_labels,
                                  text_file_path='../Results/%s/TrainingTestingResults/ActualPredictedMatrix.txt' % dt_string,
                                  predicted_lenses_filepath='../Results/%s/TrainingTestingResults' % dt_string)

# Evaluate known 47 with negative 47
known_47_images = getUnseenData('UnseenData/Known47', max_num_prediction, input_shape=input_shape)
negative_47_images = getUnseenData('UnseenData/Negative', 47, input_shape=input_shape)
images_47, labels_47, des_47_names = makeImageSet(list(known_47_images.values()),
                                                  list(negative_47_images.values()),
                                                  list(known_47_images.keys()),
                                                  list(negative_47_images.keys()),
                                                  shuffle_needed=True)

predicted_class_probabilities_47 = classifier.predict_classes(images_47, batch_size=batch_size)
lens_predicted_count_47 = np.count_nonzero(predicted_class_probabilities_47 == 1)
non_lens_predicted_count_47 = np.count_nonzero(predicted_class_probabilities_47 == 0)
print("%s/47 known images predicted" % lens_predicted_count_47)
print("%s/47 non lensed images predicted" % non_lens_predicted_count_47)

gettingTrueFalsePositiveNegatives(images_47,
                                  labels_47,
                                  text_file_path='../Results/%s/Predicted47/47_LensesPredicted.txt' % dt_string,
                                  predicted_lenses_filepath='../Results/%s/Predicted47' % dt_string)

savePredictedLenses(des_47_names,
                    predicted_class_probabilities_47,
                    predicted_lenses_filepath='../Results/%s/Predicted47' % dt_string,
                    text_file_path='../Results/%s/Predicted47/47_LensesPredicted.txt' % dt_string)

excel_headers.append("Predicted_Lens_47")
excel_dictionary.append(lens_predicted_count_47)
excel_headers.append("Predicted_No_Lens_47")
excel_dictionary.append(non_lens_predicted_count_47)

# Evaluate known 84 with negative 84
known_84_images = getUnseenData('UnseenData/Known84', max_num_prediction, input_shape=input_shape)
negative_84_images = getUnseenData('UnseenData/Negative', 84, input_shape=input_shape)
images_84, labels_84, des_84_names = makeImageSet(list(known_84_images.values()),
                                                  list(negative_84_images.values()),
                                                  list(known_84_images.keys()),
                                                  list(negative_84_images.keys()),
                                                  shuffle_needed=True)

predicted_class_probabilities_84 = classifier.predict_classes(images_84, batch_size=batch_size)
lens_predicted_count_84 = np.count_nonzero(predicted_class_probabilities_84 == 1)
non_lens_predicted_count_84 = np.count_nonzero(predicted_class_probabilities_84 == 0)
print("%s/84 known images predicted" % lens_predicted_count_84)
print("%s/84 non lensed images predicted" % non_lens_predicted_count_84)

gettingTrueFalsePositiveNegatives(images_84,
                                  labels_84,
                                  text_file_path='../Results/%s/Predicted84/84_LensesPredicted.txt' % dt_string,
                                  predicted_lenses_filepath='../Results/%s/Predicted84' % dt_string)

savePredictedLenses(des_84_names,
                    predicted_class_probabilities_84,
                    predicted_lenses_filepath='../Results/%s/Predicted84' % dt_string,
                    text_file_path='../Results/%s/Predicted84/84_LensesPredicted.txt' % dt_string)

excel_headers.append("Predicted_Lens_84")
excel_dictionary.append(lens_predicted_count_84)
excel_headers.append("Predicted_No_Lens_84")
excel_dictionary.append(non_lens_predicted_count_84)

# K-Fold for known 47
executeKFoldValidation(images_47,
                       labels_47,
                       excel_headers,
                       excel_dictionary)

# K-Fold for known 84
executeKFoldValidation(images_84,
                       labels_84,
                       excel_headers,
                       excel_dictionary)

# add row to excel table
createExcelSheet('../Results/kerasCNN_Results.csv', excel_headers)
writeToFile('../Results/kerasCNN_Results.csv', excel_dictionary)
