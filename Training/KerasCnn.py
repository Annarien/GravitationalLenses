import os
import sys
import numpy
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

# Globals
max_num_training = 1000             # Set to sys.maxsize when running entire data set
max_num_testing = sys.maxsize       # Set to sys.maxsize when running entire data set
max_num_prediction = sys.maxsize    # Set to sys.maxsize when running entire data set
validation_split = 0.1              # A float value between 0 and 1 that determines what percentage of the training
                                    # data is used for validation.
k_fold_num = 5                      # A number between 1 and 10 that determines how many times the k-fold classifier
                                    # is trained.
epochs = 20                         # A number that dictates how many iterations should be run to train the classifier
batch_size = 100                    # The number of items batched together during training.
run_k_fold_validation = False       # Set this to True if you want to run K-Fold validation as well.
image_shape = (100, 100, 3)         # The shape of the images being learned & evaluated.


# Helper methods
def getPositiveImages(images_dir, max_num, input_shape):
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))
        positive_images = numpy.zeros([num_of_images, 3, 100, 100])
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
        negative_images = numpy.zeros([num_of_images, 3, 100, 100])
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
    for root, dirs, _ in os.walk(images_dir):
        num_of_images = min(max_num, len(dirs))

        unseen_images = numpy.zeros([num_of_images, 3, 100, 100])
        index = 0
        for folder in dirs:
            g_img_path = get_pkg_data_filename('%s/g_norm.fits' % (os.path.join(root, folder)))
            r_img_path = get_pkg_data_filename('%s/r_norm.fits' % (os.path.join(root, folder)))
            i_img_path = get_pkg_data_filename('%s/i_norm.fits' % (os.path.join(root, folder)))

            g_data = fits.open(g_img_path)[0].data[0:100, 0:100]
            r_data = fits.open(r_img_path)[0].data[0:100, 0:100]
            i_data = fits.open(i_img_path)[0].data[0:100, 0:100]

            img_data = [g_data, r_data, i_data]
            unseen_images[index] = img_data
            index += 1

            if index >= num_of_images:
                break
        return unseen_images.reshape(num_of_images, input_shape[0], input_shape[1], input_shape[2])


def makeImageSet(positive_images, negative_images=None):
    if negative_images is None:
        negative_images = []
    image_set = []
    label_set = []

    for index in range(0, len(positive_images)):
        image_set.append(positive_images[index])
        label_set.append(1)

    for index in range(0, len(negative_images)):
        image_set.append(negative_images[index])
        label_set.append(0)

    return numpy.array(image_set), numpy.array(label_set)


def buildClassifier(input_shape=(100, 100, 3)):
    # Initialising the CNN
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    classifier.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    classifier.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(0.5))  # antes era 0.25
    # Adding a third convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(0.5))  # antes era 0.25
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=1, activation='softmax'))
    classifier.summary()

    # Compiling the CNN
    classifier.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier


def executeKFoldValidation(data, labels, num_of_epochs, classifier_batch_size, should_run_k_fold):
    if should_run_k_fold:
        neural_network = KerasClassifier(build_fn=buildClassifier,
                                         epochs=num_of_epochs,
                                         batch_size=classifier_batch_size)
        k_fold_scores = cross_val_score(neural_network, data, labels, scoring='accuracy', cv=k_fold_num)
        score_mean = k_fold_scores.mean() * 100
        print("kFold Scores Mean: " + str(score_mean))
        k_fold_std = k_fold_scores.std()
        print("kFold Scores Std: " + str(k_fold_std))


# __________________________________________________________________________
# MAIN


# Get positive training data
train_pos = getPositiveImages('Training/Positive', max_num_training, input_shape=image_shape)

# Get negative training data
train_neg = getNegativeImages('Training/Negative', max_num_training, input_shape=image_shape)

training_data, training_labels = makeImageSet(train_pos, train_neg)

classifier = buildClassifier()

model_checkpoint = ModelCheckpoint(filepath="best_weights.hdf5",
                                   monitor='val_acc',
                                   save_best_only=True)

early_stopping = EarlyStopping(monitor='val_acc', patience=2)

history = classifier.fit(training_data,
                         training_labels,
                         epochs=epochs,
                         steps_per_epoch=100,
                         callbacks=[model_checkpoint, early_stopping],
                         validation_split=validation_split,
                         validation_steps=50,
                         shuffle=True)

classifier.load_weights('best_weights.hdf5')
classifier.save_weights('galaxies_cnn.h5')

# K fold for training data
executeKFoldValidation(training_data, training_labels, epochs, batch_size, run_k_fold_validation)

# Plot run metrics
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Accuracies
train_val_accuracy_figure = plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
train_val_accuracy_figure.savefig('../Results/TrainingValidationAccuracy.png')

# Losses
train_val_loss_figure = plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
train_val_loss_figure.savefig('../Results/TrainingValidationLoss.png')

# Plot original image
img_tensor = getPositiveImages('Training/Positive', 1, input_shape=image_shape)
positive_train_figure = plt.figure()
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)
positive_train_figure.savefig('../Results/PositiveTrainingFigure.png')

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
for layer_name, layer_activation in zip(layer_names, activations):
    number_of_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    number_of_columns = number_of_features // images_per_row
    display_grid = numpy.zeros((size * number_of_columns, images_per_row * size))
    for col in range(number_of_columns):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = numpy.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    activations_figure = plt.figure(figsize=(scale * display_grid.shape[1],
                                             scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
    activations_figure.savefig('../Results/Activation_%s.png' % layer_name)

# Classifier evaluation
test_pos = getPositiveImages('Testing/Positive', max_num_testing, image_shape)
test_neg = getNegativeImages('Testing/Negative', max_num_testing, image_shape)
testing_data, testing_labels = makeImageSet(test_pos, test_neg)
scores = classifier.evaluate(testing_data, testing_labels, batch_size=batch_size)
print("Test loss: %s" % scores[0])
print("Test accuracy: %s" % scores[1])

# Collect & test known 47
correctly_predicted_count_47 = 0
known_47_images = getUnseenData('UnseenData/Known47', max_num_prediction, input_shape=image_shape)
for known_image in known_47_images:
    known_image = known_image.reshape(1, known_image.shape[0], known_image.shape[1], known_image.shape[2])
    # Run prediction on that image
    predicted_class = classifier.predict_classes(known_image, batch_size=10)
    print("Predicted class is: ", predicted_class)
    if predicted_class[0] == 1:
        correctly_predicted_count_47 += 1
print("%s/47 known images correctly predicted" % correctly_predicted_count_47)

# Collect & test known 84
correctly_predicted_count_84 = 0
known_84_images = getUnseenData('UnseenData/Known84', max_num_prediction, input_shape=image_shape)
for known_image in known_84_images:
    known_image = known_image.reshape(1, known_image.shape[0], known_image.shape[1], known_image.shape[2])
    # Run prediction on that image
    predicted_class = classifier.predict_classes(known_image, batch_size=10)
    print("Predicted class is: ", predicted_class)
    if predicted_class[0] == 1:
        correctly_predicted_count_84 += 1
print("%s/84 known images correctly predicted" % correctly_predicted_count_84)

# K-Fold for known 47
known_47_data, known_47_labels = makeImageSet(known_47_images)
executeKFoldValidation(known_47_data, known_47_labels, epochs, batch_size, run_k_fold_validation)

# K-Fold for known 84
known_84_data, known_84_labels = makeImageSet(known_84_images)
executeKFoldValidation(known_84_data, known_84_labels, epochs, batch_size, run_k_fold_validation)
