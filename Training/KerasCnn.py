import os
import sys
import numpy
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Globals
max_num_training = 1000  # Set to sys.maxsize when running entire data set
max_num_prediction = sys.maxsize  # Set to sys.maxsize when running entire data set
validation_split = 0.1  # A float value between 0 and 1 that determines what percentage of the training data is used
                        # for validation.
image_shape = (100, 100, 3)


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


def makeImageSet(positive_images, negative_images):
    image_set = []
    label_set = []

    for index in range(0, len(positive_images)):
        image_set.append(positive_images[index])
        label_set.append(1)

    for index in range(0, len(negative_images)):
        image_set.append(negative_images[index])
        label_set.append(0)

    return numpy.array(image_set), numpy.array(label_set)


# Get positive training data
train_pos = getPositiveImages('Training/Positive', max_num_training, input_shape=image_shape)

# Get negative training data
train_neg = getNegativeImages('Training/Negative', max_num_training, input_shape=image_shape)

training_data, training_labels = makeImageSet(train_pos, train_neg)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape, padding='same'))
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

model_checkpoint = ModelCheckpoint(filepath="best_weights.hdf5",
                                   monitor='val_acc',
                                   save_best_only=True)

early_stopping = EarlyStopping(monitor='val_acc', patience=2)

history = classifier.fit(training_data,
                         training_labels,
                         epochs=20,
                         steps_per_epoch=100,
                         callbacks=[model_checkpoint, early_stopping],
                         validation_split=validation_split,
                         validation_steps=50,
                         shuffle=True)

classifier.load_weights('best_weights.hdf5')
classifier.save_weights('galaxies_cnn.h5')

# Plot run metrics
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Accuracies
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Losses
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Plot original image
img_tensor = getPositiveImages('Training/Positive', 1, input_shape=image_shape)
plt.figure()
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)

# Run prediction on that image
predicted_class = classifier.predict_classes(img_tensor, batch_size=10)
print("Predicted class is: ", predicted_class)

# Visualize activations
layer_outputs = [layer.output for layer in classifier.layers[:12]]
activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.figure()
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()

# Collect & test known 47
correctly_predicted_count = 0
known_47_images = getUnseenData('UnseenData/Known47', max_num_prediction, input_shape=image_shape)
for known_image in known_47_images:
    known_image = known_image.reshape(1, known_image.shape[0], known_image.shape[1], known_image.shape[2])
    # Run prediction on that image
    predicted_class = classifier.predict_classes(known_image, batch_size=10)
    print("Predicted class is: ", predicted_class)
    if predicted_class[0] == 1:
        correctly_predicted_count += 1
print("%s/47 known images correctly predicted" % correctly_predicted_count)

# Collect & test known 84
correctly_predicted_count = 0
known_84_images = getUnseenData('UnseenData/Known84', max_num_prediction, input_shape=image_shape)
for known_image in known_47_images:
    known_image = known_image.reshape(1, known_image.shape[0], known_image.shape[1], known_image.shape[2])
    # Run prediction on that image
    predicted_class = classifier.predict_classes(known_image, batch_size=10)
    print("Predicted class is: ", predicted_class)
    if predicted_class[0] == 1:
        correctly_predicted_count += 1
print("%s/84 known images correctly predicted" % correctly_predicted_count)
