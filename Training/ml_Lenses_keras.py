"""This is a draft of machine learning code, so that we can test how to do the machine learning algorithm of the
gravitational lenses. """
# IMPORTS
from cnnUtils import getPositiveSimulatedTrain, getNegativeDESTrain, getPositiveSimulatedTest, getNegativeDESTest, \
    makeTrainTest, useKerasModel, plotModel, visualizeKeras, testUnseenDES2017, testUnseenJacobs, \
    testUnseenDES2017AndJacobs, getKerasKFold
from tensorflow.python.keras import backend as K
import tensorflow.python.keras as keras

print("PAST IMPORTS")

# _________________________________________________________________________________________________________________________
# MAIN

# Get training data
positive_train = getPositiveSimulatedTrain()
negative_train = getNegativeDESTrain()

# Get Testing data
positive_test = getPositiveSimulatedTest()
negative_test = getNegativeDESTest()

# make Train Test
x_train, \
x_test, \
y_train, \
y_test, \
train_percent, \
test_percent, \
image_train_std, \
image_train_mean, \
image_train_shape, \
image_labels_shape, \
x_train_shape, \
x_test_shape, \
y_train_shape, \
y_test_shape = makeTrainTest(positive_train, negative_train, positive_test, negative_test)

# Deal with format issues between different backends.  Some put the # of channels in the image before the width and
# height of image.
img_rows = 100
img_cols = 100

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

print("TRAIN X SHAPE: " + str(x_train.shape))
print("TEST X SHAPE: " + str(x_test.shape))
print("x_train original type: " + str(type(x_train[0])))

print("x_test original type: " + str(type(x_test[0])))
#   Type convert and scale the test and training data
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print("x_train processed type: " + str(type(x_train[0])))
print("x_test processed type: " + str(type(x_test[0])))

# print(y_train[:4])
# convert class vectors to binary class matrices.  One-hot encoding
#  3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
# num_classes = 1
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# print("Y TRAIN SHAPE: " + str(y_train.shape))
# print("Y TEST SHAPE: " +str(y_test.shape))
# print(y_train[:4])   # verify one-hot encoding

# use the Keras model
seq_model, model, accuracy_score = useKerasModel(x_train, x_test, y_train, y_test, input_shape)
# seq_model, model, accuracy_score = useKerasModel(x_train, x_test, y_train, y_test)
plotModel(seq_model)

print("DONE")

visualizeKeras(model, input_shape)

# ______________________________________________________________________________________________________________________
n_splits, random_state, k_fold_accuracy, k_fold_std, neural_network = getKerasKFold(x_train,
                                                                                    x_test,
                                                                                    y_train,
                                                                                    y_test,
                                                                                    input_shape)
#
# # calculating the amount of things accurately identified
# # looking at Known131
# # 1 = gravitational lens
# # 0 = negative lens
#
# #
# #_____________________________________________________________________________________________________________________
known_des_2017, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47 = testUnseenDES2017(model,
                                                                                         neural_network,
                                                                                         n_splits,
                                                                                         input_shape)
known_jacobs, accuracy_score_84, k_fold_accuracy_84, k_fold_std_84 = testUnseenJacobs(model,
                                                                                      neural_network,
                                                                                      n_splits,
                                                                                      input_shape)
accuracy_score_131, k_fold_accuracy_131, k_fold_std_131 = testUnseenDES2017AndJacobs(known_des_2017,
                                                                                     known_jacobs,
                                                                                     model,
                                                                                     neural_network,
                                                                                     n_splits,
                                                                                     input_shape)

# write to ml_Lenses_results.xlsx
# makeExcelTable.makeInitialTable()
# element_list = makeExcelTable.getElementList(description, image_train_std, image_train_mean, image_train_shape,
# image_labels_shape, train_percent, test_percent, x_train_shape, x_test_shape, y_train_shape, y_test_shape, n_splits,
# random_state, accuracy_score, k_fold_accuracy, k_fold_std, accuracy_score_47, k_fold_accuracy_47, k_fold_std_47,
# accuracy_score_84, k_fold_accuracy_84, k_fold_std_84, accuracy_score_131, k_fold_accuracy_131, k_fold_std_131)
# file_name = '../Results/ml_Lenses_results.csv' makeExcelTable.appendRowAsList(file_name, element_list)
