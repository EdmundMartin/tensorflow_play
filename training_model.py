import numpy as np
import os
from random import shuffle
from skimage import color, io
from scipy.misc import imresize
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

TRAIN_DIR = os.path.join(os.getcwd(), 'data/training')
TEST_DIR = os.path.join(os.getcwd(), 'data/testing')
print(TRAIN_DIR, TEST_DIR)
IMG_SIZE = 100
LR = 1e-3

MODEL_NAME = 'sign-yes-not-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    # yes.num.png
    word_label = img.split('.')[-3]
    if word_label == 'yes': return [1, 0]
    elif word_label == 'not': return [0, 1]


def create_train_data():
    training_data = []
    for img in os.listdir(TRAIN_DIR):
        label = label_img(img)
        print(label)
        path = os.path.join(TRAIN_DIR, img)
        img = io.imread(path)
        img = imresize(img, (IMG_SIZE, IMG_SIZE, 3))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        print(img_num)
        img = io.imread(path)
        img = imresize(img, (IMG_SIZE, IMG_SIZE, 3))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#if os.path.exists('{}.meta'.format(MODEL_NAME)):
#    model.load(MODEL_NAME)
#    print('model loaded!')

train = train_data[:-200]
test = train_data[-200:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

test_data = process_test_data()

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # car: [1,0]
    # not car: [0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Not A Car'
    else:
        str_label = 'Car'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()