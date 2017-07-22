import csv
import cv2
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# The correction factor for the left and the right images.
correction = 0.3

# Setting the batch size
batch_size = 30

# Shuffling the dataset
lines = shuffle(lines)

# Generator for generating the images batchwise.
def myGenerator(lines_model, batch_size):
	num_samples = len(lines_model)
	X_return = []
	y_return = []

	while True:
		for offset in range(0, num_samples, batch_size):
			images = []
			measurements_steering = []
			temp_line = lines_model[offset:offset+int(batch_size/6)]

			for index in range(0, len(temp_line)):
				source_path_c = temp_line[index][0]
				source_path_l = temp_line[index][1]
				source_path_r = temp_line[index][2]
				filename_c = source_path_c.split('\\')[-1]
				filename_l = source_path_l.split('\\')[-1]
				filename_r = source_path_r.split('\\')[-1]
				current_path_c = './data/IMG/' + filename_c
				current_path_l = './data/IMG/' + filename_l
				current_path_r = './data/IMG/' + filename_r

				# Load the center image
				img = Image.open(current_path_c)
				img.load()
				image_c = np.asarray(img)
				images.append(image_c)
				steering_c= float(temp_line[index][3])
				measurements_steering.append(steering_c)
				images.append(cv2.flip(image_c,1))
				measurements_steering.append(steering_c*(-1.0))

				# Load the left image
				img = Image.open(current_path_l)
				img.load()
				image_l = np.asarray(img)
				images.append(image_l)
				steering_l = float(temp_line[index][3])
				measurements_steering.append(steering_l+correction)
				images.append(cv2.flip(image_l,1))
				measurements_steering.append((steering_l+correction)*(-1.0))

				# Load the right image
				img = Image.open(current_path_r)
				img.load()
				image_r = np.asarray(img)
				images.append(image_r)
				steering_r = float(temp_line[index][3])
				measurements_steering.append(steering_r-correction)
				images.append(cv2.flip(image_r,1))
				measurements_steering.append((steering_r-correction)*(-1.0))


			X_return = np.asarray(images)
			y_return = np.asarray(measurements_steering)
			

			yield shuffle(X_return, y_return)




from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adamax
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.callbacks import ReduceLROnPlateau

# The model predicting the steering angle.
model = Sequential()

model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, (5,5), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Convolution2D(36, (5,5), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Convolution2D(48, (5,5), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Convolution2D(64, (3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Convolution2D(64, (3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(1164))

model.add(Dropout(0.5))
model.add(Dense(100))

model.add(Dropout(0.5))
model.add(Dense(50))

model.add(Dropout(0.6))
model.add(Dense(1))

# For printing the shapes of layers.
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

adam = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 1, verbose = 1,min_lr = 0.000001)


# Validation and training split
print(len(lines))
split_index = int(len(lines)*0.8)

Train_lines = lines[:split_index]
Valid_lines = lines[split_index+1:]


print(len(Train_lines))
print(len(Valid_lines))

train_gen = myGenerator(Train_lines ,batch_size)
valid_gen = myGenerator(Valid_lines, batch_size)

# Training the model
hist = model.fit_generator(train_gen, int((len(Train_lines)*6)/batch_size),epochs = 5,verbose=1, callbacks=[reduce_lr],validation_data = valid_gen, validation_steps = int((len(Valid_lines)*6)/batch_size))
model.save('model.h5')