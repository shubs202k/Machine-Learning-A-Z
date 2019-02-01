# Convolution Neural Networks

# Part 1 ====== Building the CNN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # Initialize Neural Network
from keras.layers import Conv2D # Add Convolution Layer
from keras.layers import MaxPooling2D # Add Pooling layer
from keras.layers import Flatten # Flattening
from keras.layers import Dense  # To Add Fully connected layer

# Initialize the CNN
classifier = Sequential() # Object of class Sequential

# Step 1 ==== Convolution
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), 
                             activation = 'relu'))
# Create 32 Feature Detectors/Filters of size 3*3 
# Input images have 3 channels and size 64*64
# Activation Function to remove any linearity

# Step 2 ==== Pooling on each Feature Map to reduce the number of nodes
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer to improve accuracy between 
# training set and test set results
# Input to this second convo layer is the pooled layer 
# No need to give input shape
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


# Step 3 ==== Flattening
classifier.add(Flatten()) # Vector consists spatial information 

# Step 4 ===== Fully connected layer
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid')) # Output Layer

# Compile the CNN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Part2 Fitting the CNN to the images
# Image Augmentation ==== To avoid Overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Creates the Training Set and resizing images in the set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Creates the Test Set and resizing images in the set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

        
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

# Increase target size of images to improve accuracy
        






