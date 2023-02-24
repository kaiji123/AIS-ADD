import tensorflow as tf
import cv2
import numpy as np
import mss
import time
sct = mss.mss()
top = 30
left = 0
width =642
height =480
# datasets\\cs\\test\\images\\40.jpg

class CNN:

    def __init__(self) -> None:
        

        # Define the model
        model = tf.keras.Sequential()

        # Add a convolutional layer with 32 filters and a kernel size of 3
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(480, 642, 3)))

        # Add a max pooling layer with a pool size of 2
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

        # Flatten the output of the convolutional layers
        model.add(tf.keras.layers.Flatten())

        # Add a dense layer with 64 units and ReLU activation
        model.add(tf.keras.layers.Dense(64, activation='relu'))

        # Add a final dense layer with 10 units and softmax activation for classification
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile the model with an Adam optimizer and categorical crossentropy loss function
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights('models\\cnx.h5')
        self.model = model



    def detectImg(self,img):
        
        
        image = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
        # image = cv2.imread('datasets\\cs\\test\\images\\39.jpg')
        image = cv2.resize(image, (642,480), interpolation= cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        

        
        cv2.waitKey(1)
        # print(image_array.shape)
        # Add an extra dimension to the image (since Keras expects a batch of images)
        image_array = np.expand_dims(image, axis=0)

        t1 = time.perf_counter()
        # Predict the class of the image
        predictions = self.model.predict(image_array, batch_size=1).round()

        t2 = time.perf_counter()
        elapsed_time = t2 - t1
        
        # Print the elapsed time

        # print(f"Inference took {elapsed_time:.10f} seconds")

        predictions =predictions.squeeze()

        # print(predictions)
        return predictions



class CNNFar:

    def __init__(self) -> None:
        

        # Define the model
        model = tf.keras.Sequential()

        # Add a convolutional layer with 32 filters and a kernel size of 3
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(480, 642, 3)))

        # Add a max pooling layer with a pool size of 2
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

        # Flatten the output of the convolutional layers
        model.add(tf.keras.layers.Flatten())

        # Add a dense layer with 64 units and ReLU activation
        model.add(tf.keras.layers.Dense(64, activation='relu'))

        # Add a final dense layer with 10 units and softmax activation for classification
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile the model with an Adam optimizer and categorical crossentropy loss function
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights('models\\cnn_far.h5')
        self.model = model



    def detectImg(self,img):
        
        
        image = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
        # image = cv2.imread('datasets\\cs\\test\\images\\39.jpg')
        image = cv2.resize(image, (642,480), interpolation= cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        

        
        cv2.waitKey(1)
        # print(image_array.shape)
        # Add an extra dimension to the image (since Keras expects a batch of images)
        image_array = np.expand_dims(image, axis=0)

        t1 = time.perf_counter()
        # Predict the class of the image
        predictions = self.model.predict(image_array, batch_size=1).round()

        t2 = time.perf_counter()
        elapsed_time = t2 - t1
        
        # Print the elapsed time

        # print(f"Inference took {elapsed_time:.10f} seconds")

        predictions =predictions.squeeze()

        # print(predictions)
        return predictions




class AlexNet:

    def __init__(self) -> None:
        

        # Define the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (11, 11), activation='relu', input_shape=(224, 224, 3)))
        model.add(tf.keras.layers.MaxPooling2D((3, 3)))
        model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
        model.load_weights('models\\alex.h5')
        self.model = model



    def detectImg(self,img):
        
        
        image = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
        # image = cv2.imread('datasets\\cs\\test\\images\\39.jpg')
        image = cv2.resize(image, (224,224), interpolation= cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        

        
        cv2.waitKey(1)
        # print(image_array.shape)
        # Add an extra dimension to the image (since Keras expects a batch of images)
        image_array = np.expand_dims(image, axis=0)

        t1 = time.perf_counter()
        # Predict the class of the image
        predictions = self.model.predict(image_array, batch_size=1).round()

        t2 = time.perf_counter()
        elapsed_time = t2 - t1
        
        # Print the elapsed time

        # print(f"Inference took {elapsed_time:.10f} seconds")

        predictions =predictions.squeeze()

        # print(predictions)
        return predictions




class AlexFar:

    def __init__(self) -> None:
        

        # Define the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (11, 11), activation='relu', input_shape=(224, 224, 3)))
        model.add(tf.keras.layers.MaxPooling2D((3, 3)))
        model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((3, 3)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
        model.load_weights('models\\alex_far.h5')
        self.model = model



    def detectImg(self,img):
        
        
        image = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
        # image = cv2.imread('datasets\\cs\\test\\images\\39.jpg')
        image = cv2.resize(image, (224,224), interpolation= cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        

        
        cv2.waitKey(1)
        # print(image_array.shape)
        # Add an extra dimension to the image (since Keras expects a batch of images)
        image_array = np.expand_dims(image, axis=0)

        t1 = time.perf_counter()
        # Predict the class of the image
        predictions = self.model.predict(image_array, batch_size=1).round()

        t2 = time.perf_counter()
        elapsed_time = t2 - t1
        
        # Print the elapsed time

        # print(f"Inference took {elapsed_time:.10f} seconds")

        predictions =predictions.squeeze()

        # print(predictions)
        return predictions












