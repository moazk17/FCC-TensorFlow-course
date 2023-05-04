import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split dataset into training and testing sets
train_images.shape # (60000, 28, 28)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure() # create a new figure
# plt.imshow(train_images[1]) # display an image
# plt.colorbar() # add a colorbar to a plot
# plt.grid(True) # configure the grid lines
# plt.show() # display the figure

train_images = train_images / 255.0 # scale the values to a range of 0 to 1
test_images = test_images / 255.0

model = keras.Sequential([ # build the model
    keras.layers.Flatten(input_shape=(28, 28)), # transform the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Dense(128, activation='relu'), # first Dense layer has 128 nodes (or neurons)
    keras.layers.Dense(10, activation='softmax') # second (and last) layer returns a logits array with length of 10
])
model.compile(optimizer='adam', # optimizer — This is how the model is updated based on the data it sees and its loss function.
              loss='sparse_categorical_crossentropy', # loss function — This measures how accurate the model is during training.
              metrics=['accuracy']) # metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.fit(train_images, train_labels, epochs=5) # train the model 

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1) # evaluate accuracy
print('Test accuracy:', test_acc) 
predictions = model.predict(test_images) # make predictions

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img,cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(True)
    plt.show()
    
def get_number():
    while True:
        num = input("Pick a number between 0 and 999 or enter -1 to exit: ")
        if num.isdigit():
            num = int(num)
            if 0<= num <= 999:
                return num
        elif num == '-1':
            return num
        else:
            print("Try again...")

while True:
    num = get_number()
    if num == '-1':
        break
    image = test_images[num]
    label = test_labels[num]
    prediction = predictions[num]
    guess = class_names[np.argmax(prediction)]
    show_image(image, class_names[label], guess)