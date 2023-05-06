import tensorflow as tf
import matplotlib.pyplot as plt

datasets = tf.keras.datasets
layers = tf.keras.layers
models = tf.keras.models

# Load the data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

model= models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=2,
                    validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Data Augmentation

image = tf.keras.preprocessing.image
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40, # rotate the image 40 degrees
    width_shift_range=0.2, # Shift the pic width by a max of 20%
    height_shift_range=0.2, # Shift the pic height by a max of 20%
    shear_range=0.2, # Shear means cutting away part of the image (max 20%)
    zoom_range=0.2, # Zoom in by 20% max
    horizontal_flip=True, # Allo horizontal flipping
    fill_mode='nearest' # Fill in missing pixels with the nearest filled value
)

test_img = train_images[14]
img = image.img_to_array(test_img) # Convert image to numpy array
img = img.reshape((1,) + img.shape) # Reshape image
i=0
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'): # Loops forever, adding augmented images to the end of your train set
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4: # Show 4 images
        break

plt.show()