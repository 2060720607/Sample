from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import utils
from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------
# 1. Load CIFAR-10 Dataset
# ---------------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Shape of training data =", x_train.shape)
print("Shape of testing data =", x_test.shape)

print("No. of training samples =", x_train.shape[0])
print("No. of testing samples =", x_test.shape[0])


# ---------------------------------------
# 2. Display First Image
# ---------------------------------------

print("First image before normalization:")
print(x_train[0])


# ---------------------------------------
# 3. Normalize Image Data
# ---------------------------------------

x_train = x_train / 255.0
x_test = x_test / 255.0

print("First image after normalization:")
print(x_train[0])


# ---------------------------------------
# 4. Display First Image
# ---------------------------------------

arr = y_train[0]
print("Label of first image =", arr)

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

plt.imshow(x_train[0])
plt.title(class_names[y_train[0][0]])
plt.axis('off')
plt.show()


# ---------------------------------------
# 5. Display First 10 Images
# ---------------------------------------

for i in range(10):

    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])

plt.tight_layout()
plt.show()


# ---------------------------------------
# 6. Convert Labels to Categorical
# ---------------------------------------

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)


# ---------------------------------------
# 7. Create CNN Model
# ---------------------------------------

model = Sequential([

    Input(shape=(32, 32, 3)),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation='relu'
    ),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation='relu'
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Conv2D(
        128,
        kernel_size=(3, 3),
        activation='relu'
    ),

    Conv2D(
        128,
        kernel_size=(3, 3),
        activation='relu'
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.5),

    Flatten(),

    Dense(
        10,
        activation='softmax'
    )
])


# ---------------------------------------
# 8. Compile Model
# ---------------------------------------

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ---------------------------------------
# 9. Display Model Summary
# ---------------------------------------

model.summary()


# ---------------------------------------
# 10. Train the Model
# ---------------------------------------

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test)
)


# ---------------------------------------
# 11. Evaluate the Model
# ---------------------------------------

val_loss, val_acc = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("Validation Loss =", val_loss)
print("Validation Accuracy =", val_acc)


# ---------------------------------------
# 12. Predict First Training Image
# ---------------------------------------

image = x_train[0]

# Add batch dimension
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

print("Prediction probabilities:")
print(prediction)


# ---------------------------------------
# 13. Find Predicted Class
# ---------------------------------------

label = np.argmax(prediction)

print("Predicted class number =", label)
print("Predicted class =", class_names[label])


# ---------------------------------------
# 14. Display Image with Prediction
# ---------------------------------------

plt.imshow(x_train[0])
plt.title("Predicted: " + class_names[label])
plt.axis('off')
plt.show()