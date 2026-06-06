import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

IMG_SIZE = 48
BATCH_SIZE = 64

# Load dataset
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "FER-2013/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    "FER-2013/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ANN model
model = Sequential([
    Flatten(input_shape=(48,48,1)),

    Dense(1024, activation='relu'),
Dense(512, activation='relu'),
Dense(256, activation='relu'),
Dense(7, activation='softmax')
])

model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=50,
    callbacks=[early_stop]
)
import matplotlib.pyplot as plt

# Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.savefig('accuracy.png')
plt.show()

# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.savefig('loss.png')
plt.show()

loss, acc = model.evaluate(test_data)

from sklearn.metrics import classification_report
import numpy as np

predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)

print(
    classification_report(
        test_data.classes,
        y_pred,
        target_names=list(test_data.class_indices.keys())
    )
)

print(f"\nFinal Accuracy = {acc:.4f}")

model.save("emotion_ann.keras")

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(
    test_data.classes,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(test_data.class_indices.keys()),
    yticklabels=list(test_data.class_indices.keys())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()
model.save("emotion_ann.keras")