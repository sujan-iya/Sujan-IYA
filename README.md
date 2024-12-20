## Hi there 👋

<!--
**sujan-iya/Sujan-IYA** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Regression output

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Example data (replace with actual satellite and poverty data)
# Assuming images are 64x64 pixels with 3 color channels (RGB)
num_samples = 1000
input_shape = (64, 64, 3)

# Generate random data for illustration purposes
X_train = np.random.rand(num_samples, *input_shape)
y_train = np.random.rand(num_samples)  # Poverty levels

# Create and train the model
model = create_cnn_model(input_shape)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot training history
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Example prediction (replace with actual test data)
X_test = np.random.rand(10, *input_shape)
predictions = model.predict(X_test)
print(predictions)
