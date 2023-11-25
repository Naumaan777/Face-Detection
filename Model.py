import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling and augmentation
train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
        '/content/fer2013/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all test images
validation_generator = validation_data_gen.flow_from_directory(
        '/content/fer2013/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
Model = Sequential()

Model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
Model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

Model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
Model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))


Model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
Model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

Model.add(Flatten())
Model.add(Dense(1024, activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(7, activation='softmax'))

Model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
Model_info = Model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = Model.to_json()
with open("Model.json", "w") as json_file:
    json_file.write(model_json)

Model.save_weights('Model.h5')

