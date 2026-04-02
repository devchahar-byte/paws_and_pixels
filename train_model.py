import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# 1. Setup Data Generators
# This loads the images from the 'dataset' folder you just created
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    './dataset/training', target_size=(160, 160), batch_size=32, class_mode='binary')

val_ds = val_datagen.flow_from_directory(
    './dataset/validation', target_size=(160, 160), batch_size=32, class_mode='binary')

# 2. Load MobileNetV2 (The Teacher)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False # Freeze the teacher's brain

# 3. Add Student Layers (The part that learns Cats vs Dogs)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train
print("Starting training... (This will take a few minutes)")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 5. Save
model.save('cat_dog_classifier.h5')
print("Model saved as cat_dog_classifier.h5")