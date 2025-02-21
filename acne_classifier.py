import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
def create_acne_classifier():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: Acne or No Acne
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
def prepare_data(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    val_data = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    return train_data, val_data

# Train and save the model
def train_and_save_model(train_dir, val_dir, save_path):
    train_data, val_data = prepare_data(train_dir, val_dir)
    model = create_acne_classifier()
    model.fit(train_data, epochs=10, validation_data=val_data)
    model.save(save_path)
    print(f"Model saved at {save_path}")


train_dataset_path = "Data/Train"
val_dataset_path = "Data/Test"
train_and_save_model(train_dataset_path, val_dataset_path, 'acne_classifier.h5')
