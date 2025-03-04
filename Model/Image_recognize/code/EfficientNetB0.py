import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# 🔹 Step 1: Set Dataset Path
dataset_path = "Model/Image_recognize/Split_dataset"  # Change this if needed

# 🔹 Step 2: Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# Load Train and Validation Data
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# 🔹 Step 3: Load Pretrained EfficientNetB0 Model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze Base Model Layers
base_model.trainable = False  # Freeze all base layers

# 🔹 Step 4: Add Custom Layers for Classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)  # Prevent overfitting
output = Dense(3, activation="softmax")(x)  # 3 Classes (benign, malignant, normal)

# Create Final Model
model = Model(inputs=base_model.input, outputs=output)

# 🔹 Step 5: Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# 🔹 Step 6: Train the Model
model.fit(train_generator, validation_data=val_generator, epochs=50)  # Start with 50 epochs

# 🔹 Step 7: Evaluate on Test Data
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 🔹 Step 8: Save the Best Model
model.save("breast_cancer_image_model_efficientnet.keras")
