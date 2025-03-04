import tensorflow as tf
import json
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

class COCODataGenerator(Sequence):
    def __init__(self, annotation_path, image_dir, batch_size=32, target_size=(640, 640), shuffle=True):
        with open(annotation_path, "r") as f:
            coco_data = json.load(f)
        
        # Extract image and category mappings
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        
        # Create mappings from COCO JSON
        self.image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
        self.image_id_to_label = {}
        for ann in coco_data["annotations"]:
            self.image_id_to_label[ann["image_id"]] = ann["category_id"]  # Assign label
        
        self.image_ids = list(self.image_id_to_file.keys())
        if self.shuffle:
            np.random.shuffle(self.image_ids)
    
    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))
    
    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        
        for img_id in batch_ids:
            img_path = os.path.join(self.image_dir, self.image_id_to_file[img_id])
            image = load_img(img_path, target_size=self.target_size)
            image = img_to_array(image) / 255.0  # Normalize
            batch_images.append(image)
            batch_labels.append(self.image_id_to_label[img_id])
        
        return np.array(batch_images), tf.keras.utils.to_categorical(batch_labels, num_classes=3)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

# Dataset Paths
dataset_path = "Model/Image_recognize/New_image_training"  # Update with actual path
train_annotations = os.path.join(dataset_path, "train", "_annotations.coco.json")
valid_annotations = os.path.join(dataset_path, "valid", "_annotations.coco.json")
test_annotations = os.path.join(dataset_path, "test", "_annotations.coco.json")
train_images = os.path.join(dataset_path, "train")
valid_images = os.path.join(dataset_path, "valid")
test_images = os.path.join(dataset_path, "test")

# Load Data
train_generator = COCODataGenerator(train_annotations, train_images)
valid_generator = COCODataGenerator(valid_annotations, valid_images)
test_generator = COCODataGenerator(test_annotations, test_images)

# Load Pretrained Model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(640, 640, 3))
base_model.trainable = False

# Custom Classification Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(3, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Train Model
model.fit(train_generator, validation_data=valid_generator, epochs=100)

# Evaluate Model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f'Test Loss: {loss *100:.2f}%')

# Save Model
path=os.path.join('Model/Image_recognize/Model','breast_cancer_image_model_efficientnet.keras')
model.save(path)
