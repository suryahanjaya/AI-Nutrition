import tensorflow as tf
from prepare_dataset import train_data, val_data
from build_model import build_model

# Load dataset
NUM_CLASSES = len(train_data.class_indices)

# Build model
model = build_model(NUM_CLASSES)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',  # Updated to use `.keras` format
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor='val_loss'
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[checkpoint, early_stopping]
)

# Save the model
model.save('final_model.h5')
