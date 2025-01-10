import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('trained-models/nasal_endoscope_classifier.h5')

# Convert the model to TFLite with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# (Optional) Set representative dataset for quantization (if using post-training quantization)
# You can skip this for now or provide a representative dataset if needed.

tflite_model = converter.convert()

# Save the converted and optimized model
with open('trained-models/nasal_endoscope_classifier.tflite', 'wb') as f:
    f.write(tflite_model)