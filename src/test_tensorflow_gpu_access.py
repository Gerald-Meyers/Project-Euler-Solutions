import tensorflow as tf

print("""
      --- Starting GPU Test Script ---

      TensorFlow Version:""", tf.__version__)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nFound {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
        # Set memory growth for each GPU
        tf.config.experimental.set_memory_growth(gpu, True)
    print("""
          ---
          GPU(s) are available and memory growth has been set.
          ---
          """)
else:
    print("""
          ---
          No GPU devices found. TensorFlow will use CPU.
          ---
          """)

# Simple TensorFlow operation to test GPU
try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("\nTensorFlow matrix multiplication result (should be on GPU if available):")
        print(c.numpy())
except RuntimeError as e:
    print(f"\nError during TensorFlow operation: {e}")
    print("This might indicate an issue with GPU setup or drivers.")

print("\nTensorFlow GPU access test complete.")
