import tensorflow as tf
import sys

print("--- Checking TensorFlow GPU ---")
try:
    print(f'TensorFlow Version: {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    print(f'Found GPUs: {gpus}')
    assert len(gpus) > 0, 'ERROR: No GPU detected'
    print('--- GPU Check PASSED ---')
except Exception as e:
    print(f'GPU Check FAILED: {e}', file=sys.stderr)
    sys.exit(1)  # Exit with error code if assertion fails
