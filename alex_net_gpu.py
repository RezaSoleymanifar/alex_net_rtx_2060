import tensorflow as tf

#Importing library
import keras
# from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert(tf.test.is_gpu_available())

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False) # Start with XLA disabled.

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
  x_train = x_train.astype('float32') / 256
  x_test = x_test.astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
  return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()

def generate_model():
    # Instantiation
  AlexNet = Sequential()

  # 1st Convolutional Layer
  AlexNet.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=(11, 11), strides=(4, 4), padding='same'))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))
  AlexNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  # 2nd Convolutional Layer
  AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))
  AlexNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  # 3rd Convolutional Layer
  AlexNet.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))

  # 4th Convolutional Layer
  AlexNet.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))

  # 5th Convolutional Layer
  AlexNet.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))
  AlexNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

  # Passing it to a Fully Connected layer
  AlexNet.add(Flatten())
  # 1st Fully Connected Layer
  AlexNet.add(Dense(4096, input_shape=(32, 32, 3,)))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))
  # Add Dropout to prevent overfitting
  AlexNet.add(Dropout(0.4))

  # 2nd Fully Connected Layer
  AlexNet.add(Dense(4096))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))
  # Add Dropout
  AlexNet.add(Dropout(0.4))

  # 3rd Fully Connected Layer
  AlexNet.add(Dense(1000))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('relu'))
  # Add Dropout
  AlexNet.add(Dropout(0.4))

  # Output Layer
  AlexNet.add(Dense(100))
  AlexNet.add(BatchNormalization())
  AlexNet.add(Activation('softmax'))

  # Model Summary
  AlexNet.summary()
  return AlexNet

model = generate_model()

def create_callbacks():
  earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
  mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
  tboard = TensorBoard(log_dir='tboard', update_freq = 'epoch')
  cb_list = [earlyStopping, mcp_save, tboard]
  return cb_list

def compile_model(model):
  opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model

model = compile_model(model)

def train_model(model, x_train, y_train, x_test, y_test, epochs=25):
  model.fit(x_train, y_train, batch_size=256, epochs=epochs,
            validation_data=(x_test, y_test), shuffle=True, callbacks=create_callbacks())

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

warmup(model, x_train, y_train, x_test, y_test)
train_model(model, x_train, y_train, x_test, y_test)

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
