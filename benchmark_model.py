''' This module evaluates the performance of a trained CPC encoder '''

from data_utils import MnistGenerator
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization,LeakyReLU

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_DETERMINISTIC_OPS'] = '1' #设置随机种子
os.environ['PYTHONHASHSEED'] = str(123)
tf.random.set_seed(123)

def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = tf.keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = Input(image_shape)
    x = encoder(x_input)
    x = Dense(units=128, activation='linear')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(units=10, activation='softmax')(x)

    # Model
    model = tf.keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, image_size=28, color=False):

    # Prepare data
    train_data = MnistGenerator(batch_size, subset='train', image_size=image_size, color=color, rescale=True)

    validation_data = MnistGenerator(batch_size, subset='valid', image_size=image_size, color=color, rescale=True)

    # Prepares the model
    model = build_model(encoder_path, image_shape=(image_size, image_size, 3), learning_rate=lr)

    # Callbacks
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    model.save(os.path.join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    benchmark_model(
        encoder_path='models/64x64/encoder.h5',
        epochs=15,
        batch_size=64,
        output_dir='models/64x64',
        lr=1e-3,
        image_size=64,
        color=True
    )
