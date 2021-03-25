import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential
import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # ================================================== AWS
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    # ================================================== AWS

#     # ================================================== Local
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--learning-rate', type=float, default=0.01)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--gpu-count', type=int, default=0)
#     parser.add_argument('--model-dir', type=str, default='model')
#     parser.add_argument('--training', type=str, default='data')
#     parser.add_argument('--validation', type=str, default='data')
#     # ================================================== Local
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation

#     # ==================================================
#     ds_data = np.load('dataset3/data.npy', allow_pickle=True)
#     ds_labels = np.load('dataset3/labels.npy', allow_pickle=True)
#     print(ds_data.shape)
#     print(ds_labels.shape)
#     x_train, x_val, y_train, y_val = train_test_split(ds_data, ds_labels, train_size=0.75)
#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
#     x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
#     # ==================================================

    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    print(x_train.shape)

    learning_rate = 0.001  # Learning rate.
    num_epochs = 3  # Number of epochs.

    # Regular DNN ==================================================
    class DNN():
        def __init__(self, input_shape, num_classes):
            self.model = None
            self.input_shape = input_shape
            self.num_classes = num_classes

        def createModel(self):
            layers = [
                # keras.Input(shape=self.input_shape, dtype='float64'),
                Conv1D(32, (200), input_shape=(200,1), activation="relu"),
                Conv1D(32, 1, activation="relu"),
                GlobalMaxPooling1D(),
                Dense(32, activation="relu"),
                # Dropout(0.5),
                Dense(1, activation="sigmoid", name="predictions"),

                # Conv2D(32, kernel_size=(3, 3), activation="relu"),
                # MaxPooling2D(pool_size=(2, 2)),
                # Conv2D(64, kernel_size=(3, 3), activation="relu"),
                # MaxPooling2D(pool_size=(2, 2)),
                # Conv2D(128, kernel_size=(3, 3), activation="relu"),
                # MaxPooling2D(pool_size=(2, 2)),
                # Flatten(),
                # Dropout(0.5),
                # Dense(self.num_classes, activation="softmax"),
            ]
            model = Sequential(layers)
            model.summary()

            # loss = loss=keras.losses.CategoricalCrossentropy(from_logits=True)

            # model.compile(optimizer=keras.optimizers.Adam(), loss=loss, metrics=[keras.metrics.CategoricalAccuracy()])
            model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.01), metrics=["accuracy"])
            self.model = model

        def train(self, x_train, y_train):
            if self.model is None:
                print('Model has not been created yet, run createModel() first.')
                return
            else:
                # optimizer = keras.optimizers.Adam(0.001)

                # self.model.compile(optimizer=optimizer,
                #     loss=keras.losses.CategoricalCrossentropy(),
                #     metrics=[keras.metrics.CategoricalAccuracy()])

                # self.model.fit(x_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], validation_split=self.params['validation_split'])
                self.model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1)
        
        def saveModel(self, path):
            self.model.save(path)
            tf.saved_model.save(self.model, 'models/tf/1')

        def loadModel(self, path):
            self.model = keras.models.load_model(path)

    def run_experiment():
        dnn = DNN((200,),2,)
        dnn.createModel()
        dnn.train(x_train, y_train)
        dnn.saveModel('models/current/1')
        score = dnn.model.evaluate(x_val, y_val, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    # ==================================================

    run_experiment()