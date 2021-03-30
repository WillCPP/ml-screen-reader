import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential
import argparse
import os
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

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
    
    count_0 = 0
    count_1 = 0
    index_0 = -1
    for i in y_val:
        if i == 0.0:
            count_0 += 1
            if index_0 == -1:
                index_0 = count_0 + count_1 - 2
        if i == 1.0:
            count_1 += 1
    print(f'0: {count_0}')
    print(f'1: {count_1}')
    print(f'i: {index_0}')
    
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
                           
#     print(y_val[0])
#     print(y_val[index_0])
#     print(y_val[index_0+1])
#     print(y_val[index_0+2])
#     print(y_val[index_0+3])
#     print(x_val[index_0])
#     print(x_val[index_0+1])
#     print(x_val[index_0+2])
#     print(x_val[index_0+3])

    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    print(x_train.shape)

    learning_rate = 0.01  # Learning rate.
    num_epochs = 3  # Number of epochs.

    # Regular DNN ==================================================
    class DNN():
        def __init__(self, input_shape, num_classes):
            self.model = None
            self.input_shape = input_shape
            self.num_classes = num_classes

        def createModel(self):
            layers = [
                Conv1D(32, (200), input_shape=(200, 1), activation="relu"),
                Conv1D(32, 1, activation="relu"),
                GlobalMaxPooling1D(),
                Dense(32, activation="relu"),
                Dense(2, activation="softmax", name="predictions"),
            ]
            model = Sequential(layers)
            model.summary()

            loss = keras.losses.CategoricalCrossentropy(from_logits=True)
            model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate), metrics=[keras.metrics.CategoricalAccuracy()])
            self.model = model

        def train(self, x_train, y_train):
            if self.model is None:
                print('Model has not been created yet, run createModel() first.')
                return
            else:
                self.model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1)
        
        def saveModel(self, path):
#             self.model.save(path)
            tf.saved_model.save(self.model, path + '/1')

        def loadModel(self, path):
            self.model = keras.models.load_model(path)

    def run_experiment():
        dnn = DNN((200,),2,)
        dnn.createModel()
        dnn.train(x_train, y_train)
        print(model_dir)
        dnn.saveModel(model_dir)
        score = dnn.model.evaluate(x_val, y_val, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    # ==================================================

    run_experiment()