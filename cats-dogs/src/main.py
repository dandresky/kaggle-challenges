from cnn_model import build_model
from data_generator import get_datagenerators
import datetime as dt
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf

BATCH_SIZE = 32
EPOCHS = 2
GENERATOR_VERSION = 1
MODEL_VERSION = 1
OPTIMIZER = 'sgd'

def get_data():
    '''
    Images have already been screened, resized, and converted to numpy arrays.
    They are are stored in ../../../kaggle-data/cats-dogs/
        processed_training_images.npy
        processed_test_images.npy
        training_labels.npy
        test_labels.npy
    The selected batch size will not always divide evenly into the total number
    of samples which causes errors with the Keras functions that use the image
    data generator. The leftover samples are trimmed from the data set to avoid
    this.
    '''
    logging.info('Loading numpy arrays ...')
    print('\nLoading numpy arrays ... ...')
    X_train = np.load('../../../kaggle-data/cats-dogs/processed_training_images.npy')
    X_test = np.load('../../../kaggle-data/cats-dogs/processed_test_images.npy')
    y_train = np.load('../../../kaggle-data/cats-dogs/training_labels.npy')
    y_test = np.load('../../../kaggle-data/cats-dogs/test_labels.npy')

    logging.info('Trimming data to integer number of batches ...')
    print("Trimming data to integer number of batches ...")
    num_train_batches = X_train // BATCH_SIZE
    num_test_batches = X_test // BATCH_SIZE
    X_train = X_train[:len(num_train_batches * BATCH_SIZE)]
    y_train = y_train[:len(num_train_batches * BATCH_SIZE)]
    X_test = X_test[:len(num_test_batches * BATCH_SIZE)]
    y_test = y_test[:len(num_test_batches * BATCH_SIZE)]
    logging.info("  X_train samples = %d" % X_train.shape[0])
    logging.info("  y_train samples = %d" % y_train.shape[0])
    logging.info("  X_test samples = %d" % X_test.shape[0])
    logging.info("  y_test samples = %d" % y_test.shape[0])

    return X_train, y_train, X_test, y_test

def main():

    # init random seeds to ensure consistent results during evaluation
    random.seed(39)
    np.random.seed(39)
    tf.set_random_seed(39)

    # begin a logging function to record events
    try:
        os.remove('cnn_model_v1.log')    # delete the existing file to start new
    except OSError:
        pass
    logging.basicConfig(filename='cnn_model.log',level=logging.DEBUG)
    logging.info('Begin training CNN Model v%s ...', MODEL_VERSION)
    logging.info("  Batch size = %s" % BATCH_SIZE)
    logging.info("  Epochs = %s" % EPOCHS)
    logging.info("  Optimizer = %s" % OPTIMIZER)
    start_time = dt.datetime.now()

    # read pre-processed data and trim to integer number of batches
    X_train, y_train, X_test, y_test = get_data()

    # data generators are instructions to Keras for further processing of the
    # image data (in batches) before training on the image.
    train_generator, test_generator = \
        get_datagenerators(X_train,
                           y_train,
                           X_test,
                           y_test,
                           GENERATOR_VERSION,
                           BATCH_SIZE)

    # get the model
    logging.info("Build Convolutional and Fully Connected layers ...")
    print("Build Convolutional and Fully Connected layers ... ")
    model = build_model(MODEL_VERSION,
                        OPTIMIZER,
                        input_shape=X_train.shape[1:])
    logging.info("Model Summary \n%s" % model.to_json())
    model.summary()

    logging.info('Fitting the model ...')
    print('\nFitting the model ...')
    hist = model.fit_generator(train_generator,
        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=True,
        callbacks=None,
        validation_data=test_generator,
        validation_steps=X_test.shape[0] // BATCH_SIZE,
        class_weight=None,
        max_queue_size=BATCH_SIZE*8,
        workers=8,
        use_multiprocessing=False,
        initial_epoch=0)

    logging.info('Scoring the model ...')
    print('\nScoring the model ...')
    scores = model.evaluate_generator(test_generator,
        steps=X_test.shape[0] // BATCH_SIZE,
        max_queue_size=BATCH_SIZE*8,
        workers=8,
        use_multiprocessing=False)
    print(scores)
    logging.info("  Loss: %.3f    Accuracy: %.3f" % (scores[0], scores[1]))

    logging.info('Making predictions ...')
    print('\nMaking predictions ...')
    pred = model.predict_generator(test_generator,
        steps=X_test.shape[0] // BATCH_SIZE,
        max_queue_size=BATCH_SIZE*8,
        workers=8,
        use_multiprocessing=False,
        verbose=True)

    print("Predictions: ", pred)
    print("Actual: ", y_test)
    np.save('../../../kaggle-data/cats-dogs/cnn_model_predictions.npy', pred)

    logging.info("Saving model ...")
    print("Saving model ...")
    model.save('../../../kaggle-data/cats-dogs/cnn_model.h5')

    stop_time = dt.datetime.now()
    print("Scanning and shuffling took ", (stop_time - start_time).total_seconds(), "s.\n")
    logging.info("Training complete. Elapsed time = %.0fs", (stop_time - start_time).total_seconds())

    pickle.dump(hist.history, open("../../../kaggle-data/cats-dogs/cnn_model.pkl", "wb"))









if __name__ == '__main__':
    main()
