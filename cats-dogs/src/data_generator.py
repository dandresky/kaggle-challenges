from keras.preprocessing.image import ImageDataGenerator as idg


def get_datagenerators(X_train, y_train, X_test, y_test, version, batch_size):

    if(version == 1):
        return get_datagenerators_v1(X_train, y_train, X_test, y_test, batch_size)
    else:
        return get_datagenerators_v1(X_train, y_train, X_test, y_test, batch_size)


def get_datagenerators_v1(X_train, y_train, X_test, y_test, batch_size):
    '''
    Define the image manipulation steps to be randomly applied to each image.
    Multiple versions of this function will likely exist to test different
    strategies. Return a generator for both train and test data.
    '''
    print("Create Image Data Generators for train and test ... ")
    train_datagen = idg(featurewise_center=False, # default
        samplewise_center=False,                    # default
        featurewise_std_normalization=False,        # default
        samplewise_std_normalization=False,         # default
        zca_whitening=False,                        # default
        zca_epsilon=1e-6,                           # default
        rotation_range=0.,                          # default
        width_shift_range=0.,                       # default
        height_shift_range=0.,                      # default
        shear_range=0.,                             # default
        zoom_range=0.,                              # default
        channel_shift_range=0.,                     # default
        fill_mode='nearest',                        # default
        cval=0.,                                    # default
        horizontal_flip=False,                      # default
        vertical_flip=False,                        # default
        rescale=1./255,                             # rescale RGB vales
        preprocessing_function=None,                # default
        data_format='channels_last')                # default
    test_datagen = idg(featurewise_center=False,  # default
        samplewise_center=False,                    # default
        featurewise_std_normalization=False,        # default
        samplewise_std_normalization=False,         # default
        zca_whitening=False,                        # default
        zca_epsilon=1e-6,                           # default
        rotation_range=0.,                          # default
        width_shift_range=0.,                       # default
        height_shift_range=0.,                      # default
        shear_range=0.,                             # default
        zoom_range=0.,                              # default
        channel_shift_range=0.,                     # default
        fill_mode='nearest',                        # default
        cval=0.,                                    # default
        horizontal_flip=False,                      # default
        vertical_flip=False,                        # default
        rescale=1./255,                             # rescale RGB vales
        preprocessing_function=None,                # default
        data_format='channels_last')                # default

    train_generator = train_datagen.flow(
        X_train,
        y_train,    # labels just get passed through
        batch_size=batch_size,
        shuffle=False,
        seed=None)
    test_generator = test_datagen.flow(
        X_test,
        y_test, # labels just get passed through
        batch_size=batch_size,
        shuffle=False,
        seed=None)

    return train_generator, test_generator
