import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, recall_score
from collections import Counter
import os
# import ipykernel  # 让pycharm训练时只打印一行

# 显存 https://blog.csdn.net/CxsGhost/article/details/105985955
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def get_best_all(history):
    best_val_acc_list = history.history['accuracy']
    best_val_acc = max(best_val_acc_list)
    best_val_acc_index = best_val_acc_list.index(best_val_acc)

    epoch_index = best_val_acc_index + 1
    print('epoch_index: ', epoch_index)

    best_train_acc = history.history['accuracy'][best_val_acc_index]
    best_train_loss = history.history['loss'][best_val_acc_index]

    return best_train_acc, best_train_loss


def model_Sequential(x_train, batch_size, num_lables):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(x_train.shape[1], 1, x_train.shape[3]), batch_size=batch_size),
        keras.layers.Conv2D(8, kernel_size=(5, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format='channels_last'),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        keras.layers.Conv2D(16, kernel_size=(5, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format="channels_last"),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        keras.layers.Conv2D(8, kernel_size=(5, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format="channels_last"),
        # keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(num_lables, activation='softmax')
    ])
    print(model.summary())
    return model


EPOCH = 1000
BATCH_SIZE = 16
NUM_LABLES = 2
PATIENCE = 15  # early Stop
save_dir = './2018CNN/'
path = 'data'
save_model_dir = './save_model/'

if __name__ == '__main__':
    x_train = np.load('./x_train.npy')
    y_train = np.load('./y_train.npy')
    x_test = np.load('./x_test.npy')
    y_test = np.load('./y_test.npy')
    print(x_train.shape)
    x_val = x_test
    y_val = y_test

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    # reshape
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))
    y_test_true = y_test
    y_train = keras.utils.to_categorical(y_train, NUM_LABLES)
    y_test = keras.utils.to_categorical(y_test, NUM_LABLES)

    # cut
    train_num = x_train.shape[0] % BATCH_SIZE
    test_num = x_test.shape[0] % BATCH_SIZE
    if train_num != 0:
        x_train = x_train[0:-train_num]
        y_train = y_train[0:-train_num]

    if test_num != 0:
        x_test = x_test[0:-test_num]
        y_test = y_test[0:-test_num]
        y_test_true = y_test_true[0:-test_num]

    print('x_train: ', x_train.shape)
    print('x_test: ', x_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape)

    cnn_model = model_Sequential(
        x_train=x_train, batch_size=BATCH_SIZE, num_lables=NUM_LABLES)

    model_save_path = save_model_dir + '2018CNN.h5'

    # 检查点回调
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_best_only=True,
                                                    save_weights_only=True, monitor='loss')

    # earlystop 回调
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=PATIENCE, restore_best_weights=True, monitor='loss')

    # 编译
    cnn_model.compile(optimizer=keras.optimizers.Adam(clipnorm=1., lr=0.0001),
                      loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # 训练
    history = cnn_model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1,
                            callbacks=[checkpoint_cb, early_stopping_cb])

    early_stopping_epoch = early_stopping_cb.stopped_epoch - PATIENCE + 1
    print('Early stopping epoch: ' + str(early_stopping_epoch))
    # 画图
    font2 = {'size': 10}
    plt.figure()
    # plt.subplot(2,2,1)
    # plt.title('2018CNN-acc')
    plt.plot(history.history['accuracy'],
             label='train accuracy', linewidth=1.5)
    # plt.plot(history.history['val_accuracy'],
    #          label='val accuracy', linewidth=1.5)
    plt.legend(prop=font2)
    # plt.show()
    plt.savefig(save_dir + '2018CNN-acc.png')

    plt.figure()
    # plt.title(f'2018CNN-loss')
    plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
    # plt.plot(history.history['val_loss'], label='val loss', linewidth=1.5)
    plt.legend(prop=font2)
    # plt.show()
    plt.savefig(save_dir + '2018CNN-loss.png')

    # load model
    cnn_model.load_weights(model_save_path)
    train_acc, train_loss = get_best_all(history)
    test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test)

    fp = open(save_dir + f"2018CNN.txt", 'w+')
    fp.write('train_loss: ' + str(train_loss) + '\n' +
             'train_acc:  ' + str(train_acc) + '\n')
    # fp.write('val_loss: ' + str(val_loss) + '\n' + 'val_acc: ' + str(val_acc) + '\n')
    fp.write('test_loss: ' + str(test_loss) + '\n' +
             'test_accuracy: ' + str(test_accuracy) + '\n')
    fp.close()

    print('test_loss:', test_loss)
    print('test_accuracy', test_accuracy)
    # 测试集验证
    y_test_predict = cnn_model.predict(x_test, batch_size=BATCH_SIZE)
    y_test_predict = np.array(y_test_predict)
    y_test_predict = np.argmax(y_test_predict, axis=1)

    report = classification_report(
        y_test_true, y_test_predict, digits=5, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df.to_csv(save_dir + f"2018CNN.csv", index=True)
