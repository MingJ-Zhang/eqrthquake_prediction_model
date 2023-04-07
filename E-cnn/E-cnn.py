import tensorflow as tf
from tensorflow import keras
# import keras
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score
from collections import Counter
from sklearn.model_selection import cross_val_score, StratifiedKFold


# 显存 https://blog.csdn.net/CxsGhost/article/details/105985955
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def get_best_all(history):
    best_val_acc_list = history.history['val_accuracy']
    best_val_acc = max(best_val_acc_list)
    best_val_acc_index = best_val_acc_list.index(best_val_acc)

    epoch_index = best_val_acc_index + 1
    print('epoch_index: ', epoch_index)

    best_train_acc = history.history['accuracy'][best_val_acc_index]
    best_train_loss = history.history['loss'][best_val_acc_index]
    best_val_acc_test = history.history['val_accuracy'][best_val_acc_index]
    best_val_loss = history.history['val_loss'][best_val_acc_index]

    return best_train_acc, best_train_loss, best_val_acc, best_val_loss


# 张明杰
def model_Sequential(x_train, batch_size, num_lables):
    model = keras.models.Sequential([
        keras.layers.Input(
            shape=(x_train.shape[1], 1, x_train.shape[3]), batch_size=batch_size),
        keras.layers.Conv2D(8, kernel_size=(7, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format='channels_last'),
        keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9, weights=None),
        keras.layers.Conv2D(16, kernel_size=(15, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format="channels_last"),
        keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9, weights=None),
        keras.layers.Conv2D(8, kernel_size=(3, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format="channels_last"),
        keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9, weights=None),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(num_lables, activation='sigmoid')
    ])

    print(model.summary())
    return model


EPOCH = 500
BATCH_SIZE = 16
NUM_LABLES = 2
PATIENCE = 10  # early Stop
save_dir = 'images'
path = 'data'
save_model_dir = 'save_model'


def pltfig(history, i):
    font2 = {'size': 10}
    plt.figure()
    # plt.title('CNN-acc-loss')
    plt.plot(history.history['accuracy'],
             label='train accuracy', linewidth=1.5)
    # plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
    plt.legend(prop=font2)
    plt.savefig('./data/_3_1/testmodelacc' + str(i) + '.png')
    
    plt.figure()
    # plt.title('CNN-acc-loss')
    # plt.plot(history.history['accuracy'],
    #          label='train accuracy', linewidth=1.5)
    plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
    plt.legend(prop=font2)
    plt.savefig('./data/_3_1/testmodelloss' + str(i) + '.png')


if __name__ == '__main__':
    x_train = np.load('./data/_3_1/data.npy')
    y_train = np.load('./data/_3_1/label.npy')

    x_train = np.expand_dims(x_train, axis=2)
    # reshape
    y_train = y_train.reshape([y_train.shape[0], 1])
    print(y_train.shape)
    y_train = keras.utils.to_categorical(y_train, NUM_LABLES)

    # cut
    # train_num = x_train.shape[0] % BATCH_SIZE

    # if train_num != 0:
    #     x_train = x_train[0:-train_num]
    #     y_train = y_train[0:-train_num]

    # print('x_train: ', x_train.shape)

    # print('y_train: ', y_train.shape)

    seed = 9
    np.random.seed(seed)

    cnn_model = model_Sequential(
        x_train=x_train, batch_size=BATCH_SIZE, num_lables=NUM_LABLES)
    cnn_model.compile(optimizer=keras.optimizers.Adam(clipnorm=1., lr=0.0001),
                      loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cvscores = []
    rescorces = []
    f1scorces = []
    n = 0
    for train, test in kfold.split(x_train, np.argmax(y_train, axis=1)):
        # cut
        train_num = x_train[train].shape[0] % BATCH_SIZE

        if train_num != 0:
            x = x_train[train][0:-train_num]
            y = y_train[train][0:-train_num]

        test_num = x_train[test].shape[0] % BATCH_SIZE
        if test_num != 0:
            x_ = x_train[test][0:-test_num]
            y_ = y_train[test][0:-test_num]

        print('x_train: ', x.shape)

        print('y_train: ', y.shape)
        history = cnn_model.fit(x, y, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1)
        n += 1
        pltfig(history=history, i=n)
        scores = cnn_model.evaluate(x_, y_, verbose=1)
        cvscores.append(scores[1] * 100)
        # 测试集验证
        # y_test_predict = cnn_model.predict(x_, batch_size=BATCH_SIZE)
        # y_test_predict = np.array(y_test_predict)
        # y_test_predict = np.argmax(y_test_predict, axis=1)

        # rs = recall_score(np.argmax(y_, axis=1), y_test_predict)
        # rescorces.append(rs * 100)
        
        # fs = f1_score(np.argmax(y_, axis=1), y_test_predict)
        # f1scorces.append(fs * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(rescorces), np.std(rescorces)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(f1scorces), np.std(f1scorces)))


    # train_num = x_train.shape[0] % BATCH_SIZE

    # if train_num != 0:
    #     x = x_train[0:-train_num]
    #     y = y_train[0:-train_num]
    
    # model_save_path = save_model_dir + 'tm.h5'

    # # 检查点回调
    # # checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_best_only=True,
    # #                                                 save_weights_only=True, monitor='val_accuracy')

    # # # earlystop 回调
    # # early_stopping_cb = keras.callbacks.EarlyStopping(
    # #     patience=PATIENCE, restore_best_weights=True, monitor='val_loss')

    # # 编译
    # cnn_model.compile(optimizer=keras.optimizers.Adam(clipnorm=1., lr=0.0001),
    #                   loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    # # 训练
    # history = cnn_model.fit(x_train, y_train, epochs=EPOCH,
    #                         batch_size=BATCH_SIZE, verbose=1)

    # # early_stopping_epoch = early_stopping_cb.stopped_epoch - PATIENCE + 1
    # # print('Early stopping epoch: ' + str(early_stopping_epoch))
    # # 画图
    # font2 = {'size': 10}
    # plt.figure()
    # # plt.subplot(2,2,1)
    # plt.title('tm-acc')
    # plt.plot(history.history['accuracy'],
    #          label='train accuracy', linewidth=1.5)
    # plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
    # # plt.plot(history.history['val_accuracy'],
    # #          label='val accuracy', linewidth=1.5)
    # plt.legend(prop=font2)
    # plt.savefig('./data/_3_1/testmodel.png')
    # plt.show()

    # # plt.figure()
    # # plt.title(f'tm-loss')
    # # plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
    # # plt.plot(history.history['val_loss'], label='val loss', linewidth=1.5)
    # # plt.legend(prop=font2)
    # # plt.show()
    # # plt.savefig(save_dir + 'tm-loss.png')

