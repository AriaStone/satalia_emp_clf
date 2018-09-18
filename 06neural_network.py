import tensorflow as tf
from tensorflow import keras

import numpy as np

#train = pd.concat([train_private_wordid,train_public_wordid],ignore_index = True)
#test = pd.concat([test_private_wordid,test_public_wordid],ignore_index = True)

def mapping_job_to_num_for_softmaxLabel():
    jobs = list(train.primary_discipline.unique())
    dict_jobs = dict()
    i = 0
    for job in jobs:
        dict_jobs[job] = i
        i += 1
    return dict_jobs
def get_train_test_sentenceCLF():
    dict_jobs =  mapping_job_to_num_for_softmaxLabel()
    test["label"] = 0
    for discipline in list(test.primary_discipline.unique()): 
        test.label[test.primary_discipline == discipline] = dict_jobs[discipline]
    train["label"] = 0
    for discipline in list(train.primary_discipline.unique()): 
        train.label[train.primary_discipline == discipline] = dict_jobs[discipline]
        
    train = train[['message','standup_id','label']]
    test = test[['message','standup_id','label']]
    return train, test

#def get_train_test_userTestCLF():
#    
#    return trainUserText, testUserText

def data_to_feed_nns():
    #train_x = list(train.message)
    #train_y = list(train.label)
    #test_x = list(test.message)
    #test_y = list(test.label)
    #train_labels_multi = np.zeros()
    
    #train_data = keras.preprocessing.sequence.pad_sequences(train_x,padding='post',maxlen=30)
    #test_data = keras.preprocessing.sequence.pad_sequences(test_x,padding='post',maxlen=30)
    #train_labels = train_y[:]
    #test_labels = test_y[:]
    #
    #vocab_size = len(word_count_index)
    return 0

#train_x = list(train.message)
#train_y = list(train.label)
#test_x = list(test.message)
#test_y = list(test.label)
##train_labels_multi = np.zeros()
#
#train_data = keras.preprocessing.sequence.pad_sequences(train_x,padding='post',maxlen=30)
#test_data = keras.preprocessing.sequence.pad_sequences(test_x,padding='post',maxlen=30)
#train_labels = train_y[:]
#test_labels = test_y[:]
#
#vocab_size = len(word_count_index)

#train_labels1 = np.zeros((599982,8))
#test_labels1 = np.zeros((149970,8))
#for i in range(len(train_labels1)):
#    train_labels1[i][train_labels[i]] = 1
#for i in range(len(test_labels1)):
#    test_labels1[i][test_labels[i]] = 1

import tensorflow as tf
from tensorflow import keras

import numpy as np
def lstm_clf():
    data_dim = 16
    timesteps = 8
    
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size , output_dim=data_dim))
    model.add(keras.layers.LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(keras.layers.LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(8, activation= tf.nn.softmax))
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    history = model.fit(train_data, train_labels1, batch_size=1024, epochs=100,validation_data=(test_data, test_labels1),verbose = 1)
    history_dict = history.history
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    return 0 

def cnn_clf(train_x,train_y, test_x, test_y):
    return 0

#model = keras.Sequential()
#model.add(keras.layers.Embedding(vocab_size, 128))
#model.add(keras.layers.Conv1D(64, 3, activation='relu'))
#model.add(keras.layers.Conv1D(64, 3, activation='relu'))
#model.add(keras.layers.MaxPooling1D(3))
#model.add(keras.layers.Dropout(0.5))
#model.add(keras.layers.Conv1D(32, 3, activation='relu'))
#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dropout(0.5))
#model.add(keras.layers.Dense(8, activation= tf.nn.softmax))
#model.summary()
#
#model.compile(loss='sparse_categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
#history = model.fit(train_data,
#                    train_labels,
#                    epochs=100,
#                    batch_size=200,
#                    validation_data=(test_data, test_labels),
#                    verbose=1)

#history_dict = history.history
#history_dict.keys()
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#test_loss, test_acc = model.evaluate(test_data, test_labels)
#print('Test accuracy:', test_acc)
#predictions_prob = model.predict(test_data)
#predictions = []
#for i in predictions_prob:
#    predictions.append(np.argmax(i))
#test["label_prd"] = predictions
#user_prd = dict()
#user_test = dict()
#for user in test.standup_id.unique():
#    temp1 = test[test.standup_id == user]
#    user_test[user] = list(temp1.label.unique())[0]
#    job_label_prd = dict() 
#    for job_prd in temp1.label_prd.unique():
#        job_label_prd[job_prd] = len(temp1[temp1.label_prd == job_prd])
#    user_prd[user] = job_label_prd

#user_prd_labels = dict()
#for user in user_prd.keys():
#    user_prd_labels[user] = max(user_prd[user],key = user_prd[user].get)
#user_acc = 0
#for user in user_prd.keys():
#    if user_prd_labels[user]==user_test[user]:
#        user_acc += 1
#user_acc /= 59
#confusion_cnn = np.zeros((8,8))
#for user in user_prd.keys():
#    confusion_cnn[user_test[user]][user_prd_labels[user]] += 1
#confusion_cnn_msgbased = np.zeros((8,8))
#for i in test.index:
#    print(i)
#    confusion_cnn_msgbased[test.label[i]][test.label_prd[i]] += 1
    

def mlp_clf(train_x,train_y, test_x, test_y ):


    #model = keras.Sequential()
    #model.add(keras.layers.Embedding(vocab_size, 128))
    #model.add(keras.layers.GlobalAveragePooling1D())
    #model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    #model.add(keras.layers.Dropout(0.5))
    ##model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    ##model.add(keras.layers.Dropout(0.5))
    ##model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    ##model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    #model.add(keras.layers.Dense(8, activation=tf.nn.softmax))
    #
    #model.summary()
    #
    #model.compile(optimizer='rmsprop',
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])
    #
    #
    #
    #history = model.fit(train_data,
    #                    train_labels,
    #                    epochs=100,
    #                    batch_size=200,
    #                    validation_data=(test_data, test_labels),
    #                    verbose=1)
    
    #history_dict = history.history
    #history_dict.keys()
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    #test_loss, test_acc = model.evaluate(test_data, test_labels)
    #print('Test accuracy:', test_acc)
    #predictions_prob = model.predict(test_data)
    #predictions = [np.argmax(p) for p in predictions_prob]
    train_data = keras.preprocessing.sequence.pad_sequences(train_x,padding='post',maxlen=30)
    test_data = keras.preprocessing.sequence.pad_sequences(test_x,padding='post',maxlen=30)
    train_labels = train_y[:]
    test_labels = test_y[:]
    
    vocab_size = len(word_count_index)
    
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 256))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(8, activation=tf.nn.softmax))
    
    model.summary()
    
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    
    history = model.fit(train_data,
                        train_labels,
                        epochs=500,
                        batch_size=20000,
                        validation_data=(test_data, test_labels),
                        verbose=1)
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Test accuracy:', test_acc)
    predictions_prob = model.predict(test_data)
    predictions = np.argmax(predictions_prob)
    
    history_dict = history.history
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.clf()   # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    return predictions_prob, predictions, acc, val_acc
#predictions_prob, predictions, acc, val_acc = mlp_clf(train_x,train_y, test_x, test_y )