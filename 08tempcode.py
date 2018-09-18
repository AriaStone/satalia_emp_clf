import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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

#vocab_size = 20002

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
#                    batch_size=512,
#                    validation_data=(test_data, test_labels),
#                    verbose=1)
#
#history_dict = history.history
#history_dict.keys()
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#test_loss, test_acc = model.evaluate(test_data, test_labels)
#print('Test accuracy:', test_acc)
#predictions_prob = model.predict(test_data)
#predictions = [np.argmax(p) for p in predictions_prob]
#acc_values = history_dict['acc']
#val_acc_values = history_dict['val_acc']
#epochs = 100
#
#plt.plot( acc,  label='Training acc')
#plt.plot( val_acc,  label='Validation acc')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#
#plt.show() 

#test["label_prd"] = predictions
#confusion_mlp = np.zeros((8,8))
#for i in test.index:
#    confusion_mlp[test.label[i]][test.label_prd[i]] += 1
    
#user_prd = dict()
#user_test = dict()
#for user in test.standup_id.unique():
#    temp1 = test[test.standup_id == user]
#    user_test[user] = list(temp1.label.unique())[0]
#    job_label_prd = dict() 
#    for job_prd in temp1.label_prd.unique():
#        job_label_prd[job_prd] = len(temp1[temp1.label_prd == job_prd])
#    user_prd[user] = job_label_prd
#
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

########################################
#data_dim = 16
#timesteps = 8
#
#model = keras.Sequential()
#model.add(keras.layers.Embedding(vocab_size , output_dim=data_dim))
#model.add(keras.layers.LSTM(32, return_sequences=True,
#               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(keras.layers.LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(keras.layers.LSTM(32))
#model.add(keras.layers.Dense(8, activation= tf.nn.softmax))
#model.summary()
#
#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

#history = model.fit(train_data, train_labels1, batch_size=2048, epochs=30,validation_data=(test_data, test_labels1),verbose = 1)
#history_dict = history.history
#history_dict.keys()
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#test_loss, test_acc = model.evaluate(test_data, test_labels1)
#print('Test accuracy:', test_acc)
#predictions_prob = model.predict(test_data)
#predictions = [np.argmax(p) for p in predictions_prob]


##load LSTM results firstly ###########3
    
#job_msgs = dict()
#for l in list(train.label.unique()):
#    job_msgs[l] = len(train[train.label == l])
#    
#job_msg_prop = []
#for i in range(8):
#    job_msg_prop.append(job_msgs[i])
#    
#z = sum(job_msg_prop)
#
#for i in range(8):
#    job_msg_prop[i] /=z
#adjust_labels = job_msg_prop
##############################################################################
#predictions_prob1 = np.zeros((149970,8))
#for i in range(149970):
#    predictions_prob1[i] = (predictions_prob[i] - adjust_labels)/adjust_labels
#
#predictions1 = []
#for i in predictions_prob1:
#    predictions1.append(np.argmax(i))
#    
#test.label_prd = predictions1
#
#test["label_prd"] = predictions1
#confusion_lstm_msgbased = np.zeros((8,8))
#for i in test.index:
#    confusion_lstm_msgbased[test.label[i]][test.label_prd[i]] += 1
#    
#user_prd = dict()
#user_test = dict()
#for user in test.standup_id.unique():
#    temp1 = test[test.standup_id == user]
#    user_test[user] = list(temp1.label.unique())[0]
#    job_label_prd = dict() 
#    for job_prd in temp1.label_prd.unique():
#        job_label_prd[job_prd] = len(temp1[temp1.label_prd == job_prd])
#    user_prd[user] = job_label_prd
#
#user_prd_labels = dict()
#for user in user_prd.keys():
#    user_prd_labels[user] = max(user_prd[user],key = user_prd[user].get)
#user_acc = 0
#for user in user_prd.keys():
#    if user_prd_labels[user]==user_test[user]:
#        user_acc += 1
#user_acc /= 59
#confusion_lstm = np.zeros((8,8))
#for user in user_prd.keys():
#    confusion_lstm[user_test[user]][user_prd_labels[user]] += 1
#user_msg_acc = 0
#for i in range(149970):
#    if predictions[i] == predictions1[i]:
#        user_msg_acc += 1
#user_msg_acc /= 149970
#    
#a_true = list(test.label)
#a_predict = list(test.label_prd)
#
#for user in user_test.keys():
#    a_true.append(user_test[user])
#    a_predict.append(user_prd_labels[user])
    
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#a_precision = precision_score(a_true,a_predict,average='weighted')
#a_recall =recall_score(a_true,a_predict,average = 'weighted')
#a_f1 = f1_score(a_true,a_predict,average = 'weighted')
#a_acc = accuracy_score(a_true,a_predict)
    
  #############################################################################  
#class_names = []
#for i in dict_jobs.keys():
#    class_names.append(i)
plot_confusion_matrix(confusion_lstm_msgbased, classes=class_names, normalize=True,
                      title='Confusion matrix for adjusted MLP')


#confusion_tf1=np.zeros((8,8))
#confusion_tfidf1=np.zeros((8,8))
#a = dict()
#a[0]=0
#a[1]=5
#a[2]=7
#a[3]=3
#a[4]=4
#a[5]=1
#a[6]=2
#a[7]=6
#for i in range(8):
#    for j in range(8):
#        confusion_tf1[a[i]][a[j]] = confusion_tf[i][j]
#for i in range(8):
#    for j in range(8):
#        confusion_tfidf1[a[i]][a[j]] = confusion_tfidf[i][j]    


#a_precision_tf = precision_score(max_coss_labels,max_coss_tf, average='weighted')
#a_precision_tfidf = precision_score(max_coss_labels,max_coss_tfidf, average='weighted')
#a_recall_tf = recall_score(max_coss_labels,max_coss_tf, average='weighted')
#a_recall_tfidf = recall_score(max_coss_labels,max_coss_tfidf, average='weighted')
#
#a_acc_tf = accuracy_score(max_coss_labels,max_coss_tf)
#a_acc_tfidf =accuracy_score(max_coss_labels,max_coss_tfidf)

#a_f1_tf = f1_score(max_coss_labels,max_coss_tf, average='weighted')
#a_f1_tfidf = f1_score(max_coss_labels,max_coss_tfidf, average='weighted')

