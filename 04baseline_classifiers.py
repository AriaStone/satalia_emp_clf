import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


def get_data_to_feed_baseline():
    job_tf = []  
    job_tfidf = []
    for job in list(jobBased_tf_vector.keys()):
        temp_tf = []
        temp_tfidf = []
        for word in list(word_count.keys()):
            temp_tf.append(jobBased_tf_vector[job][word])
            temp_tfidf.append(jobBased_tfidf_vector[job][word])
        job_tf.append(temp_tf)
        job_tfidf.append(temp_tfidf)
    
    user_tf_train = []
    user_tfidf_train = []
    user_tf_test = []
    user_tfidf_test = [] 
    for user in list(userBased_tf_vector_train.keys()):
        temp_tf_train = []
        temp_tf_test = []
        temp_tfidf_train = []
        temp_tfidf_test = []
        for word in list(word_count.keys()):
            temp_tf_train.append(userBased_tf_vector_train[user][word])
            temp_tf_test.append(userBased_tf_vector_test[user][word])
            temp_tfidf_train.append(userBased_tfidf_vector_train[user][word])
            temp_tfidf_test.append(userBased_tfidf_vector_test[user][word]) 
        user_tf_train.append(temp_tf_train)
        user_tfidf_train.append(temp_tfidf_train)
        user_tf_test.append(temp_tf_test)
        user_tfidf_test.append(temp_tfidf_test)
    return job_tf, job_tfidf, user_tf_train, user_tfidf_train, user_tf_test, user_tfidf_test

def cos_sim(a,b):
    result = np.sum(a * b)/((np.linalg.norm(a)) * (np.linalg.norm(b)))
    return result

def apply_cossim():
    tf_cossim = []
    for i in range(len(user_tf_train)):
        temp = []
        for j in range(len(job_tf)):
            temp.append(cos_sim(np.array(user_tf_train[i]),np.array(job_tf[j])))
        tf_cossim.append(temp)
            
    tfidf_cossim = []
    for i in range(len(user_tfidf_train)):
        temp = []
        for j in range(len(job_tfidf)):
            temp.append(cos_sim(np.array(user_tfidf_train[i]),np.array(job_tfidf[j])))
        tfidf_cossim.append(temp)
    return tf_cossim, tfidf_cossim


    
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
    
#plt.figure()
#plot_confusion_matrix(z_cf_tfidf, classes=class_names, normalize=True,
#                      title='Confusion matrix for KL-divergence tfidf-NB')
#
#plt.show()  

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def cal_acc_prf1():
    z_precision = precision_score(z_true,z_predictions_tfidf,average='weighted')
    z_recall =recall_score(z_true,z_predictions_tfidf,average = 'weighted')
    z_f1 = f1_score(z_true,z_predictions_tfidf,average = 'weighted')
    z_acc = accuracy_score(z_true,z_predictions_tfidf)
    return z_precision, z_recall, z_f1,z_acc


################codes below are extremely bad organised###################
###########if needed please contact the email lilapoloy@gmail.com. Then I'll try to make it cleaner. And people could directly use the code######

def jobBased_KLdiv():
    KL = scipy.stats.entropy(x, y) 
    z_jb_tf = []
    z_jb_tfidf = []
    z_ub_tf = []
    z_ub_tfidf = []
    smooth = 0.0001
    for job in jobBased_tf_vector.keys():
        temp = []
        for word in word_count.keys():   
            temp.append(jobBased_tf_vector[job][word] + smooth)
        z_jb_tf.append(temp)
    for job in jobBased_tfidf_vector.keys():
        temp = []
        for word in word_count.keys():
            temp.append(jobBased_tfidf_vector[job][word]+ smooth)
        z_jb_tfidf.append(temp)
        
       
    for user in userBased_tf_vector_train.keys():
        temp = [] 
        for word in word_count.keys():
            temp.append(userBased_tf_vector_train[user][word]+ smooth)
        z_ub_tf.append(temp)
    for user in userBased_tfidf_vector_train.keys():
        temp = [] 
        for word in word_count.keys():
            temp.append(userBased_tfidf_vector_train[user][word]+ smooth)
        z_ub_tfidf.append(temp)
    
    z_tf = []    
    for i in range(59):
        temp = []
        for j in range(8):
            temp.append(scipy.stats.entropy(z_ub_tf[i],z_jb_tf[j]))
        z_tf.append(temp)
        
    z_tfidf = []    
    for i in range(59):
        temp = []
        for j in range(8):
            temp.append(scipy.stats.entropy(z_ub_tfidf[i],z_jb_tfidf[j]))
        z_tfidf.append(temp)
    z_predictions_tf = [np.argmax(p) for p in z_tf]
    z_predictions_tfidf = [np.argmax(p) for p in z_tfidf]
    
    z_job = dict()
    z_job['Software Developer'] = 0
    z_job['Optimisation'] = 1
    z_job['Geospatial Developer'] = 2
    z_job['Project Navigation'] = 3
    z_job['Data Science'] = 4
    z_job['Executive'] = 5
    z_job['Human Operations'] = 6
    z_job['Business Development'] = 7
    
    z_true = []
    
    for i in range(60):
        if user_skills.user_id[i] in userBased_tf_vector_test.keys():
            z_true.append(z_job[user_skills.primary_discipline[i]])
    
    
    z_cf_tf = np.zeros((8,8))
    z_cf_tfidf = np.zeros((8,8))
    for i in range(59):
        z_cf_tf[z_true[i]][z_predictions_tf[i]] += 1
        z_cf_tfidf[z_true[i]][z_predictions_tfidf[i]] += 1
    return 0



def code_bins_orgnised_needed():
    reverse_job_mapping = dict()
    for i in range(len(job_mapping)):
        reverse_job_mapping[i] = job_mapping[i]
    max_coss_tf = []
    for i in range(len(tf_cossim)):
        max_cos = 0
        for j in range(len(tf_cossim[i])):
            if tf_cossim[i][j] > tf_cossim[i][max_cos]:
                max_cos = j
        max_coss_tf.append(max_cos)
    max_coss_tfidf = []
    for i in range(len(tfidf_cossim)):
        max_cos = 0
        for j in range(len(tfidf_cossim[i])):
            if tfidf_cossim[i][j] > tfidf_cossim[i][max_cos]:
                max_cos = j
        max_coss_tfidf.append(max_cos)
        
    for i in range(len(max_coss_tf)):
        max_coss_tf[i] = reverse_job_mapping[max_coss_tf[i]]
        max_coss_tfidf[i] = reverse_job_mapping[max_coss_tfidf[i]]
    
    user_skills = user_skills[user_skills.user_id.isin(list(userBased_tf_vector_train.keys()))]
    max_coss_labels = list(user_skills.primary_discipline)
    tf_acc = 0
    for i in range(59):
        if max_coss_labels[i] == max_coss_tf[i]:
            tf_acc += 1
    tf_acc /= 59
    
    tfidf_acc = 0
    for i in range(59):
        if max_coss_labels[i] == max_coss_tfidf[i]:
            tfidf_acc += 1
    tfidf_acc /= 59
    
    re_job_mapping = dict()
    i = 0
    for job in list(jobBased_tf_vector.keys()):
        re_job_mapping[job] = i
        i += 1
    confusion_tf = np.zeros((8,8))
    for i in range(59):
        confusion_tf[re_job_mapping[max_coss_labels[i]]][re_job_mapping[max_coss_tf[i]]] += 1
    confusion_tfidf = np.zeros((8,8))
    for i in range(59):
        confusion_tfidf[re_job_mapping[max_coss_labels[i]]][re_job_mapping[max_coss_tfidf[i]]] += 1
    
    confusion_tf_prop = np.zeros((8,8))
    for i in range(59):
        for j in range(8):
            confusion_tf_prop[re_job_mapping[max_coss_labels[i]]][j]  += (tf_cossim[i][j]/sum(tf_cossim[i]))
    
    confusion_tfidf_prop = np.zeros((8,8))
    for i in range(59):
        for j in range(8):
            confusion_tfidf_prop[re_job_mapping[max_coss_labels[i]]][j]  += (tfidf_cossim[i][j]/sum(tfidf_cossim[i]))
    return 0




