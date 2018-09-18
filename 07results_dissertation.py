import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
########################################################################################
#total msg ，channels，days
#z = dict()
#for i in user_skills.primary_discipline.unique():
#    z[i] = 0
#for i in user_skills.primary_discipline:
#    z[i] += 1
def z_public(z_public):
    z1_public = []
    for i in range(len(z_public)):
        print(i)
        z1_public.append(z_public[i][0:10])
    return z1_public



def z2_public(z1_public):
    z2_public = dict()
    for i in z1_public.unique():
        z2_public[i] = 0
    for i in z1_public:
        z2_public[i] += 1
    return z2_public
#z2_private = z2_public(z1_private)
#z2_public = z2_public(z1_public)

def z3(z2_public):
    z3_public = []
    for day in z2_public.keys():
        z3_public.append(z2_public[day])
    return z3_public
#z3_private = z3(z2_private)

def plot_avg_msg_per_day(z3):
    sns.kdeplot(z3_public,label = "Daily Messages in Public Channels",shade = True)
    sns.kdeplot(z3_private,label = "Daily Messages in Private Channels",shade = True)
    plt.legend()
#########################################################################################
    #avg msg per day
#satalia_public = pd.read_csv("G:\\satalia\\Slack_Public_19062018.csv", encoding="utf-8-sig",index_col = 0)
#satalia_private = pd.read_csv("G:\\satalia\\Slack_Private_19062018.csv", encoding="utf-8-sig",index_col = 0)
#user_mapping = pd.read_csv("G:\\satalia\\user_mapping.csv")
#user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
#
#user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
#user_mapping = user_mapping.dropna()
#user_skills = user_skills[user_skills.user_id.isin(user_mapping.standup_id)]
#user_mapping.rename(columns = {'standup_id':'user_id'},inplace = True)
#user_skills = pd.merge(user_skills,user_mapping,on = 'user_id',how = 'left')
def get_time_user_in_satalia(user_skills):
    #generate data to be used in next few steps
    msg_time = []
    for user in user_skills.slack_id:
        temp = []    
#        user = 'U02EENFGK'
        temp.append(user)
        temp.append(int(user_skills.user_id[user_skills.slack_id == user]))
        temp.append(list(user_skills.primary_discipline[user_skills.slack_id == user])[0])
        if len(list(satalia_private[satalia_private.user_id==user].timestamp)) < 2:
            temp.append(0)
            temp.append(0)
            temp.append(0)
        else:
            temp.append(list(satalia_private[satalia_private.user_id==user].timestamp)[0])
            temp.append(list(satalia_private[satalia_private.user_id==user].timestamp)[-1])
            startdate = pd.to_datetime(temp[3])
            enddate = pd.to_datetime(temp[4])
            days = str(enddate - startdate)[0:-13]
            temp.append(int(days))
        private_msg_num = len(satalia_private[satalia_private.user_id == user])
        public_msg_num = len(satalia_public[satalia_public.userid == user])
        msg_num = private_msg_num +  public_msg_num
        temp.append(msg_num)
        msg_time.append(temp)
    return msg_time
#msg_time = get_time_user_in_satalia(user_skills)

def plot_working_days():
    #working days
    workingdays = []
    for i in range(60):
        workingdays.append(msg_time[i][-2])
    fig_1 = sns.distplot(workingdays,bins = None)
    return workingdays
def plot_avg_msg_len():
    # average number of messages of users
    avg_msg_len = []
    for i in range(60):
        temp = []
        temp.append(msg_time[i][2])
        temp.append(msg_time[i][6]/(msg_time[i][5]+1))
        avg_msg_len.append(temp)
    avg_len = [i[1] for i in avg_msg_len]
    fig_2 = sns.distplot(avg_len,bins = None)
    return avg_len

def plot_avg_msg_per_job():
    job_title = dict()
    for i in user_skills.primary_discipline.unique():
        job_title[i] = [0,0]
    for i in range(60):
        job_title[msg_time[i][2]][0] += msg_time[i][6]/(msg_time[i][5] + 1)
        job_title[msg_time[i][2]][1] += 1
    for i in job_title.keys():
        job_title[i][0] = job_title[i][0]/job_title[i][1]
    return job_title
##########################################################################
#average length of messages
def tag_len_to_sets(train_public_wordid):
    train_public_wordid = train_public_wordid.reset_index(drop=True)
    avg_msg_len_public = []
    len_train_public_wordid = len(train_public_wordid)
    for i in range(len_train_public_wordid):
        print(str(i) + "/" + str(len_train_public_wordid))
        avg_msg_len_public.append(len(train_public_wordid.message[i]))
    z = pd.DataFrame(avg_msg_len_public,columns = [""])
    train_public_wordid["len"] = avg_msg_len_public
    return train_public_wordid

def reload_data_1():
    train_private_wordid = tag_len_to_sets(train_private_wordid)
    train_public_wordid = tag_len_to_sets(train_public_wordid) 
    test_private_wordid = tag_len_to_sets(test_private_wordid)   
    test_public_wordid = tag_len_to_sets(test_public_wordid)
    
    user_mapping = pd.read_csv("G:\\satalia\\user_mapping.csv")
    user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
    
    user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
    user_mapping = user_mapping.dropna()
    user_skills = user_skills[user_skills.user_id.isin(user_mapping.standup_id)]
    user_mapping.rename(columns = {'standup_id':'user_id'},inplace = True)
    user_skills = pd.merge(user_skills,user_mapping,on = 'user_id',how = 'left')
    return 0

def avg_msg_len_userbased_channels():
    avg_msg_len_userbased_public = dict()
    avg_msg_len_userbased_private = dict()
    for user in list(user_skill.user_id.unique()):
        avg_msg_len_userbased_public[user] = [0,0]
        avg_msg_len_userbased_private[user] = [0,0]
        
    for user in list(user_skill.user_id.unique()):
        temp_data = test_public_wordid[test_public_wordid.standup_id == user]
        for msg_len in temp_data.len:
            print(str(user) + ":" + str(avg_msg_len_userbased_public[user][1]))
            avg_msg_len_userbased_public[user][0] += msg_len
            avg_msg_len_userbased_public[user][1] += 1
    for user in list(user_skill.user_id.unique()):
        temp_data1 = train_public_wordid[train_public_wordid.standup_id == user]
        for msg_len in temp_data1.len:
            print(str(user) + ":" + str(avg_msg_len_userbased_public[user][1]))
            avg_msg_len_userbased_public[user][0] += msg_len
            avg_msg_len_userbased_public[user][1] += 1
    for user in list(user_skill.user_id.unique()):        
        temp_data = test_private_wordid[test_private_wordid.standup_id == user]
        for msg_len in temp_data.len:
            print(str(user) + ":" + str(avg_msg_len_userbased_private[user][1]))
            avg_msg_len_userbased_private[user][0] += msg_len
            avg_msg_len_userbased_private[user][1] += 1
        temp_data = train_private_wordid[train_private_wordid.standup_id == user]
        for msg_len in temp_data.len:
            print(str(user) + ":" + str(avg_msg_len_userbased_private[user][1]))
            avg_msg_len_userbased_private[user][0] += msg_len
            avg_msg_len_userbased_private[user][1] += 1
    return avg_msg_len_userbased_public,avg_msg_len_userbased_private

def avg_msg_len_jobbased_channels():
    avg_msg_len_jobbased_public = dict()
    avg_msg_len_jobbased_private = dict()
    for user in list(user_skill.primary_discipline.unique()):
        avg_msg_len_jobbased_public[user] = [0,0]
        avg_msg_len_jobbased_private[user] = [0,0]
        
    for user in list(user_skill.primary_discipline.unique()):
        temp_data = test_public_wordid[test_public_wordid.primary_discipline == user]
        for msg_len in temp_data.len:
            print(str(user) + ":" + str(avg_msg_len_jobbased_public[user][1]))
            avg_msg_len_jobbased_public[user][0] += msg_len
            avg_msg_len_jobbased_public[user][1] += 1
    for user in list(user_skill.primary_discipline.unique()):
        temp_data1 = train_public_wordid[train_public_wordid.primary_discipline == user]
        for msg_len in temp_data1.len:
            print(str(user) + ":" + str(avg_msg_len_jobbased_public[user][1]))
            avg_msg_len_jobbased_public[user][0] += msg_len
            avg_msg_len_jobbased_public[user][1] += 1
    for user in list(user_skill.primary_discipline.unique()):        
        temp_data = test_private_wordid[test_private_wordid.primary_discipline== user]
        for msg_len in temp_data.len:
            print(str(user) + ":" + str(avg_msg_len_jobbased_private[user][1]))
            avg_msg_len_jobbased_private[user][0] += msg_len
            avg_msg_len_jobbased_private[user][1] += 1
        temp_data = train_private_wordid[train_private_wordid.primary_discipline == user]
        for msg_len in temp_data.len:
            print(str(user) + ":" + str(avg_msg_len_jobbased_private[user][1]))
            avg_msg_len_jobbased_private[user][0] += msg_len
            avg_msg_len_jobbased_private[user][1] += 1
    return avg_msg_len_jobbased_public,avg_msg_len_jobbased_private
#avg_msg_len_jobbased_public,avg_msg_len_jobbased_private = avg_msg_len_jobbased_channels()
def get_message_len_overall():
    message_len_overall = []
    for i in list(test_public_wordid.len):
        message_len_overall.append(i)
    for i in list(test_private_wordid.len):
        message_len_overall.append(i)
    for i in list(train_public_wordid.len):
        message_len_overall.append(i)
    for i in list(train_private_wordid.len):
        message_len_overall.append(i)
    return message_len_overall
def get_avg_msg_len_jobbased():
    for user in avg_msg_len_userbased_private.keys():
        avg_msg_len_userbased_private[user][0] = avg_msg_len_userbased_private[user][0]/(avg_msg_len_userbased_private[user][1]+1)
    for user in avg_msg_len_userbased_public.keys():
        avg_msg_len_userbased_public[user][0] = avg_msg_len_userbased_public[user][0]/(avg_msg_len_userbased_public[user][1]+1)
        
    for user in avg_msg_len_jobbased_private.keys():
        avg_msg_len_jobbased_private[user][0] = avg_msg_len_jobbased_private[user][0]/(avg_msg_len_jobbased_private[user][1]+1)
    for user in avg_msg_len_jobbased_public.keys():
        avg_msg_len_jobbased_public[user][0] = avg_msg_len_jobbased_public[user][0]/(avg_msg_len_jobbased_public[user][1]+1)
    #
    for user in list(user_skill.user_id.unique()):
        avg_msg_len_userbased_private[user][0] = avg_msg_len_userbased_private[user][0]/(avg_msg_len_userbased_private[user][1]+1)
        avg_msg_len_userbased_public[user][0] = avg_msg_len_userbased_public[user][0]/(avg_msg_len_userbased_public[user][1] +1)
    return 0
def get_avg_msg_len_userbased():
    a_private = []
    for user in avg_msg_len_userbased_private.keys():
        a_private.append(avg_msg_len_userbased_private[user][0])
    a_public = []
    for user in avg_msg_len_userbased_public.keys():
        a_public.append(avg_msg_len_userbased_public[user][0])
    return 0
#sns.kdeplot(a_private,label = "User Avg. Message Len(Private)",shade = True)
#sns.kdeplot(a_public,label = "User Avg. Message Len(Public)",shade = True)
#plt.legend()
#for user in avg_msg_len:
#    avg_msg_jobs[user[0]][0] += user[1]
#    avg_msg_jobs[user[0]][1] += 1
#for job in avg_msg_jobs.keys():
#    avg_msg_jobs[job][0] /= avg_msg_jobs[job][1]

#avg_msg_len_private = []
#for i in train_private_wordid.index:
#    print("i" + " train")
#    temp = []
#    temp.append(len(train_private_wordid.message[i]))
#    temp.append(train_private_wordid.standup_id[i])
#    temp.append(train_private_wordid.primary_discipline[i])
#    avg_msg_len_private.append(temp)
#for i in test_private_wordid.index:
#    print("i" + " test")
#    tmep = []
#    temp.append(len(test_private_wordid.message[i]))
#    temp.append(test_private_wordid.standup_id[i])
#    temp.append(test_private_wordid.primary_discipline[i])
#    avg_msg_len_private.append(temp)



#for i in test_public_wordid.index:
#    print(str(i) + " test")
#    tmep = []
#    temp.append(len(test_public_wordid.message[i]))
#    temp.append(test_public_wordid.standup_id[i])
#    temp.append(test_public_wordid.primary_discipline[i])
#    avg_msg_len_public.append(temp)    
   
#z = list(train_public_wordid.index)
#for i in z:
#    print(str(i) + " train")
#    train_public_wordid.message[i] = len(train_public_wordid.message[i])
    
#plt.clf()   # clear figure
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



# Plot non-normalized confusion matrix
#class_names = []
#for i in range(8):
#    class_names.append(reverse_job_mapping[i])


#plt.figure()
#plot_confusion_matrix(confusion_tfidf, classes=class_names,normalize=True,
#                      title='Confusion matrix for tfidf-Naive Bayes')
## Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()   

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

    
    
    
    
    
    
    
    
    
    
    