import pandas as pd
import numpy as np
#####compared with data_preprocess.py this file splits and samples the messages into train and test set
def get_train_test_dataset():
    temp_public_dual = pd.read_csv("G:\\satalia\\temp_public_dual.csv", encoding="utf-8-sig") # 25 to 19
    temp_private_dual = pd.read_csv("G:\\satalia\\temp_private_dual.csv", encoding="utf-8-sig")
    user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
    temp_public_dual = temp_public_dual.dropna(subset = ['message'])
    temp_private_dual = temp_private_dual.dropna(subset = ['message'])
    
    train_ratio = 0.8
    train_private = temp_private_dual.sample(frac = train_ratio)
    train_public = temp_public_dual.sample(frac = train_ratio)
    
    test_ratio = 1 - train_ratio
    test_private = temp_private_dual.drop(list(train_private.index))
    test_public = temp_public_dual.drop(list(train_public.index))
    return train_private, train_public, test_private, test_public

def add_label_to_msg(train_private, train_public, test_private, test_public):
    
    test_private = test_private[['message','user_id']]
    test_public = test_public[['message','userid']]
    train_private = train_private[['message','user_id']]
    train_public = train_public[['message','userid']]
        
    
    user_mapping.rename(columns = {'slack_id': 'user_id'}, inplace = True)
    test_public.rename(columns = {'userid':'user_id'},inplace = True)
    train_public.rename(columns = {'userid':'user_id'},inplace = True)
    
    test_private = pd.merge(test_private,user_mapping, on ='user_id', how = 'left')
    test_public = pd.merge(test_public,user_mapping, on ='user_id', how = 'left')
    train_private = pd.merge(train_private,user_mapping, on ='user_id', how = 'left')
    train_public = pd.merge(train_public,user_mapping, on ='user_id', how = 'left')
    
    user_skills.rename(columns = {'user_id':'standup_id'},inplace = True)
    test_private = pd.merge(test_private,user_skills, on ='standup_id', how = 'left')
    test_public = pd.merge(test_public,user_skills, on ='standup_id', how = 'left')
    train_private = pd.merge(train_private,user_skills, on ='standup_id', how = 'left')
    train_public = pd.merge(train_public,user_skills, on ='standup_id', how = 'left')
    
    test_private = test_private.dropna()
    test_public = test_public.dropna()
    train_private = train_private.dropna()
    train_public = train_public.dropna()
#    subset = ['primary_discipline']  subset = ['standup_id']
    
    return train_private, train_public, test_private, test_public
#train_private, train_public, test_private, test_public = get_train_test_dataset()
#train_private, train_public, test_private, test_public = add_label_to_msg(train_private, train_public, test_private, test_public)
def get_dict_UserBased_msg(train_private, train_public, test_private, test_public):
    discipline_msg_train_public = dict()
    for discipline in  test_private.standup_id.unique():
        discipline_msg_train_public[discipline] = list(train_public.message[train_public.standup_id == discipline])

    
    discipline_msg_train_private = dict()
    for discipline in  test_private.standup_id.unique():
        discipline_msg_train_private[discipline] = list(train_private.message[train_private.standup_id == discipline])

        
    discipline_msg_test_public = dict()
    for discipline in  test_private.standup_id.unique():
        discipline_msg_test_public[discipline] = list(test_public.message[test_public.standup_id == discipline])

        
    discipline_msg_test_private = dict()
    for discipline in  test_private.standup_id.unique():
        discipline_msg_test_private[discipline] = list(test_private.message[test_private.standup_id == discipline])
    return discipline_msg_train_public,discipline_msg_train_private,discipline_msg_test_public,discipline_msg_test_private    
    
def get_dict_JobBased_msg(train_private, train_public, test_private, test_public):
    discipline_msg_train_public = dict()
    for discipline in user_skills.primary_discipline.unique():
        discipline_msg_train_public[discipline] = list(train_public.message[train_public.primary_discipline == discipline])
    del discipline_msg_train_public['Advisor']
    del discipline_msg_train_public['Design']
    
    discipline_msg_train_private = dict()
    for discipline in user_skills.primary_discipline.unique():
        discipline_msg_train_private[discipline] = list(train_private.message[train_private.primary_discipline == discipline])
    del discipline_msg_train_private['Advisor']
    del discipline_msg_train_private['Design']
        
    discipline_msg_test_public = dict()
    for discipline in user_skills.primary_discipline.unique():
        discipline_msg_test_public[discipline] = list(test_public.message[test_public.primary_discipline == discipline])
    del discipline_msg_test_public['Advisor']
    del discipline_msg_test_public['Design']
        
    discipline_msg_test_private = dict()
    for discipline in user_skills.primary_discipline.unique():
        discipline_msg_test_private[discipline] = list(test_private.message[test_private.primary_discipline == discipline])
    del discipline_msg_test_private['Advisor']
    del discipline_msg_test_private['Design']
    return discipline_msg_train_public,discipline_msg_train_private,discipline_msg_test_public,discipline_msg_test_private

def main():
    train_private, train_public, test_private, test_public =  get_train_test_dataset()
    train_private, train_public, test_private, test_public = add_label_to_msg(train_private, train_public, test_private, test_public)
    discipline_msg_train_public,discipline_msg_train_private,discipline_msg_test_public,discipline_msg_test_private = get_dict_JobBased_msg(train_private, train_public, test_private, test_public)

#private_msg = temp_private_dual.message.copy().str.lower().dropna()
#public_msg = temp_public_dual.message.copy().str.lower().dropna()