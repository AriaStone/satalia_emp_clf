import pandas as pd
import numpy as np
from datetime import datetime

def convert_timestamp(x):
    
    return datetime.fromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S')

def import_metadata():######step 1: import all relevant datasets ############################
    
    satalia_public = pd.read_csv("G:\\satalia\\Slack_Public_19062018.csv", encoding="utf-8-sig",index_col = 0)
    satalia_private = pd.read_csv("G:\\satalia\\Slack_Private_19062018.csv", encoding="utf-8-sig",index_col = 0)
    channel_info = pd.read_csv("G:\\satalia\\channel_info.csv",index_col = 0)
    interest_group_info = pd.read_csv("G:\\satalia\\interest_group_info.csv",index_col = 0)
    user_mapping = pd.read_csv("G:\\satalia\\user_mapping.csv")
    all_standups = pd.read_csv("all_standups_jun17_jun18.csv")
    
    temp_public = pd.read_csv("G:\\satalia\\temp_public.csv", encoding="utf-8-sig")
    temp_private = pd.read_csv("G:\\satalia\\temp_private.csv", encoding="utf-8-sig")
    temp_public_dual = pd.read_csv("G:\\satalia\\temp_public_dual.csv", encoding="utf-8-sig")
    temp_private_dual = pd.read_csv("G:\\satalia\\temp_private_dual.csv", encoding="utf-8-sig")
    
    
    return satalia_public, satalia_private, channel_info, interest_group_info, user_mapping, all_standups

###############operations for message csv docs #######################################
def wrongly_encoded_list(): #### listing wrongly encoded characters ####
    
    ill_encoded_characters = ["â€™","â€œ","â€”","â€¦","Ã¶","Ã¼","Â£"]
    correct_encoded_characters = ["'","\"","-","...","o","u","£"]
    
    return ill_encoded_characters, correct_encoded_characters

def correct_wrong_characters():######replace all wrontly encoded characters with the right ones #####
    
    private_len = len(satalia_private)
    public_len = len(satalia_public)
    ill_list_len = len(ill_encoded_characters)
    for i in range(private_len):
        for j in range(ill_list_len):
            print("private",i)
            if ill_encoded_characters[j] in str(satalia_private.message[i]):
                satalia_private.message[i] = satalia_private.message[i].replace(ill_encoded_characters[j],correct_encoded_characters[j])
                
    for i in range(public_len):
        for j in range(ill_list_len):
            print("public",i)
            if ill_encoded_characters[j] in str(satalia_public.message[i]):
                satalia_public.message[i] = satalia_public.message[i].replace(ill_encoded_characters[j],correct_encoded_characters[j])
                
    return satalia_private, satalia_public

def replace_system_msg():
    #1 remove system messages when joining and leaving channels
    private_len = len(satalia_private)
    public_len = len(satalia_public)
    for i in range(private_len):
        print("private",i)
        if ("has joined the channel" in str(satalia_private.message[i])) or ("has left the channel" in str(satalia_private.message[i])) :
            satalia_private.message[i] = "CHANNELSYSTEMMESSAGE"
        if ("shared a file" in str(satalia_private.message[i])) or ("uploaded a file" in str(satalia_private.message[i])):
            satalia_private.message[i] = "FILEMESSAGE"
    
    for j in range(public_len):
        print("public",j)
        if ("has joined the channel" in str(satalia_public.message[j])) or ("has left the channel" in str(satalia_public.message[j])) :
            satalia_public.message[j] = "CHANNELSYSTEMMESSAGE"
        if ("shared a file" in str(satalia_public.message[j])) or ("uploaded a file" in str(satalia_public.message[j])):
            satalia_public.message[j] = "FILEMESSAGE"
    return satalia_private, satalia_public
###################operations for user_mapping csv doc ##################
def useful_user_mapping(user_mapping):
    return user_mapping.dropna()

def useful_user_message(temp_private,temp_public,user_mapping_dual):
    temp_private_dual = temp_private.copy()
    temp_private_dual = temp_private_dual[ temp_private_dual.user_id.isin(user_mapping_dual.slack_id)]
    
    temp_public_dual = temp_public.copy()
    temp_public_dual = temp_public_dual[temp_public_dual.userid.isin(user_mapping_dual.slack_id)]
    temp_public_dual = temp_public_dual[temp_public_dual.channelid.isin(channel_info.channel_id)]
    return temp_private_dual, temp_public_dual

#########################################################################        
def merge_all_msg():#####merging all messages from satalia staffs and drop messages of non-satalia people###
    #satalia_private,satalia_public
    len_private_message = len(satalia_private)
    len_public_message = len(satalia_public)
    private_message = []
    public_message = []
    
    for i in range(len_private_message):
        if satalia_private.user_id[i] in list(user_mapping.slack_id):
            private_message.append(satalia_private.message[i])
    
    for i in range(len_public_message):
        if satalia_public.userid[i] in list(user_mapping.slack_id):
            public_message.append(satalia_public.message[i])
    
    pp_msg = private_message + public_message
    
    return private_message, public_message, pp_msg

def merge_usermsg_docs(satalia_private,satali_public,user_mapping, all_standups):
    
    return 0

def export_processed_data():
    
    temp_public.to_csv("G:/satalia/temp_public.csv",index = False)
    temp_private.to_csv("G:/satalia/temp_private.csv",index = False)
    
    temp_public_dual.to_csv("G:/satalia/temp_public_dual.csv",index = False)
    temp_private_dual.to_csv("G:/satalia/temp_private_dual.csv",index = False) 
    skills_prop.to_csv("G:/satalia/skills_prop.csv",index = False)

def user_based_msg_merge():

    private_user_msg_all = []
    i = 0
    
    for slackid in user_mapping.slack_id:
        private_user_msg_all.append([int(user_mapping.standup_id[user_mapping.slack_id==slackid])])
        for msg in temp_private_dual.message[temp_private_dual.user_id == slackid]:
            private_user_msg_all[i].append(msg)
        i += 1
    
    public_user_msg_all = []
    i = 0
    for slackid in user_mapping.slack_id:
        public_user_msg_all.append([int(user_mapping.standup_id[user_mapping.slack_id==slackid])])
        for msg in temp_public_dual.message[temp_public_dual.userid == slackid]:
            public_user_msg_all[i].append(msg)
        i += 1
    return private_user_msg_all, public_user_msg_all

def convert_worktime4interestgroups(skills_prop): #the input here is all_standups
    skills_prop = skills_prop.drop(['comment','interest_name','standup_date','project_id'],axis = 1)
    skills_prop.time = skills_prop.time.copy().str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    return skills_prop    

def summarise_skill_working_time(skills_prop):
    skill_working_time = []
    i = 0
    for user in skills_prop.user_id.unique():
        skill_working_time.append([user])
        z = skills_prop[skills_prop.user_id == user].copy()
        for skill in skills_prop[skills_prop.user_id == user].interest_id.unique():
            zz = z[z.interest_id == skill].copy()
            working_time =  zz.time.sum()
            skill_working_time[i].append([skill, working_time])
        i += 1
    return skill_working_time

def convert_wktime_wktimeprop(skill_working_time):
    for i in range(len(skill_working_time)):
        overall_minutes = 0 
        for j in range(1,len(skill_working_time[i])):
            overall_minutes += skill_working_time[i][j][1]
        for j in range(1,len(skill_working_time[i])):
            skill_working_time[i][j][1] /= overall_minutes
    return skill_working_time

def add_primary_discipline_to_standupid(private_user_msg_all,public_user_msg_all ):
    for i in range(len(private_user_msg_all)):
        pr_job_private = list(user_skills.primary_discipline[user_skills.user_id == private_user_msg_all[i][0]])
        private_user_msg_all[i][0] = str(private_user_msg_all[i][0]) + ':' + pr_job_private[0]
        pr_job_public = list(user_skills.primary_discipline[user_skills.user_id == public_user_msg_all[i][0]])
        public_user_msg_all[i][0] = str(public_user_msg_all[i][0]) + ':' + pr_job_public[0]
    return private_user_msg_all,public_user_msg_all

def user_based_corpora(private_user_msg_all,public_user_msg_all ):    
    z_msg = []
    for user in range(63):
        z = ''  
        i = 1  
        #user = 1
        for msg in private_user_msg_all[user][1:]:
            print(user,i)
            msg = str([i]) + str(msg) + ' '
            z = z + msg 
            i +=1
        z_msg.append([z])
    
    z_msg_public = []
    for user in range(63):
        z = ''  
        i = 1  
        #user = 1
        for msg in public_user_msg_all[user][1:]:
            print(user,i)
            msg = str([i]) + str(msg) + ' '
            z = z + msg 
            i +=1
        z_msg_public.append([z])
    return z_msg, z_msg_public

def run_log(): # This would be the __main__ at last when the entire preprocess source code has been completed
    
    temp_public_dual = pd.read_csv("G:\\satalia\\temp_public_dual.csv", encoding="utf-8-sig") # 25 to 19
    temp_private_dual = pd.read_csv("G:\\satalia\\temp_private_dual.csv", encoding="utf-8-sig") #75 to 55
    channel_info = pd.read_csv("G:\\satalia\\channel_info.csv",index_col = 0)
    interest_group_info = pd.read_csv("G:\\satalia\\interest_group_info.csv",index_col = 0)
    user_mapping = pd.read_csv("G:\\satalia\\user_mapping.csv")
    user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
    all_standups = pd.read_csv("all_standups_jun17_jun18.csv") 
   
    
    user_mapping = user_mapping.dropna() # 78 to 63
    channel_info = channel_info.dropna(subset = ["interest_id"]) # 325 to 41
    user_skills = user_skills[user_skills.user_id.isin(user_mapping.standup_id)] # 85 to 60 
    private_user_msg_all, public_user_msg_all = user_based_msg_merge() #if splitting train/test is needed, to be done here
    private_user_msg_all,public_user_msg_all = add_primary_discipline_to_standupid(private_user_msg_all,public_user_msg_all )
    z_msg_private,z_msg_public = user_based_corpora(private_user_msg_all,public_user_msg_all)
    
    skills_prop = all_standups.copy()
    skills_prop = convert_worktime4interestgroups(skills_prop)
    skill_working_time = summarise_skill_working_time(skills_prop)
    skill_working_time_prop = convert_wktime_wktimeprop(skill_working_time)
    skill_working_time = summarise_skill_working_time(skills_prop)


def result_representation():
    satalia_public = pd.read_csv("G:\\satalia\\Slack_Public_19062018.csv", encoding="utf-8-sig",index_col = 0)
    satalia_private = pd.read_csv("G:\\satalia\\Slack_Private_19062018.csv", encoding="utf-8-sig",index_col = 0)
    channel_info = pd.read_csv("G:\\satalia\\channel_info.csv",index_col = 0)
    interest_group_info = pd.read_csv("G:\\satalia\\interest_group_info.csv",index_col = 0)
    user_mapping = pd.read_csv("G:\\satalia\\user_mapping.csv")
    all_standups = pd.read_csv("all_standups_jun17_jun18.csv")
    user_skills = pd.read_csv("G:\\satalia\\user_skills.csv")
    user_mapping = user_mapping.dropna()
    user_skills = user_skills[user_skills.user_id.isin(user_mapping.standup_id)]
    user_mapping.rename(columns = {'standup_id':'user_id'},inplace = True)
    user_skills = pd.merge(user_skills,user_mapping,on = 'user_id',how = 'left')
    return 0

def get_time_user_in_satalia(user_skills):
    msg_time = []
    for user in user_skills.slack_id:
        temp = []
        temp.append(user)
        temp.append(int(user_skill.user_id[user_skill.slack_id == user]))
        temp.append(list(user_skill.primary_discipline[user_skill.slack_id == user])[0])
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

#temp_public_dual = pd.read_csv("G:\\satalia\\temp_public_dual.csv", encoding="utf-8-sig") # 25 to 19
#temp_private_dual = pd.read_csv("G:\\satalia\\temp_private_dual.csv", encoding="utf-8-sig")
#
#private_msg = temp_private_dual.message.copy().str.lower().dropna()
#public_msg = temp_public_dual.message.copy().str.lower().dropna()

#What've done: 1. fixed encoding errors 2. replaced all trival file sharing and channel-related system messages

#What to do next: 
#1. drop msg who are not in user_mapping
#2. based on user, make msg to be a collection
#3. tagg every collection with interest(topic) id
#4. generate docs form collections and seperate train/validation/test sets with 8/1/1
    
#Questions to solve:
#1. relationship btn public and private
#2. use all/part of overall collections to build language models
    
#P.S. directly import temp_private/public 