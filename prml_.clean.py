# importing libraries

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score

from sklearn.ensemble import RandomForestClassifier

le = LabelEncoder()    

# reading all necessary .csv files

train = pd.read_csv(r'I:\prml20jslot\train.csv')
test  = pd.read_csv(r'I:\prml20jslot\test.csv')
remarks = pd.read_csv(r'I:\prml20jslot\remarks.csv')
remarks_supp_opp = pd.read_csv(r'I:\prml20jslot\remarks_supp_opp.csv')
ratings = pd.read_csv(r'I:\prml20jslot\ratings.csv')

# utility functions

# Creates a unique id for a employee, (emp,comp)
def unique_id(df):
    el = df['emp'].tolist()
    cl = df['comp'].tolist()
    tup = tuple(zip(el,cl))
    df['unique_id'] = tup
    df['unique_id'] = df['unique_id'].astype(str)
    return df

# returns length of the remark string

def remark_length(s):
    n = len(s)
    return n

#cleaning remarks dataframe

def clean_remarks(remarks):
    
    remarks = remarks[remarks['emp'] > 0] #droping negative emp rows
    remarks = remarks.drop(remarks[remarks['txt'].isnull()].index) #null remarks are dropped
    remarks['length'] = remarks['txt'].apply(remark_length) # adding column of remark length
    remarks.drop('txt',axis=1,inplace=True)
    remarks = unique_id(remarks)

    return remarks


#cleaning remarks_supp_opp

def clean_remarks_supp_opp(remarks_supp_opp):
    
    remarks_supp_opp = remarks_supp_opp[remarks_supp_opp['emp'] > 0] #droping negative emp rows
    remarks_supp_opp.drop('oppose',axis=1,inplace=True)   #oppose and support convey same info
    remarks_supp_opp = remarks_supp_opp[remarks_supp_opp['remarkId'].isin(remarks['remarkId'].unique())] 
    #dropping rows which give support info for remarks that are not listed in the remarks file
    remarks_supp_opp['support'] = le.fit_transform(remarks_supp_opp['support'] )
    remarks_supp_opp=remarks_supp_opp.replace(to_replace=0 , value = -1)
    remarks_supp_opp = unique_id(remarks_supp_opp)

    return remarks_supp_opp

##################################################################
  #                           CLEANING

remarks_supp_opp = clean_remarks_supp_opp(remarks_supp_opp)
remarks = clean_remarks(remarks)

##################################################################



#calculates how much support a remark got from other employees

def support_factor(g,remarks):
    
    remarkid_g = g.index.tolist()
    sf = g['support'].tolist()
    d = dict(zip(remarkid_g,sf))
    remarkid_remarks = remarks['remarkId'].tolist() 
    sup=[]
    for r in remarkid_remarks:
        if r in remarkid_g:
            sup.append(d[r])
        else:
            sup.append(0)
    
    return sup

#feature engineering remarks

def feature_engineer_remarks(remarks,remarks_supp_opp):
    
    g = remarks_supp_opp.groupby('remarkId').mean()
    s_ = support_factor(g,remarks)
    remarks['support_factor'] = s_
    
    return remarks

##########################################################################


remarks = feature_engineer_remarks(remarks,remarks_supp_opp)


##########################################################################


train = unique_id(train)
ratings = unique_id(ratings)
test = unique_id(test)


##########################################################################



#calculates average rating made by an employee

def avg_rating(g_,train):
    train_unique = train['unique_id'].tolist()
    rating =   g_['rating'].tolist()
    g_unique = g_.index.tolist()
    d = dict(zip(g_unique,rating))
    avg_rate =[]
    for u in train_unique:
        if u in g_unique:
            avg_rate.append(d[u])
        else:
            avg_rate.append(0)
    return avg_rate

#feature engineering ratings

def feature_engineer_ratings(ratings,train):
   
    g = ratings.groupby('unique_id').mean()
    a = avg_rating(g,train)
    train['avg_rating'] = a

    return train

# feature engineer train further

def feature_engg(remarks,ratings,train):
   
    train = feature_engineer_ratings(ratings,train)

    train_unique = train['unique_id'].tolist()
    g = remarks.groupby('unique_id').mean()
    uniqueid_g = g.index.tolist()
    sf = g['support_factor'].tolist()
    avg_len = g['length'].tolist()
    d1 = dict(zip(uniqueid_g,sf))
    d2 = dict(zip(uniqueid_g,avg_len))
    al =[]
    sf_ = []
    for u in train_unique:
        if u in uniqueid_g:
            al.append(d2[u]) 
            sf_.append(d1[u])
        else:
            al.append(0)
            sf_.append(0)
    train['avg_remark_length'] = al
    train['avg_support_factor'] = sf_
    
    train = extract_time(train)

    return train

# extract time since last remark

def extract_time(nt):
    
    nt['lastratingdate'] = pd.to_datetime(nt['lastratingdate'], zdayfirst=True)
    nt['lastratingdate'] = pd.to_datetime('2017-03-20 00:00:00') - nt['lastratingdate']
    nt['lastratingdate'] = nt['lastratingdate'].astype('timedelta64[D]')
    
    return nt


##############################################################################

train = feature_engg(remarks,ratings,train)
test  = feature_engg(remarks,ratings,test)

###############################################################################


#prepares train and test data for training and testing

def prepare_train_test(train,test):
    
    y = train['left'].copy()
    
    train.drop(['id','emp','unique_id','left'], axis=1, inplace=True)

    index = test['id'].copy()
    test.drop(['id','emp','unique_id'],axis=1,inplace=True)
    
    frames = [train,test]
    total = pd.concat(frames)
    x = pd.get_dummies(total)
    xtrain = x.iloc[:3526]
    test_x  = x.iloc[3526:]
    
    X_train, X_test, y_train,y_test = train_test_split(xtrain,y,test_size=0.2,random_state=42)

    cw2 = compute_class_weight('balanced', np.unique(y_train), y_train)
    c = (cw2[1]/cw2[0])*0.1

    cls_weight = {1:0.1, 0:c}

    return X_train, X_test, y_train, y_test, cls_weight, test_x, index

################################################################################

X_train, X_test, y_train, y_test, cls_weight, test_x, index = prepare_train_test(train,test)

################################################################################



# checks performance of the model

def clas_perf(y_perfect,y_pred):
    
    p=precision_score(y_perfect,y_pred)
    r=recall_score(y_perfect,y_pred)
    f=f1_score(y_perfect,y_pred)
    n_correct=sum(y_pred==y_perfect)
    acc=n_correct/len(y_perfect)
    bas = balanced_accuracy_score(y_perfect, y_pred)
    print('Accuracy: ',acc,"\nPrecision: ",p,"\nRecall   :",r,"\nF1 Score :",f,'\nBalanced Accuracy Score: ',bas)

# Training


def create_model(cls_weight):
    
    model = RandomForestClassifier(class_weight=cls_weight,random_state=42)

    return model


model = create_model(cls_weight)

def train_model(model,X_train, X_test, y_train, y_test):

    print('________________TRAINING_____________\n')
    
    model.fit(X_train,y_train)

    print('________________TRAINING OVER_____________\n')
    
    y_pred_train = model.predict(X_train)
    print('\n_______________Training Accuracy__________\n')
    clas_perf(y_train, y_pred_train)
    
    y_pred_test = model.predict(X_test)
    print('\n_______________Testing Accuracy__________\n')
    clas_perf(y_test, y_pred_test)


train_model(model,X_train, X_test, y_train, y_test)  


print('\n_________Preparing submission file__________\n')



def create_submission(model,test_x):
    
    test_pred = model.predict(test_x)
    prediction = pd.DataFrame(test_pred)
    prediction.rename(columns = {0:'left'}, inplace = True)

    df = pd.DataFrame(index)
    df.rename(columns = {0:'id'}, inplace = True)
    df = pd.concat([df,prediction],axis=1)
    df = df.set_index('id')
    prediction = df.to_csv('prml_submission.csv')
    print('_____________DONE_____________')

create_submission(model,test_x)



    





