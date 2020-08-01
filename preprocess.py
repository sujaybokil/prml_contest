import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PowerTransformer

def handle_DateTime(df, col, ls):  # Separates datetime into day/month/year as per need
    days = []
    months = []
    years = []
    date_list = df[col].tolist()
    for obj in date_list:
        time = str(obj)
        days.append(int(time[:2]))
        months.append(int(time[3:5]))
        years.append(int(time[6:10]))

    if ls[0] == True:
        df[str(col)+'_day'] = days
    if ls[1] == True:
        df[str(col)+'_month'] = months
    if ls[2] == True:
        df[str(col)+'_year'] = years
    df.drop(columns=[col], inplace=True)


def length(s):  # Utility function for txt_len()
    return len(s)


def txt_len(df):  # calculates length of remark
    df['txt'] = df['txt'].astype(str)
    df['txt_length'] = df['txt'].apply(length)
    df.drop(columns=['txt'], inplace=True)


def basic_clean(df):  # Removes negative ids and null valued rows
    clean_df = df.dropna(how='any')
    clean_df['emp'] = df['emp'].astype(int)
    return clean_df[clean_df['emp'] > 0]


def get_Ids(df):  # Creates a list of unique ids in that dataframe
    emp_ = df['emp'].astype(int).tolist()
    comp_ = df['comp'].astype(str).tolist()
    ids = []
    for i in range(len(df)):
        id = "{:04d}".format(emp_[i])+comp_[i]
        ids.append(id)
    df['Id'] = ids


def encode_so(df_so):  # so stands for support oppose
    support = df_so['support'].tolist()
    oppose = df_so['oppose'].tolist()

    ref_dict = {True: 1, False: 0, np.nan: 0.5}
    supp_en = [ref_dict[x] for x in support]
    opp_en = [ref_dict[x] for x in oppose]
    df_so['support'] = supp_en
    df_so['oppose'] = opp_en


def scale_matrix(M):
    return (M-(M.min())*np.ones(shape=M.shape))/(M.max()-M.min())


def cvt2binary(labels):
    class0 = []
    class1 = []
    for x in labels.tolist():
        if x == 0:
            class0.append(1)
            class1.append(0)
        else:
            class0.append(0)
            class1.append(1)
    return pd.DataFrame(list(zip(class0, class1)))


# Does all transformations and makes the data model ready
def feature_engineer(data, rt_dict, rlen_dict, rnum_dict, supp_dict, opp_dict):
    df = data.dropna(how='any')
    df.drop(index=df[df['emp'] <= 0].index, inplace=True)
    get_Ids(df)
    df.drop(columns=['id', 'emp'], inplace=True)
    handle_DateTime(df, 'lastratingdate', [False, True, True])

    ratings_ = []
    rlen_ = []
    rct_ = []
    supp_ = []
    opp_ = []

    for id in df['Id'].tolist():
        if id in rt_dict:
            ratings_.append(rt_dict[id])
        else:
            # That number is the mean of all ratings = 0.5904
            ratings_.append(0.5904)

        if id in rlen_dict:
            rlen_.append(rlen_dict[id])
        else:
            rlen_.append(0)

        if id in rnum_dict:
            rct_.append(rnum_dict[id])
        else:
            rct_.append(0)

        if id in supp_dict:
            supp_.append(supp_dict[id])
        else:
            supp_.append(0)

        if id in opp_dict:
            opp_.append(opp_dict[id])
        else:
            opp_.append(0)

    df['ratings'] = ratings_
    df['support'] = supp_
    df['oppose'] = opp_
    df['remark_length'] = rlen_
    df['num_remarks'] = rct_

    y = df['left']
    df.drop(columns=['left', 'Id'], inplace=True)

    ### Encoding categorical data ###
    categorical_cols = ['comp', 'lastratingdate_month', 'lastratingdate_year']
    df[categorical_cols] = df[categorical_cols].astype(object)
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = pd.Series(encoder.fit_transform(df[col]))

    ### Scaling numerical data ###
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df_sc = pd.DataFrame(pt.fit_transform(df), columns=df.columns)

    return df_sc, y
