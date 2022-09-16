import enum
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# understanding Covid19 the dataset
def covid19_data_look(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)
    features = dataFrame[['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'gender', 'test_indication']]
    corona_results = dataFrame['corona_result']

    print(features['cough'].unique()) # ['0' '1' 'None']
    print(features['fever'].unique()) # ['0' '1' 'None']
    print(features['sore_throat'].unique()) # ['0' '1' 'None']
    print(features['shortness_of_breath'].unique()) # ['0' '1' 'None']
    print(features['head_ache'].unique()) # ['0' '1' 'None']
    print(features['age_60_and_above'].unique()) # 'None' 'Yes' 'No']
    print(features['gender'].unique()) # ['female' 'male' 'None']
    print(features['test_indication'].unique()) # ['Other' 'Abroad' 'Contact with confirmed']

    print(corona_results.unique()) # ['negative' 'positive' 'other']

    print(features.shape)
    print(corona_results.shape)


# load Covid19 data
def covid19_load_data(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)
    # romove all None and 
    dataFrame['cough'] = dataFrame['cough'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['cough'])
    dataFrame['fever'] = dataFrame['fever'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['fever'])
    dataFrame['sore_throat'] = dataFrame['sore_throat'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['sore_throat'])
    dataFrame['shortness_of_breath'] = dataFrame['shortness_of_breath'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['shortness_of_breath'])
    dataFrame['head_ache'] = dataFrame['head_ache'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['head_ache'])
    dataFrame['age_60_and_above'] = dataFrame['age_60_and_above'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['age_60_and_above'])
    dataFrame['gender'] = dataFrame['gender'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['gender'])
    # dataFrame['test_indication'] = dataFrame['test_indication'].replace('None', np.nan)
    # dataFrame = dataFrame.dropna(axis=0, subset=['test_indication'])
    dataFrame['corona_result'] = dataFrame['corona_result'].replace('other', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['corona_result'])

    # make data balanced
    dataFrame = dataFrame.groupby('corona_result')
    dataFrame = dataFrame.apply(lambda x: x.sample(dataFrame.size().min()).reset_index(drop=True))

    # replace Yes/No, Positive/Negative ... to 0, 1
    # for - cough, fever, sore_throat, shortness_of_breath, head_ache
    dataFrame = dataFrame.replace("1", 1)
    dataFrame = dataFrame.replace("0", 0)
    # for - age_60_and_above
    dataFrame = dataFrame.replace("Yes", 1)
    dataFrame = dataFrame.replace("No", 0)
    # for - gender
    dataFrame = dataFrame.replace("male", 1)
    dataFrame = dataFrame.replace("female", 0)
    # for - test_indication ['Other' 'Abroad' 'Contact with confirmed']
    dataFrame = dataFrame.replace("Contact with confirmed", 1)
    dataFrame = dataFrame.replace("Other", 0)
    dataFrame = dataFrame.replace("Abroad", 0)
    # for - corona_result
    dataFrame = dataFrame.replace("positive", 1)
    dataFrame = dataFrame.replace("negative", 0)


    features = dataFrame[['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'gender', 'test_indication']]
    corona_results = dataFrame['corona_result']
    
    X = features.to_numpy()
    y = corona_results.to_numpy()

    return X, y


# understanding Adults dataset
def adults_data_look(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)

    dataFrame.info()
    
    column_names = list(dataFrame.columns.values)
    print(column_names)

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num', 'marital_status', 'occupation', 'relationship',
'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    dataFrame.columns=col_names
    
    print(len(dataFrame['age'].unique()))
    print(dataFrame['age'].unique()) #  74 unique values
    print(len(dataFrame['workclass'].unique()))
    print(dataFrame['workclass'].unique()) # 9 unique values, ['Private' 'Local-gov' '?' 'Self-emp-not-inc' 'Federal-gov' 'State-gov' 'Self-emp-inc' 'Without-pay' 'Never-worked']
    print(len(dataFrame['fnlwgt'].unique()))
    print(dataFrame['fnlwgt'].unique()) #  28523 unique values
    print(len(dataFrame['education'].unique()))
    print(dataFrame['education'].unique()) #  16 unique values
    print(len(dataFrame['educational_num'].unique()))
    print(dataFrame['educational_num'].unique()) #  16 unique values
    print(len(dataFrame['marital_status'].unique()))
    print(dataFrame['marital_status'].unique()) #  7 unique values
    print(len(dataFrame['occupation'].unique()))
    print(dataFrame['occupation'].unique()) #  15 unique values
    print(len(dataFrame['relationship'].unique()))
    print(dataFrame['relationship'].unique()) #  6 unique values
    print(len(dataFrame['race'].unique()))
    print(dataFrame['race'].unique()) #  5 unique values
    print(len(dataFrame['gender'].unique()))
    print(dataFrame['gender'].unique()) #  2 unique values
    print(len(dataFrame['capital_gain'].unique()))
    print(dataFrame['capital_gain'].unique()) #  123 unique values
    print(len(dataFrame['capital_loss'].unique()))
    print(dataFrame['capital_loss'].unique()) #  99 unique values
    print(len(dataFrame['hours_per_week'].unique()))
    print(dataFrame['hours_per_week'].unique()) #  96 unique values
    print(len(dataFrame['native_country'].unique()))
    print(dataFrame['native_country'].unique()) #  42 unique values

    lb=LabelEncoder()
    dataFrame.workclass=lb.fit_transform(dataFrame.workclass)
    label_id_list = range(0, len(dataFrame['workclass'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute workclass:")
    print(label_name_list)

    dataFrame.education=lb.fit_transform(dataFrame.education)
    label_id_list = range(0, len(dataFrame['education'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute education:")
    print(label_name_list)

    dataFrame.marital_status=lb.fit_transform(dataFrame.marital_status)
    label_id_list = range(0, len(dataFrame['marital_status'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute marital_status:")
    print(label_name_list)

    dataFrame.occupation=lb.fit_transform(dataFrame.occupation)
    label_id_list = range(0, len(dataFrame['occupation'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute occupation:")
    print(label_name_list)

    dataFrame.relationship=lb.fit_transform(dataFrame.relationship)
    label_id_list = range(0, len(dataFrame['relationship'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute relationship:")
    print(label_name_list)

    dataFrame.race=lb.fit_transform(dataFrame.race)
    label_id_list = range(0, len(dataFrame['race'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute race:")
    print(label_name_list)

    dataFrame.gender=lb.fit_transform(dataFrame.gender)
    label_id_list = range(0, len(dataFrame['gender'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute gender:")
    print(label_name_list)

    dataFrame.native_country=lb.fit_transform(dataFrame.native_country)
    label_id_list = range(0, len(dataFrame['native_country'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute native_country:")
    print(label_name_list)

    dataFrame.income=lb.fit_transform(dataFrame.income)
    label_id_list = range(0, len(dataFrame['income'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print(label_name_list)

    print(dataFrame)


# load Adults data
def adults_load_data(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    dataFrame.columns=col_names

    # dataFrame=dataFrame.drop(['fnlwgt'],axis=1)

    lb=LabelEncoder()
    dataFrame.workclass=lb.fit_transform(dataFrame.workclass)
    label_id_list = range(0, len(dataFrame['workclass'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.education=lb.fit_transform(dataFrame.education)
    label_id_list = range(0, len(dataFrame['education'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.marital_status=lb.fit_transform(dataFrame.marital_status)
    label_id_list = range(0, len(dataFrame['marital_status'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.occupation=lb.fit_transform(dataFrame.occupation)
    label_id_list = range(0, len(dataFrame['occupation'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.relationship=lb.fit_transform(dataFrame.relationship)
    label_id_list = range(0, len(dataFrame['relationship'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.race=lb.fit_transform(dataFrame.race)
    label_id_list = range(0, len(dataFrame['race'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.gender=lb.fit_transform(dataFrame.gender)
    label_id_list = range(0, len(dataFrame['gender'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.native_country=lb.fit_transform(dataFrame.native_country)
    label_id_list = range(0, len(dataFrame['native_country'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.income=lb.fit_transform(dataFrame.income)
    label_id_list = range(0, len(dataFrame['income'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    X = dataFrame.iloc[:, :-1].to_numpy()
    # scaler=StandardScaler()
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)

    y=dataFrame.iloc[:,-1].to_numpy()

    return X, y, scaler


# understanding Fivethiryeight dataset
def fivethirtyeight_data_look(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)

    dataFrame.info()

    dataFrame = dataFrame.iloc[1: , :] # remove the 1st row which does not contain data
    dataFrame = dataFrame.drop(['RespondentID'],axis=1)
    
    column_names = list(dataFrame.columns.values)
    print(column_names)

    col_names = ['lottery', 'cigaretters', 'alcohol', 'gamble', 'skydiving', 'speed_limit', 'cheated', 'eat_steak', 'steak_label', 'gender', 'age', 'income', 'education', 'location']
    dataFrame.columns=col_names

    dataFrame = dataFrame.dropna()
    
    print(len(dataFrame['lottery'].unique()))
    print(dataFrame['lottery'].unique()) #  2 unique values ['Lottery B' 'Lottery A']
    print(len(dataFrame['cigaretters'].unique()))
    print(dataFrame['cigaretters'].unique()) # 2 unique values  'No' 'Yes']
    print(len(dataFrame['alcohol'].unique()))
    print(dataFrame['alcohol'].unique()) #  2 unique values  'Yes' 'No']
    print(len(dataFrame['gamble'].unique()))
    print(dataFrame['gamble'].unique()) #  2 unique values  'Yes' 'No']
    print(len(dataFrame['skydiving'].unique()))
    print(dataFrame['skydiving'].unique()) #  2 unique values ['Yes' 'No']
    print(len(dataFrame['speed_limit'].unique()))
    print(dataFrame['speed_limit'].unique()) # 2 unique values  ['Yes' 'No']
    print(len(dataFrame['cheated'].unique()))
    print(dataFrame['cheated'].unique()) #  2 unique values ['Yes' 'No']
    print(len(dataFrame['eat_steak'].unique()))
    print(dataFrame['eat_steak'].unique()) #  2 unique values ['Yes' 'No']
    print(len(dataFrame['steak_label'].unique()))
    print(dataFrame['steak_label'].unique()) # 5 unique values ['Medium rare' 'Rare' 'Medium' 'Medium Well' 'Well']
    print(len(dataFrame['gender'].unique()))
    print(dataFrame['gender'].unique()) #  2 unique values ['Male' 'Female']
    print(len(dataFrame['age'].unique()))
    print(dataFrame['age'].unique()) #  4 unique values ['> 60' '18-29' '30-44' '45-60']
    print(len(dataFrame['income'].unique()))
    print(dataFrame['income'].unique()) #  5 unique values ['$50,000-$99,999' '$150,000+' '$0-$24,999' '$25,000-$49,999' '$100,000-$149,999']
    print(len(dataFrame['education'].unique()))
    print(dataFrame['education'].unique()) #  9 unique values ['Bachelor degree' 'Graduate degree' 'High school degree'  'Less than high school degree' 'Some college or Associate degree']
    print(len(dataFrame['location'].unique()))
    print(dataFrame['location'].unique()) #  9 unique values ['East North Central' 'South Atlantic' 'New England' 'Middle Atlantic' 'West South Central' 'West North Central' 'Pacific' 'Mountain' 'East South Central']

    dataFrame = dataFrame.drop(['eat_steak'],axis=1) # Because all people eat steak!!!
    dataFrame.info()

    print(dataFrame.head())

    lb=LabelEncoder()
    dataFrame.lottery=lb.fit_transform(dataFrame.lottery)
    label_id_list = range(0, len(dataFrame['lottery'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute lottery:")
    print(label_name_list)

    dataFrame.cigaretters=lb.fit_transform(dataFrame.cigaretters)
    label_id_list = range(0, len(dataFrame['cigaretters'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute cigaretters:")
    print(label_name_list)

    dataFrame.alcohol=lb.fit_transform(dataFrame.alcohol)
    label_id_list = range(0, len(dataFrame['alcohol'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute alcohol:")
    print(label_name_list)

    dataFrame.gamble=lb.fit_transform(dataFrame.gamble)
    label_id_list = range(0, len(dataFrame['gamble'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute gamble:")
    print(label_name_list)

    dataFrame.skydiving=lb.fit_transform(dataFrame.skydiving)
    label_id_list = range(0, len(dataFrame['skydiving'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute skydiving:")
    print(label_name_list)

    dataFrame.speed_limit=lb.fit_transform(dataFrame.speed_limit)
    label_id_list = range(0, len(dataFrame['speed_limit'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute speed_limit:")
    print(label_name_list)

    dataFrame.cheated=lb.fit_transform(dataFrame.cheated)
    label_id_list = range(0, len(dataFrame['cheated'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute cheated:")
    print(label_name_list)

    # dataFrame.eat_steak=lb.fit_transform(dataFrame.eat_steak)
    # label_id_list = range(0, len(dataFrame['eat_steak'].unique()))
    # label_name_list = lb.inverse_transform(label_id_list)
    # print("label id for attribute eat_steak:")
    # print(label_name_list)

    dataFrame.gender=lb.fit_transform(dataFrame.gender)
    label_id_list = range(0, len(dataFrame['gender'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute gender:")
    print(label_name_list)

    dataFrame.age=lb.fit_transform(dataFrame.age)
    label_id_list = range(0, len(dataFrame['age'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute age:")
    print(label_name_list)

    dataFrame.income=lb.fit_transform(dataFrame.income)
    label_id_list = range(0, len(dataFrame['income'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print(label_name_list)

    dataFrame.education=lb.fit_transform(dataFrame.education)
    label_id_list = range(0, len(dataFrame['education'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute education:")
    print(label_name_list)

    dataFrame.location=lb.fit_transform(dataFrame.location)
    label_id_list = range(0, len(dataFrame['location'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute location:")
    print(label_name_list)
    
    dataFrame.steak_label=lb.fit_transform(dataFrame.steak_label)
    label_id_list = range(0, len(dataFrame['steak_label'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute steak_label:")
    print(label_name_list)

    print(dataFrame)


# laod Fivethiryeight data
def fivethirtyeight_load_data(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)

    dataFrame = dataFrame.iloc[1: , :] # remove the 1st row which does not contain data

    dataFrame = dataFrame.drop(['RespondentID'],axis=1)
    
    col_names = ['lottery', 'cigaretters', 'alcohol', 'gamble', 'skydiving', 'speed_limit', 'cheated', 'eat_steak', 'steak_label', 'gender', 'age', 'income', 'education', 'location']
    dataFrame.columns=col_names

    dataFrame = dataFrame.dropna()
    
    dataFrame = dataFrame.drop(['eat_steak'],axis=1) # Because all people eat steak!!!

    # column index 0
    lb=LabelEncoder()
    dataFrame.lottery=lb.fit_transform(dataFrame.lottery)
    label_id_list = range(0, len(dataFrame['lottery'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 1
    dataFrame.cigaretters=lb.fit_transform(dataFrame.cigaretters)
    label_id_list = range(0, len(dataFrame['cigaretters'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 2
    dataFrame.alcohol=lb.fit_transform(dataFrame.alcohol)
    label_id_list = range(0, len(dataFrame['alcohol'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 3
    dataFrame.gamble=lb.fit_transform(dataFrame.gamble)
    label_id_list = range(0, len(dataFrame['gamble'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 4
    dataFrame.skydiving=lb.fit_transform(dataFrame.skydiving)
    label_id_list = range(0, len(dataFrame['skydiving'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 5
    dataFrame.speed_limit=lb.fit_transform(dataFrame.speed_limit)
    label_id_list = range(0, len(dataFrame['speed_limit'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 6
    dataFrame.cheated=lb.fit_transform(dataFrame.cheated)
    label_id_list = range(0, len(dataFrame['cheated'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # dataFrame.eat_steak=lb.fit_transform(dataFrame.eat_steak)
    # label_id_list = range(0, len(dataFrame['eat_steak'].unique()))
    # label_name_list = lb.inverse_transform(label_id_list)

    # column index 7
    dataFrame.steak_label=lb.fit_transform(dataFrame.steak_label)
    label_id_list = range(0, len(dataFrame['steak_label'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 8
    dataFrame.gender=lb.fit_transform(dataFrame.gender)
    label_id_list = range(0, len(dataFrame['gender'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 9
    dataFrame.age=lb.fit_transform(dataFrame.age)
    label_id_list = range(0, len(dataFrame['age'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 10
    dataFrame.income=lb.fit_transform(dataFrame.income)
    label_id_list = range(0, len(dataFrame['income'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 11
    dataFrame.education=lb.fit_transform(dataFrame.education)
    label_id_list = range(0, len(dataFrame['education'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    # column index 12
    dataFrame.location=lb.fit_transform(dataFrame.location)
    label_id_list = range(0, len(dataFrame['location'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    feature_index = [0,1,2,3,4,5,6,8,9,10,11,12]
    X = dataFrame.iloc[:, feature_index].to_numpy()
    # scaler=StandardScaler()
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)

    y=dataFrame.iloc[:,7].to_numpy()
    
    # print(type(X))
    # print(type(y))
    # print(X.shape)
    # print(y.shape)
    # print(X)
    # print(y)

    return X, y, scaler


# understanding GSS dataset
def gss_data_look(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)

    dataFrame.info()
    
    column_names = list(dataFrame.columns.values)
    print(column_names)
    
    print(len(dataFrame['year'].unique()))
    print(dataFrame['year'].unique()) #  74 unique values
    print(len(dataFrame['marital'].unique()))
    print(dataFrame['marital'].unique()) # 9 unique values, ['Private' 'Local-gov' '?' 'Self-emp-not-inc' 'Federal-gov' 'State-gov' 'Self-emp-inc' 'Without-pay' 'Never-worked']
    print(len(dataFrame['divorce'].unique()))
    print(dataFrame['divorce'].unique()) #  28523 unique values
    print(len(dataFrame['childs'].unique()))
    print(dataFrame['childs'].unique()) #  16 unique values
    print(len(dataFrame['age'].unique()))
    print(dataFrame['age'].unique()) #  16 unique values
    print(len(dataFrame['educ'].unique()))
    print(dataFrame['educ'].unique()) #  7 unique values
    print(len(dataFrame['sex'].unique()))
    print(dataFrame['sex'].unique()) #  15 unique values
    print(len(dataFrame['race'].unique()))
    print(dataFrame['race'].unique()) #  6 unique values
    print(len(dataFrame['relig'].unique()))
    print(dataFrame['relig'].unique()) #  5 unique values
    print(len(dataFrame['xmovie'].unique()))
    print(dataFrame['xmovie'].unique()) #  2 unique values
    print(len(dataFrame['pornlaw'].unique()))
    print(dataFrame['pornlaw'].unique()) #  123 unique values
    print(len(dataFrame['hapmar'].unique()))
    print(dataFrame['hapmar'].unique()) #  42 unique values

    lb=LabelEncoder()
    dataFrame.year=lb.fit_transform(dataFrame.year)
    label_id_list = range(0, len(dataFrame['year'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute year:")
    print(label_name_list)

    dataFrame.marital=lb.fit_transform(dataFrame.marital)
    label_id_list = range(0, len(dataFrame['marital'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute marital:")
    print(label_name_list)

    dataFrame.divorce=lb.fit_transform(dataFrame.divorce)
    label_id_list = range(0, len(dataFrame['divorce'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute divorce:")
    print(label_name_list)

    dataFrame.childs=lb.fit_transform(dataFrame.childs)
    label_id_list = range(0, len(dataFrame['childs'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute childs:")
    print(label_name_list)

    dataFrame.age=lb.fit_transform(dataFrame.age)
    label_id_list = range(0, len(dataFrame['age'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute age:")
    print(label_name_list)

    dataFrame.educ=lb.fit_transform(dataFrame.educ)
    label_id_list = range(0, len(dataFrame['educ'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute educ:")
    print(label_name_list)

    dataFrame.sex=lb.fit_transform(dataFrame.sex)
    label_id_list = range(0, len(dataFrame['sex'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute sex:")
    print(label_name_list)

    dataFrame.race=lb.fit_transform(dataFrame.race)
    label_id_list = range(0, len(dataFrame['race'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute race:")
    print(label_name_list)

    dataFrame.relig=lb.fit_transform(dataFrame.relig)
    label_id_list = range(0, len(dataFrame['relig'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute relig:")
    print(label_name_list)

    dataFrame.xmovie=lb.fit_transform(dataFrame.xmovie)
    label_id_list = range(0, len(dataFrame['xmovie'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute xmovie:")
    print(label_name_list)

    dataFrame.pornlaw=lb.fit_transform(dataFrame.pornlaw)
    label_id_list = range(0, len(dataFrame['pornlaw'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute pornlaw:")
    print(label_name_list)

    dataFrame.hapmar=lb.fit_transform(dataFrame.hapmar)
    label_id_list = range(0, len(dataFrame['hapmar'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)
    print("label id for attribute hapmar:")
    print(label_name_list)

    print(dataFrame)


# load Adults data
def gss_load_data(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)

    # print(dataFrame)

    # dataFrame=dataFrame.drop(['fnlwgt'],axis=1)

    lb=LabelEncoder()
    dataFrame.year=lb.fit_transform(dataFrame.year)
    label_id_list = range(0, len(dataFrame['year'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.marital=lb.fit_transform(dataFrame.marital)
    label_id_list = range(0, len(dataFrame['marital'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.divorce=lb.fit_transform(dataFrame.divorce)
    label_id_list = range(0, len(dataFrame['divorce'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.childs=lb.fit_transform(dataFrame.childs)
    label_id_list = range(0, len(dataFrame['childs'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.age=lb.fit_transform(dataFrame.age)
    label_id_list = range(0, len(dataFrame['age'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.educ=lb.fit_transform(dataFrame.educ)
    label_id_list = range(0, len(dataFrame['educ'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.sex=lb.fit_transform(dataFrame.sex)
    label_id_list = range(0, len(dataFrame['sex'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.race=lb.fit_transform(dataFrame.race)
    label_id_list = range(0, len(dataFrame['race'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.relig=lb.fit_transform(dataFrame.relig)
    label_id_list = range(0, len(dataFrame['relig'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.xmovie=lb.fit_transform(dataFrame.xmovie)
    label_id_list = range(0, len(dataFrame['xmovie'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.pornlaw=lb.fit_transform(dataFrame.pornlaw)
    label_id_list = range(0, len(dataFrame['pornlaw'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    dataFrame.hapmar=lb.fit_transform(dataFrame.hapmar)
    label_id_list = range(0, len(dataFrame['hapmar'].unique()))
    label_name_list = lb.inverse_transform(label_id_list)

    X = dataFrame.iloc[:, :-1].to_numpy()

    # scaler=StandardScaler()
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)

    y=dataFrame.iloc[:,-1].to_numpy()

    return X, y, scaler


class TabularData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
    

    def __getitem__(self, index):
        return self.X[index], self.y[index]