import pandas as pd
from sklearn.model_selection import train_test_split

def handling_categorical(data):
    data['gender'] = data['gender'].map({'male': 1, 'female': 0})
    data['discount_eligibility'] = data['discount_eligibility'].map({'yes': 1, 'no': 0})

    dummies = pd.get_dummies(data['region'], drop_first=False)

    data.drop(columns=['region'], inplace=True)
    
    insert_at = 4
    for i, col in enumerate(dummies.columns):
        data.insert(insert_at + i, col, dummies[col])
        
    return data

def preprocessing_join(path):
    data = pd.read_csv(path)
    
    data = handling_categorical(data)
    # print(data.head())
        
    x = data.iloc[:, :-2].values
    y = data.iloc[:, -2:].values
    # print(x)
    # print(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    
    return x_train, x_test, y_train, y_test

def preprocessing_premium_goal(path):
    data = pd.read_csv(path)
    
    data = handling_categorical(data)
    # print(data.head())
        
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # print(x)
    # print(y)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    
    return x_train, x_test, y_train, y_test

def preprocessing_expenses_goal(path):
    data = pd.read_csv(path)
    
    data = handling_categorical(data)
    # print(data.head())
        
    x = data.drop(columns=["expenses"]).values
    y = data.iloc[:, -2].values
    # print(x)
    # print(y)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    preprocessing_join('dataset/medical_insurance.csv')
    preprocessing_expenses_goal('dataset/medical_insurance.csv')
    preprocessing_premium_goal('dataset/medical_insurance.csv')
    
    
    
    