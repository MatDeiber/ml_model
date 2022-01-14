from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    coastal_data = pd.read_csv('data/database_with_artificial_values.csv')
    
    X = coastal_data.drop(columns = ['Damage [%]'], axis = 1)
    y = coastal_data['Damage [%]']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    return X_train, X_test, y_train, y_test
     