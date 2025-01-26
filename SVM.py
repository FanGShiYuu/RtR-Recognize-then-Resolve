import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data1 = pd.read_csv('argo_data/interaction_av_lt/left_turn_rush_output.csv')
    data2 = pd.read_csv('argo_data/interaction_av_lt/left_turn_yield_output.csv')
    data3 = pd.read_csv('argo_data/interaction_av_gs/left_turn_rush_output.csv')
    data4 = pd.read_csv('argo_data/interaction_av_gs/left_turn_yield_output.csv')
    
    merged_data_av_lt = pd.concat([data1, data2])
    merged_data_av_gs = pd.concat([data3, data4])
    merged_data_all = pd.concat([merged_data_av_lt, merged_data_av_gs])
    
    return merged_data_av_lt, merged_data_av_gs, merged_data_all

def train_svm_model(X_train, y_train):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(svm_model, X_test, y_test):
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

def plot_accuracy_by_bin(test_data):
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '>10']
    test_data['ttcp_bin'] = pd.cut(test_data['ego_ttcp'], bins=bins, labels=labels)

    accuracy_by_bin = test_data.groupby('ttcp_bin').apply(
        lambda x: pd.Series({
            'accuracy': accuracy_score(x['y_true'], x['y_pred']),
            'sample_count': x.shape[0]
        })
    )
    
    accuracy_by_bin.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('ego_ttcp range')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()

def main():
    merged_data_av_lt, merged_data_av_gs, merged_data_all = load_data()

    for i, merged_data in enumerate([merged_data_av_lt, merged_data_av_gs, merged_data_all]):
        data_type = 'av_lt' if i == 0 else 'av_gs' if i == 1 else 'all'
        
        X = merged_data[['agent_ttcp', 'ego_ttcp', 'ego_a_c']]
        y = merged_data['av_pass_first'].map({True: 0, False: 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        
        print(f'Training model for {data_type}...')
        svm_model = train_svm_model(X_train, y_train)
        
        y_pred, accuracy = evaluate_model(svm_model, X_test, y_test)
        
        print(f'Model {data_type} Accuracy: {accuracy:.2f}')
        
        test_data = X_test.copy()
        test_data['y_true'] = y_test.values
        test_data['y_pred'] = y_pred

        plot_accuracy_by_bin(test_data)

if __name__ == "__main__":
    main()
