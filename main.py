import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from helpers import get_ticket_info, split_data, metrics_plot, adjusted_metrics_plot


def load_data(path):
    data = pd.read_csv(path)
    data['priority'] = 'H'
    inds = data[data['EVENT_CODE'].isin(['DI', 'DI1', 'DI2', 'ET1', '*T2', 'DV', 'FT', '*FT', 'TB1'])].index
    data.loc[data.index.isin(inds), "priority"] = 'L'
    data[['CUSTOMER_ID', 'RLP', 'TX_ID', 'EVENT_CODE', 'ALARM_DESCRIPTION', 'AREA', 'ZONE', 'ALARM_DETAIL', 'COMMENT', 'priority']] = data[['CUSTOMER_ID', 'RLP', 'TX_ID', 'EVENT_CODE', 'ALARM_DESCRIPTION', 'AREA', 'ZONE', 'ALARM_DETAIL', 'COMMENT', 'priority']].astype(str)
    data = data.drop(['CUSTOMER_SERIAL_NO', 'COMMENT_LOGDATE'], axis=1)
    data['ticket'] = data['COMMENT'].astype('str').apply(get_ticket_info)
    return data

def preprocess_data(data):
    grouped_data = data.groupby(['CUSTOMER_ID', 'ALARM_DESCRIPTION', 'ALARM_DETAIL']).agg(
        reps=('TX_ID', 'count'),
        num_unique_labels=('ticket', lambda x: x.nunique()),
        unique_labels=('ticket', lambda x: list(x.unique()))
    ).reset_index().sort_values(by='num_unique_labels')
    grouped_data['ticket'] = grouped_data['unique_labels'].apply(lambda x: x[0] if len(x) == 1 else True)
    return grouped_data



def main():
    mlflow.start_run()
    data_path = "./data/att_sample_data_raw.csv"
    test_ratio = 0.2
    output_path = "./model_output"
    
    data = load_data(data_path)
    processed_data = preprocess_data(data)
    print(processed_data.head())
    feature_cols = ['CUSTOMER_ID', 'ALARM_DESCRIPTION']
    target = 'ticket'
    X, Y = processed_data[feature_cols], processed_data[target].astype(int)

    x_train, x_test, y_train, y_test = split_data(X, Y, test_ratio=test_ratio)

    pipeline = Pipeline([
        ('encoder', TargetEncoder()),
        ('model', xgb.XGBClassifier(random_state=7))
    ])

    sample_weight = compute_sample_weight('balanced', y_train)
    pipeline.fit(x_train, y_train, model_weight=sample_weight)


    model_file = Path(output_path) / 'pipeline.pkl'
    joblib.dump(pipeline, model_file)


    report_train, report_test = metrics_plot(pipeline, x_train, x_test, y_train, y_test)
    with open(Path(output_path) / 'classification_report_train.txt', 'w') as f:
        f.write(report_train)
    with open(Path(output_path) / 'classification_report_test.txt', 'w') as f:
        f.write(report_test)

    adjusted_report_train, adjusted_report_test = adjusted_metrics_plot(pipeline, x_train, x_test, y_train, y_test)
    with open(Path(output_path) / 'adjusted_classification_report_train.txt', 'w') as f:
        f.write(adjusted_report_train)
    with open(Path(output_path) / 'adjusted_classification_report_test.txt', 'w') as f:
        f.write(adjusted_report_test)

    print('Model training done successfully')
    mlflow.end_run()

if __name__ == "__main__":
    main()










# import argparse
# import pandas as pd
# import numpy as np
# import pickle
# from pathlib import Path
# import mlflow
# from helpers import get_ticket_info, split_data, target_encode, train_model, metrics_plot, adjusted_metrics_plot
# from sklearn.externals import joblib
# import joblib
# from sklearn.utils.class_weight import compute_sample_weight
# from sklearn.pipeline import Pipeline
# from category_encoders import TargetEncoder
