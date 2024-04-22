import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import joblib
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import pickle
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

print("XGBoost version:", xgb.__version__)

from pathlib import Path

def get_ticket_info(txt):
    str_exp = r'(?<!\d)\d{16}(?!\d)'
    tickets = re.findall(str_exp, txt)
    tickets = [item.strip() for item in tickets]
    return len(tickets)>0



def split_data(features, target, test_ratio):
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_ratio, train_size=1-test_ratio, random_state=7, shuffle=True, stratify=target)
    
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return x_train, x_test, y_train, y_test


def target_encode(x_train, x_test, y_train, cat_cols):

    
    target_encoder = TargetEncoder().fit(x_train[cat_cols],y_train)
    
    x_train_enc = pd.DataFrame(target_encoder.transform(x_train[cat_cols]), columns = cat_cols)
    x_test_enc = pd.DataFrame(target_encoder.transform(x_test[cat_cols]), columns = cat_cols)
    
    x_train_enc = pd.concat([x_train_enc, x_train.drop(cat_cols,axis=1)], axis=1)
    x_test_enc = pd.concat([x_test_enc, x_test.drop(cat_cols,axis=1)], axis=1)


    
    return x_train_enc, x_test_enc, target_encoder




def train_and_save_model(x_train, y_train, output_path, model_name):
 
    pipeline = Pipeline([
        ('encoder', TargetEncoder()), 
        ('model', xgb.XGBClassifier(random_state=7, sample_weight=compute_sample_weight("balanced", y_train)))
    ])
    
    pipeline.fit(x_train, y_train)

   
    model_file = Path(output_path) / f'{model_name}.pkl'
    joblib.dump(pipeline, model_file)


def get_prediction(model, features):
    return model.predict(features), model.predict_proba(features)


def metrics_plot(model, x_train_enc, x_test_enc, y_train, y_test):
    
    y_train_pred, y_train_score = get_prediction(model, x_train_enc)
    y_test_pred, y_test_score = get_prediction(model, x_test_enc)
    
    actual, predicted, actual_te, predicted_te = y_train, y_train_pred, y_test, y_test_pred
    
    # cm = confusion_matrix(actual, predicted)

    # fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    # fig.suptitle('Classification results for train and test')
    
    # sns.heatmap(cm, annot = True, fmt = '.2f', xticklabels = ['Not Critical', 'Critical'], yticklabels = ['Not Critical', 'Critical'], ax=axes[0])
    # axes[0].set_title('Train')
    # axes[0].set_ylabel('Actual')
    # axes[0].set_xlabel('Predicted')

    # cm_te = confusion_matrix(actual_te, predicted_te)

    # sns.heatmap(cm_te, annot = True, fmt = '.2f', xticklabels = ['Not Critical', 'Critical'], yticklabels = ['Not Critical', 'Critical'], ax=axes[1])
    # axes[1].set_title('Test')
    # axes[1].set_ylabel('Actual')
    # axes[1].set_xlabel('Predicted')
    
    # plt.show()
    
    # print(f'+++++++++++++++++++++++ TRAIN REPORT +++++++++++++++++++++++\n',
    #       classification_report(actual, predicted),
    #       f'\n+++++++++++++++++++++++ TEST REPORT +++++++++++++++++++++++\n',
    #       classification_report(actual_te, predicted_te))

    report_train = classification_report(actual, predicted)
    report_test = classification_report(actual_te, predicted_te)

    return report_train, report_test


    

def adjusted_metrics_plot(model, x_train_enc, x_test_enc, y_train, y_test):
    
    y_train_pred, y_train_score = get_prediction(model, x_train_enc)
    y_test_pred, y_test_score = get_prediction(model, x_test_enc)
    
    actual, predicted_prob, actual_te, predicted_prob_te = y_train, y_train_score[:,1], y_test, y_test_score[:,1]

    precision, recall, thresholds = precision_recall_curve(actual, predicted_prob)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    best_threshold, best_fscore = thresholds[ix], fscore[ix]
    # print(f'best_thr: {best_threshold}\nbest_fscore: {best_fscore}')
    
    # fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    # fig.suptitle('Classification results for train and test')
    
    y_pred_adj = np.where(predicted_prob >= best_threshold, 1,0)
    cm = confusion_matrix(actual, y_pred_adj)
    # sns.heatmap(cm, annot = True, fmt = '.2f', xticklabels = ['Not Critical', 'Critical'], yticklabels = ['Not Critical', 'Critical'], ax=axes[0])
    # axes[0].set_title('Train')
    # axes[0].set_ylabel('Actual')
    # axes[0].set_xlabel('Predicted')
    
    y_pred_adj_te = np.where(predicted_prob_te >= best_threshold, 1,0)
    cm_te = confusion_matrix(actual_te, y_pred_adj_te)
    # sns.heatmap(cm_te, annot = True, fmt = '.2f', xticklabels = ['Not Critical', 'Critical'], yticklabels = ['Not Critical', 'Critical'], ax=axes[1])
    # axes[1].set_title('Test')
    # axes[1].set_ylabel('Actual')
    # axes[1].set_xlabel('Predicted')
    
    # plt.show()
    
    # print(f'+++++++++++++++++++++++ TRAIN REPORT +++++++++++++++++++++++\n',
    #       classification_report(actual, y_pred_adj),
    #       f'\n+++++++++++++++++++++++ TEST REPORT +++++++++++++++++++++++\n',
    #       classification_report(actual_te, y_pred_adj_te))

    report_train = classification_report(actual, y_pred_adj)
    report_test = classification_report(actual_te, y_pred_adj_te)

    return report_train, report_test
    

def get_decile_analysis(scores, labels):
    ticks = np.arange(0,1.1,.1)
    ints, critical_cnt, all_cnt = [], [], []
    for i in range(len(ticks)-1):
        ints.append(f'{np.round(ticks[i],2)}-{np.round(ticks[i+1],2)}')
        inds = (scores >= ticks[i]) & (scores <ticks[i+1])
        temp_labels = labels[inds]
        critical_cnt.append(sum(temp_labels))
        all_cnt.append(len(temp_labels))    
    return np.array(ints), np.array(critical_cnt), np.array(all_cnt)


def performance_summary(model, x_train_enc, x_test_enc, y_train, y_test, vectorizer=None):
    
    y_train_pred, y_train_score = get_prediction(model, x_train_enc)
    y_test_pred, y_test_score = get_prediction(model, x_test_enc)
    
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_score[:, 1])
    fscore = (2 * precisions * recalls / (precisions + recalls))
    
    print(f'+++++++++++++++++++++++ PERICISION/RECALL CURVE (TRAIN) +++++++++++++++++++++++\n')
    plt.figure()
    plt.plot(thresholds, precisions[:-1], 'b--', label = 'precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    plt.plot(thresholds, fscore[:-1], 'r--', label = 'fscore')
    plt.axvline(x=thresholds[np.argmax(fscore)], color='black', ls='-')
    plt.axvline(x=0.5, color='black', ls='-')
    plt.xlabel('Threshold')
    plt.legend(loc = 'upper left')
    plt.ylim([0, 1])
    plt.grid('minor')
    plt.show()
    
    print(f'\n ++++++++++++++++++++++++++++++++++++++++++++++ ROC-AUC ++++++++++++++++++++++++++++++++++++++++++++++\n')
    print(f"rocauc_train: {roc_auc_score(y_train, y_train_score[:, 1])}")
    print(f"rocauc_test: {roc_auc_score(y_test, y_test_score[:, 1])}")
    
    print(f'\n ++++++++++++++++++++++++++++++++++++++++++++++ CONFUSION MTX ++++++++++++++++++++++++++++++++++++++++++++++\n')
    metrics_plot(model, x_train_enc, x_test_enc, y_train, y_test)
    adjusted_metrics_plot(model, x_train_enc, x_test_enc, y_train, y_test)
    
    print(f'\n++++++++++++++++++++++++++++++++++++++++++++++ FEATURE IMPORTANCE ++++++++++++++++++++++++++++++++++++++++++++++\n')
    plt.figure(figsize=(18,3))
    pd.Series(model.feature_importances_, index=x_train_enc.columns).nlargest(80).plot(kind='bar')
    plt.ylabel('features')
    plt.xlabel('importance')
    plt.grid('minor')
    plt.show()
    
    print(f'\n++++++++++++++++++++++++++++++++++++++++++++++ DECILE ANALYSIS ++++++++++++++++++++++++++++++++++++++++++++++\n')
    ints, critical_cnt, all_cnt = get_decile_analysis(y_test_score[:,1], y_test)
    plt.figure(figsize=(12,4))
    ax = sns.lineplot(x=ints[::-1], y=critical_cnt[::-1]/all_cnt[::-1], marker="o")
    ax.tick_params(axis='x', rotation=45)
    ax.set(xlabel='(model) probability interval', ylabel='class 1 (important) density (percentage)')
    plt.grid('minor')
    plt.show()
    
    graph_df = pd.DataFrame(zip(ints[::-1],critical_cnt[::-1],all_cnt[::-1]-critical_cnt[::-1]), columns=['interval','critical_cnt','non-critical_cnt']).set_index('interval')
    fig, ax = plt.subplots(figsize = (12, 4))
    graph_df.plot(kind='bar', stacked=True, color=['red','blue'], ax=ax)
    plt.ylabel('counts')
    plt.xlabel('(model) probability interval')
    plt.grid('minor')
    plt.show()
