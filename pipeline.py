from pyspark.sql import functions as F
from pyspark.sql.functions import when
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, recall_score, precision_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
import math


from lifelines import KaplanMeierFitter 

@transform_pandas(
    Output(rid="ri.vector.main.execute.d57e86ae-f217-4d6f-b0c2-e6c0efb581c9"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    predictions_by_date=Input(rid="ri.foundry.main.dataset.647f3798-efd2-45ed-9a54-303cfb2c997e")
)
def sensitivity_analysis_1(predictions_by_date, Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_):
    # Using pred_300 as the computable phenotype
    thresholds = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95']
    df1 = predictions_by_date
    df2 = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_
    df3 = df2.join(df1, 'person_id', 'left')
    for i in thresholds:
        df3 = df3.withColumn('LC_u09_computable_phenotype_threshold_' + i[2:], (F.when(F.col('y_pred_300') >= float(i), 1).otherwise(0)))

    dic_precision = []
    dic_recall = []
    dic_f1 = []
    # Using precision score, recall score and f1 score to determine which threshold is the best
    for i in thresholds:
       
        precision = precision_score(df3.select('Long_COVID_diagnosis_post_covid_indicator').toPandas(), df3.select('LC_u09_computable_phenotype_threshold_' + i[2:]).toPandas())
        recall = recall_score(df3.select('Long_COVID_diagnosis_post_covid_indicator').toPandas(), df3.select('LC_u09_computable_phenotype_threshold_' + i[2:]).toPandas())
        f1 = f1_score(df3.select('Long_COVID_diagnosis_post_covid_indicator').toPandas(), df3.select('LC_u09_computable_phenotype_threshold_' + i[2:]).toPandas())
        dic_precision.append((i, precision))
        dic_recall.append((i, recall))
        dic_f1.append((i, f1))
    
    sorted_precision = sorted(dic_precision, key=lambda x:x[1], reverse = True)
    sorted_recall = sorted(dic_recall, key=lambda x:x[1], reverse = True)
    sorted_f1 = sorted(dic_f1, key=lambda x:x[1], reverse = True)
    print(sorted_precision)
    print(sorted_recall)
    print(sorted_f1)
    return df3

@transform_pandas(
    Output(rid="ri.vector.main.execute.1fda7cdf-629b-43e2-a38e-affea03e65d5"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    predictions_by_date=Input(rid="ri.foundry.main.dataset.647f3798-efd2-45ed-9a54-303cfb2c997e")
)
def sensitivity_analysis_2(predictions_by_date, Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_):
    # Using max of pred_100 through pred_300 as the computable phenotype
    thresholds = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95']
    df1 = predictions_by_date
    df1 = df1.withColumn('max_ypred', F.greatest('y_pred_100', 'y_pred_130', 'y_pred_160', 'y_pred_190', 'y_pred_220', 'y_pred_250', 'y_pred_280', 'y_pred_300'))
    df2 = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_
    df3 = df2.join(df1, 'person_id', 'left')
    for i in thresholds:
        df3 = df3.withColumn('LC_u09_computable_phenotype_threshold_' + i[2:], (F.when(F.col('max_ypred') >= float(i), 1).otherwise(0)))

    dic_precision = []
    dic_recall = []
    dic_f1 = []
    # Using precision score, recall score and f1 score to determine which threshold is the best
    for i in thresholds:
       
        precision = precision_score(df3.select('Long_COVID_diagnosis_post_covid_indicator').toPandas(), df3.select('LC_u09_computable_phenotype_threshold_' + i[2:]).toPandas())
        recall = recall_score(df3.select('Long_COVID_diagnosis_post_covid_indicator').toPandas(), df3.select('LC_u09_computable_phenotype_threshold_' + i[2:]).toPandas())
        f1 = f1_score(df3.select('Long_COVID_diagnosis_post_covid_indicator').toPandas(), df3.select('LC_u09_computable_phenotype_threshold_' + i[2:]).toPandas())
        dic_precision.append((i, precision))
        dic_recall.append((i, recall))
        dic_f1.append((i, f1))
    
    sorted_precision = sorted(dic_precision, key=lambda x:x[1], reverse = True)
    sorted_recall = sorted(dic_recall, key=lambda x:x[1], reverse = True)
    sorted_f1 = sorted(dic_f1, key=lambda x:x[1], reverse = True)
    print(sorted_precision)
    print(sorted_recall)
    print(sorted_f1)
    return df3

