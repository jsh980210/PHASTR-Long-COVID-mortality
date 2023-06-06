from pyspark.sql import functions as F
from pyspark.sql.functions import when
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, recall_score)
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


#from lifelines import KaplanMeierFitter 

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4f161901-2489-46e9-b59a-9bbcdec5834c"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    predictions_by_date=Input(rid="ri.foundry.main.dataset.647f3798-efd2-45ed-9a54-303cfb2c997e")
)
def Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype(Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_, predictions_by_date):

    df1 = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_
    df2 = predictions_by_date.select('person_id', 'y_pred_300')

    df = df1.join(df2, 'person_id', 'left')
    df = df.withColumn('LC_u09_computable_phenotype_threshold_75', (F.when(F.col('y_pred_300') >= 0.75, 1).otherwise(0)))

    df = df.drop('y_pred_300')

    return df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.364ef2c6-eeb6-4759-babe-e62045fa3ae8")
)
def analysis_2_PASC_case_cohort_2a(Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype):
    
    # COVID_first_poslab_or_diagnosis_date as index date
    df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype
    df = df.filter((df.Long_COVID_diagnosis_post_covid_indicator == 1) | (df.Long_COVID_clinic_visit_post_covid_indicator == 1))

    # Age >= 18
    df = df.filter(df.age_at_covid >= 18)
    return df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.95970a65-6859-407c-8218-b9f29af84002")
)
def analysis_2_PASC_case_cohort_2b(Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype):
    
    # COVID_first_poslab_or_diagnosis_date as index date
    df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype
    df = df.filter((df.Long_COVID_diagnosis_post_covid_indicator == 1) | (df.Long_COVID_clinic_visit_post_covid_indicator == 1) | (df.LC_u09_computable_phenotype_threshold_75 == 1))

    # Age >= 18
    df = df.filter(df.age_at_covid >= 18)
    return df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.65008359-a0ba-451f-b5ea-ad2af49bc191"),
    analysis_2_PASC_case_cohort_2a=Input(rid="ri.vector.main.execute.364ef2c6-eeb6-4759-babe-e62045fa3ae8")
)
def analysis_2a(analysis_2_PASC_case_cohort_2a, cci_score_covid_positive):
    df = analysis_2_PASC_case_cohort_2a.join(cci_score_covid_positive, 'person_id', 'left')
    df = df.select('race_ethnicity', 'sex', 'CCI_score_up_through_index_date', 'TUBERCULOSIS_before_or_day_of_covid_indicator', 'MILDLIVERDISEASE_before_or_day_of_covid_indicator', 'MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator', 'THALASSEMIA_before_or_day_of_covid_indicator', 'RHEUMATOLOGICDISEASE_before_or_day_of_covid_indicator', 'DEMENTIA_before_or_day_of_covid_indicator', 'CONGESTIVEHEARTFAILURE_before_or_day_of_covid_indicator', 'SUBSTANCEUSEDISORDER_before_or_day_of_covid_indicator', 'DOWNSYNDROME_before_or_day_of_covid_indicator', 'KIDNEYDISEASE_before_or_day_of_covid_indicator', 'MALIGNANTCANCER_before_or_day_of_covid_indicator', 'DIABETESCOMPLICATED_before_or_day_of_covid_indicator', 'CEREBROVASCULARDISEASE_before_or_day_of_covid_indicator', 'PERIPHERALVASCULARDISEASE_before_or_day_of_covid_indicator', 'PREGNANCY_before_or_day_of_covid_indicator', 'HEARTFAILURE_before_or_day_of_covid_indicator', 'HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator', 'PSYCHOSIS_before_or_day_of_covid_indicator', 'OBESITY_before_or_day_of_covid_indicator', 'CORONARYARTERYDISEASE_before_or_day_of_covid_indicator', 'SYSTEMICCORTICOSTEROIDS_before_or_day_of_covid_indicator', 'DEPRESSION_before_or_day_of_covid_indicator', 'METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator', 'HIVINFECTION_before_or_day_of_covid_indicator', 'CHRONICLUNGDISEASE_before_or_day_of_covid_indicator', 'PEPTICULCER_before_or_day_of_covid_indicator', 'SICKLECELLDISEASE_before_or_day_of_covid_indicator', 'MYOCARDIALINFARCTION_before_or_day_of_covid_indicator', 'DIABETESUNCOMPLICATED_before_or_day_of_covid_indicator', 'CARDIOMYOPATHIES_before_or_day_of_covid_indicator', 'HYPERTENSION_before_or_day_of_covid_indicator', 'OTHERIMMUNOCOMPROMISED_before_or_day_of_covid_indicator', 'Antibody_Neg_before_or_day_of_covid_indicator', 'PULMONARYEMBOLISM_before_or_day_of_covid_indicator', 'TOBACCOSMOKER_before_or_day_of_covid_indicator', 'SOLIDORGANORBLOODSTEMCELLTRANSPLANT_before_or_day_of_covid_indicator', 'Antibody_Pos_before_or_day_of_covid_indicator', 'BMI_max_observed_or_calculated_post_covid', 'number_of_COVID_vaccine_doses_post_covid', 'COVID_associated_hospitalization_indicator', 'COVID_patient_death_indicator')

    df = df.withColumn('White_Non_Hispanic', when(df.race_ethnicity == 'White Non-Hispanic', 1).otherwise(0))
    df = df.withColumn('Hispanic_or_Latino_Any_Race', when(df.race_ethnicity == 'Hispanic or Latino Any Race', 1).otherwise(0))
    df = df.withColumn('Asian_Non_Hispanic', when(df.race_ethnicity == 'Asian Non-Hispanic', 1).otherwise(0))
    df = df.withColumn('Black_or_African_American_Non_Hispanic', when(df.race_ethnicity == 'Black or African American Non-Hispanic', 1).otherwise(0))
    df = df.withColumn('Other_Non_Hispanic', when((df.race_ethnicity == 'Other Non-Hispanic') | (df.race_ethnicity == 'Native Hawaiian or Other Pacific Islander Non-Hispanic') | (df.race_ethnicity == 'American Indian or Alaska Native Non-Hispanic'), 1).otherwise(0))
    df = df.withColumn('Unknown_race', when(df.race_ethnicity == 'Unknown', 1).otherwise(0))

    df = df.withColumn('Male', when((df.sex == 'MALE'), 1).otherwise(0))

    return df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.c1f45ebd-5dc5-444e-a28a-e6887d05d985"),
    analysis_2a=Input(rid="ri.vector.main.execute.65008359-a0ba-451f-b5ea-ad2af49bc191")
)
def analysis_2a_xgboost(analysis_2a):
    df = analysis_2a.drop(columns = ['race_ethnicity', 'sex'])
    
    y = df['COVID_patient_death_indicator']
    X = df.drop(columns = ['COVID_patient_death_indicator'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    classifier = XGBClassifier(random_state = 42)
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)[:, 1]
    #print(y_pred)
    y_pred_cat = np.where(y_score >= 0.132, 1, 0)

    print(roc_auc_score(y_test, y_score))
    print(recall_score(y_test, y_pred_cat))
    print(confusion_matrix(y_test, y_pred_cat))
    plot_importance(classifier)

    pred_scores = dict(y_true=y_test, y_score=y_score)
    cols = ['False Positive Rate', 'True Positive Rate', 'threshold']
    roc = pd.DataFrame(dict(zip(cols, roc_curve(**pred_scores))))

    precision, recall, ts = precision_recall_curve(y_true=y_test, probas_pred=y_score)
    pr_curve = pd.DataFrame({'Precision': precision, 'Recall': recall})

    f1 = pd.Series({t: f1_score(y_true=y_test, y_pred=y_score>t) for t in ts})
    best_threshold = f1.idxmax()
    print(best_threshold)

    fig, axes = plt.subplots(ncols=3, figsize=(13, 5))

    sns.scatterplot(x='False Positive Rate', y='True Positive Rate', data=roc, s=50, legend=False, ax=axes[0])
    axes[0].plot('False Positive Rate', 'True Positive Rate', data=roc, lw=1, color='k')
    axes[0].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k', ls='--', lw=1)
    axes[0].fill_between(y1=roc['True Positive Rate'], x=roc['False Positive Rate'], alpha=.3, color='red')
    axes[0].set_title('Receiver Operating Characteristic')

    sns.scatterplot(x='Recall', y='Precision', data=pr_curve, ax=axes[1])
    axes[1].set_ylim(0,1)
    axes[1].set_title('Precision-Recall Curve')

    #print(f1)
    axes[2].plot(f1)
    axes[2].set_xlabel('Threshold')
    axes[2].axvline(best_threshold, lw=1, ls='--', color='k')
    #axes[2].text(text=f'Max F1 @ {best_threshold:.2f}', x=.60, y=.95, s=5)
    
    fig.suptitle(f'roc_auc_score = {round(roc_auc_score(**pred_scores),2)}', fontsize=24)
    fig.tight_layout()
    #fig.savefig("roc.png")
    plt.subplots_adjust(top=.8)
    plt.show()
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.463afdd2-d86e-47a5-b609-1f70641b88b8"),
    analysis_2a=Input(rid="ri.vector.main.execute.65008359-a0ba-451f-b5ea-ad2af49bc191")
)
def analysis_2a_xgboost_cv(analysis_2a):
    df = analysis_2a.drop(columns = ['race_ethnicity', 'sex'])
    
    y = df['COVID_patient_death_indicator']
    X = df.drop(columns = ['COVID_patient_death_indicator'])
    n_splits = 5
    n_repeats = 5
    
    

    features = list(X.columns)
    features_pd = pd.DataFrame (features, columns = ['feature'])
    
    X = X.to_numpy()

    #Mean AUC of 0.91 +/- 0.02:
    classifier =  XGBClassifier(colsample_bytree=0.1, gamma=0.4, learning_rate=0.09, max_depth=8, min_child_weight=0, n_estimators=400, subsample=0.9, random_state=42)

    #cv = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state=42)

    tprs = []
    aucs = []

    # 100 evenly spaced points from 0 to 1
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    # Plot the individual ROC curves from the split
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict_proba(X[test])
        feature_importances = pd.DataFrame(classifier.feature_importances_, index = features, columns=[('importance' + str(i))])
        features_pd = features_pd.join(feature_importances, 'feature', 'inner')

        viz = plot_roc_curve(classifier, X[test], y[test],
                            name='ROC fold {}'.format(i),
                            label ='_nolegend_',
                            alpha=0.3, lw=1, ax=ax)
        
        # Interpolate the calculated TPR at the evenly spaced points, given the calculated TPR and FPR
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Plot the random classifier line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random Classifier', alpha=.8)

    # Calculate and plot the mean of all the ROC curves from the splits
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Standard Error = Standard Deviation / SQRT(n)
    std_err_auc = std_auc / (math.sqrt(len(aucs)))

    # 95% confidence interval = 1.96 * std_err
    confidence_interval = 1.96 * std_err_auc

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot standard deviation of ROC curves, and fill the space
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title= 'ROC Curve:  {}-fold Cross-Validation, {} Repeats'.format(n_splits, n_repeats) )
    ax.legend(loc="lower right")

    set_output_image_type('svg')
    plt.rcParams['svg.fonttype'] = 'none'
    
    plt.show()

    return features_pd
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.b573a346-1982-4e97-8a8c-900989954f0c"),
    analysis_2a_xgboost_cv=Input(rid="ri.vector.main.execute.463afdd2-d86e-47a5-b609-1f70641b88b8")
)
def analysis_2a_xgboost_cv_feature_importance(analysis_2a_xgboost_cv):

    df = analysis_2a_xgboost_cv
    df_features = analysis_2a_xgboost_cv['feature']

    df = df.drop(columns = ['feature'])

    df['mean'] = (df.mean(axis = 1))

    df['features'] = df_features

    df = df.sort_values("mean", ascending = False).head(50)
    df.index = df["features"]
    plt.figure(figsize = (7, 14))
    sns.barplot(x = df["mean"], y = df["features"], palette = sns.color_palette("RdYlBu", df.shape[0]))
    plt.tight_layout()
    plt.show()

    return(df)

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.0027efbc-bc53-48b2-bd5a-a6feb4d078cf"),
    analysis_2_PASC_case_cohort_2b=Input(rid="ri.vector.main.execute.95970a65-6859-407c-8218-b9f29af84002")
)
def analysis_2b(analysis_2_PASC_case_cohort_2b, cci_score_covid_positive):
    df = analysis_2_PASC_case_cohort_2b.join(cci_score_covid_positive, 'person_id', 'left')
    df = df.select('race_ethnicity', 'sex', 'CCI_score_up_through_index_date', 'TUBERCULOSIS_before_or_day_of_covid_indicator', 'MILDLIVERDISEASE_before_or_day_of_covid_indicator', 'MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator', 'THALASSEMIA_before_or_day_of_covid_indicator', 'RHEUMATOLOGICDISEASE_before_or_day_of_covid_indicator', 'DEMENTIA_before_or_day_of_covid_indicator', 'CONGESTIVEHEARTFAILURE_before_or_day_of_covid_indicator', 'SUBSTANCEUSEDISORDER_before_or_day_of_covid_indicator', 'DOWNSYNDROME_before_or_day_of_covid_indicator', 'KIDNEYDISEASE_before_or_day_of_covid_indicator', 'MALIGNANTCANCER_before_or_day_of_covid_indicator', 'DIABETESCOMPLICATED_before_or_day_of_covid_indicator', 'CEREBROVASCULARDISEASE_before_or_day_of_covid_indicator', 'PERIPHERALVASCULARDISEASE_before_or_day_of_covid_indicator', 'PREGNANCY_before_or_day_of_covid_indicator', 'HEARTFAILURE_before_or_day_of_covid_indicator', 'HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator', 'PSYCHOSIS_before_or_day_of_covid_indicator', 'OBESITY_before_or_day_of_covid_indicator', 'CORONARYARTERYDISEASE_before_or_day_of_covid_indicator', 'SYSTEMICCORTICOSTEROIDS_before_or_day_of_covid_indicator', 'DEPRESSION_before_or_day_of_covid_indicator', 'METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator', 'HIVINFECTION_before_or_day_of_covid_indicator', 'CHRONICLUNGDISEASE_before_or_day_of_covid_indicator', 'PEPTICULCER_before_or_day_of_covid_indicator', 'SICKLECELLDISEASE_before_or_day_of_covid_indicator', 'MYOCARDIALINFARCTION_before_or_day_of_covid_indicator', 'DIABETESUNCOMPLICATED_before_or_day_of_covid_indicator', 'CARDIOMYOPATHIES_before_or_day_of_covid_indicator', 'HYPERTENSION_before_or_day_of_covid_indicator', 'OTHERIMMUNOCOMPROMISED_before_or_day_of_covid_indicator', 'Antibody_Neg_before_or_day_of_covid_indicator', 'PULMONARYEMBOLISM_before_or_day_of_covid_indicator', 'TOBACCOSMOKER_before_or_day_of_covid_indicator', 'SOLIDORGANORBLOODSTEMCELLTRANSPLANT_before_or_day_of_covid_indicator', 'Antibody_Pos_before_or_day_of_covid_indicator', 'BMI_max_observed_or_calculated_post_covid', 'number_of_COVID_vaccine_doses_post_covid', 'COVID_associated_hospitalization_indicator', 'COVID_patient_death_indicator')

    df = df.withColumn('White_Non_Hispanic', when(df.race_ethnicity == 'White Non-Hispanic', 1).otherwise(0))
    df = df.withColumn('Hispanic_or_Latino_Any_Race', when(df.race_ethnicity == 'Hispanic or Latino Any Race', 1).otherwise(0))
    df = df.withColumn('Asian_Non_Hispanic', when(df.race_ethnicity == 'Asian Non-Hispanic', 1).otherwise(0))
    df = df.withColumn('Black_or_African_American_Non_Hispanic', when(df.race_ethnicity == 'Black or African American Non-Hispanic', 1).otherwise(0))
    df = df.withColumn('Other_Non_Hispanic', when((df.race_ethnicity == 'Other Non-Hispanic') | (df.race_ethnicity == 'Native Hawaiian or Other Pacific Islander Non-Hispanic') | (df.race_ethnicity == 'American Indian or Alaska Native Non-Hispanic'), 1).otherwise(0))
    df = df.withColumn('Unknown_race', when(df.race_ethnicity == 'Unknown', 1).otherwise(0))

    df = df.withColumn('Male', when((df.sex == 'MALE'), 1).otherwise(0))

    return df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.32e25c39-d215-4579-91c1-c62ab293273d"),
    analysis_2b=Input(rid="ri.vector.main.execute.0027efbc-bc53-48b2-bd5a-a6feb4d078cf")
)
def analysis_2b_xgboost(analysis_2b):
    df = analysis_2b.drop(columns = ['race_ethnicity', 'sex'])
    
    y = df['COVID_patient_death_indicator']
    X = df.drop(columns = ['COVID_patient_death_indicator'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    classifier = XGBClassifier(random_state = 42)
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)[:, 1]
    #print(y_pred)
    y_pred_cat = np.where(y_score >= 0.125, 1, 0)

    print(roc_auc_score(y_test, y_score))
    print(recall_score(y_test, y_pred_cat))
    print(confusion_matrix(y_test, y_pred_cat))
    plot_importance(classifier)

    pred_scores = dict(y_true=y_test, y_score=y_score)
    cols = ['False Positive Rate', 'True Positive Rate', 'threshold']
    roc = pd.DataFrame(dict(zip(cols, roc_curve(**pred_scores))))

    precision, recall, ts = precision_recall_curve(y_true=y_test, probas_pred=y_score)
    pr_curve = pd.DataFrame({'Precision': precision, 'Recall': recall})

    f1 = pd.Series({t: f1_score(y_true=y_test, y_pred=y_score>t) for t in ts})
    best_threshold = f1.idxmax()
    print(best_threshold)

    fig, axes = plt.subplots(ncols=3, figsize=(13, 5))

    sns.scatterplot(x='False Positive Rate', y='True Positive Rate', data=roc, s=50, legend=False, ax=axes[0])
    axes[0].plot('False Positive Rate', 'True Positive Rate', data=roc, lw=1, color='k')
    axes[0].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k', ls='--', lw=1)
    axes[0].fill_between(y1=roc['True Positive Rate'], x=roc['False Positive Rate'], alpha=.3, color='red')
    axes[0].set_title('Receiver Operating Characteristic')

    sns.scatterplot(x='Recall', y='Precision', data=pr_curve, ax=axes[1])
    axes[1].set_ylim(0,1)
    axes[1].set_title('Precision-Recall Curve')

    #print(f1)
    axes[2].plot(f1)
    axes[2].set_xlabel('Threshold')
    axes[2].axvline(best_threshold, lw=1, ls='--', color='k')
    #axes[2].text(text=f'Max F1 @ {best_threshold:.2f}', x=.60, y=.95, s=5)
    
    fig.suptitle(f'roc_auc_score = {round(roc_auc_score(**pred_scores),2)}', fontsize=24)
    fig.tight_layout()
    #fig.savefig("roc.png")
    plt.subplots_adjust(top=.8)
    plt.show()
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.3ea41ca9-1492-469b-8d91-2e43e50fef48"),
    analysis_2b=Input(rid="ri.vector.main.execute.0027efbc-bc53-48b2-bd5a-a6feb4d078cf")
)
def analysis_2b_xgboost_cv(analysis_2b):
    df = analysis_2b.drop(columns = ['race_ethnicity', 'sex'])
    
    y = df['COVID_patient_death_indicator']
    X = df.drop(columns = ['COVID_patient_death_indicator'])
    n_splits = 5
    n_repeats = 5
    
    

    features = list(X.columns)
    features_pd = pd.DataFrame (features, columns = ['feature'])
    
    X = X.to_numpy()

    #Mean AUC of 0.91 +/- 0.02:
    classifier =  XGBClassifier(colsample_bytree=0.1, gamma=0.4, learning_rate=0.09, max_depth=8, min_child_weight=0, n_estimators=400, subsample=0.9, random_state=42)

    #cv = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state=42)

    tprs = []
    aucs = []

    # 100 evenly spaced points from 0 to 1
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    # Plot the individual ROC curves from the split
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict_proba(X[test])
        feature_importances = pd.DataFrame(classifier.feature_importances_, index = features, columns=[('importance' + str(i))])
        features_pd = features_pd.join(feature_importances, 'feature', 'inner')

        viz = plot_roc_curve(classifier, X[test], y[test],
                            name='ROC fold {}'.format(i),
                            label ='_nolegend_',
                            alpha=0.3, lw=1, ax=ax)
        
        # Interpolate the calculated TPR at the evenly spaced points, given the calculated TPR and FPR
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Plot the random classifier line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random Classifier', alpha=.8)

    # Calculate and plot the mean of all the ROC curves from the splits
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Standard Error = Standard Deviation / SQRT(n)
    std_err_auc = std_auc / (math.sqrt(len(aucs)))

    # 95% confidence interval = 1.96 * std_err
    confidence_interval = 1.96 * std_err_auc

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot standard deviation of ROC curves, and fill the space
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title= 'ROC Curve:  {}-fold Cross-Validation, {} Repeats'.format(n_splits, n_repeats) )
    ax.legend(loc="lower right")

    set_output_image_type('svg')
    plt.rcParams['svg.fonttype'] = 'none'
    
    plt.show()

    return features_pd
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.95724237-cc6c-4966-bb38-684faa9244a7"),
    analysis_2b_xgboost_cv=Input(rid="ri.vector.main.execute.3ea41ca9-1492-469b-8d91-2e43e50fef48")
)
def analysis_2b_xgboost_cv_feature_importance(analysis_2b_xgboost_cv):

    df = analysis_2b_xgboost_cv
    df_features = analysis_2b_xgboost_cv['feature']

    df = df.drop(columns = ['feature'])

    df['mean'] = (df.mean(axis = 1))

    df['features'] = df_features

    df = df.sort_values("mean", ascending = False).head(50)
    df.index = df["features"]
    plt.figure(figsize = (7, 14))
    sns.barplot(x = df["mean"], y = df["features"], palette = sns.color_palette("RdYlBu", df.shape[0]))
    plt.tight_layout()
    plt.show()

    return(df)

    

