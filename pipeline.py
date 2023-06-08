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
    Output(rid="ri.vector.main.execute.edc68d09-273b-4824-856a-e23808f2d29a")
)
def Analysis_1_COVID_positive_control_matched(analysis_1_COVID_positive_control_matching, analysis_1_PASC_case_matched):
    df1 = analysis_1_COVID_positive_control_matching
    df2 = (analysis_1_PASC_case_matched.select('person_id')).join(df1, 'person_id', 'inner')

    df3 = df1.filter(df1.long_covid == 0).union(df2)
    

    df4 = df3.groupBy('subclass').agg(F.count('*').alias('count_same_subclass'))

    df5 = df3.join(df4, 'subclass', 'left')

    result = df5.filter(df5.count_same_subclass == 2)

    result = result.filter(result.long_covid == 0)

    

    

    return result

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
    Output(rid="ri.foundry.main.dataset.cabcd0ef-fb38-471c-a325-493a9ca7b458"),
    Logic_Liaison_All_patients_fact_day_table_lds=Input(rid="ri.foundry.main.dataset.fbc0bab4-32b6-4bad-b8c7-0e0e16e3a3bc"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571"),
    analysis_1_PASC_case=Input(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def analysis_1_COVID_negative_control(visit_occurrence, analysis_1_PASC_case, Logic_Liaison_All_patients_summary_facts_table_lds, Logic_Liaison_All_patients_fact_day_table_lds):
    df1 = Logic_Liaison_All_patients_summary_facts_table_lds
    df2 = Logic_Liaison_All_patients_fact_day_table_lds
    df3 = visit_occurrence
    df4 = analysis_1_PASC_case.select('person_id')
    df3 = df3.groupBy('person_id').agg(F.max('visit_start_date').alias('latest_visit_date'))

    # index date: latest negative COVID test date
    df2 = df2.filter(df2.PCR_AG_Neg == 1)
    df2 = df2.groupBy('person_id').agg(F.max('date').alias('latest_PCR_AG_Neg_date'))
    

    result = df1.join(df2, 'person_id', 'left')
    result = result.join(df3, 'person_id', 'left')
    result = result.withColumn('index_date', F.col('latest_PCR_AG_Neg_date'))
    
    # Make the is_long_COVID_dx_site column
    df1 = df1.filter(df1.LL_Long_COVID_diagnosis_indicator == 1)
    long_covid_dx_sites = df1.select(F.collect_set('data_partner_id').alias('data_partner_id')).first()['data_partner_id']    
    result = result.withColumn('is_long_COVID_dx_site', F.when(result.data_partner_id.isin(long_covid_dx_sites), 1).otherwise(0))

    # Make the Oct 2021 index date
    result = result.withColumn('2021oct_index_date', F.lit("2021-10-01"))

    # From a site that is reporting U09.9 in their N3C data
    result = result.filter(result.is_long_COVID_dx_site == 1)

    # At least one visit >=45 days after index date
    result = result.filter(F.datediff(F.col('latest_visit_date'), F.col('latest_PCR_AG_Neg_date')) >= 45)

    # With at least one visit Oct.1, 2021 or later
    result = result.filter(F.datediff(F.col('latest_visit_date'), F.col('2021oct_index_date')) >= 0)

    # Age >= 18
    result = result.filter(result.age >= 18)

    # Exclude confirmed COVID patients
    result = result.filter(result.confirmed_covid_patient == 0)

    # Exclude possible COVID patients
    result = result.filter(result.possible_covid_patient == 0)

    # exclude PASC case
    result = result.join(df4, 'person_id', 'left_anti')

    # Long COVID control label
    result = result.withColumn('long_covid', F.lit(0))

    

    return result
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4bc2a605-ffb8-4a08-b9c2-623fa9730224"),
    analysis_1_COVID_negative_control=Input(rid="ri.foundry.main.dataset.cabcd0ef-fb38-471c-a325-493a9ca7b458"),
    analysis_1_PASC_case=Input(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68")
)
def analysis_1_COVID_negative_control_pre_matching(analysis_1_COVID_negative_control, analysis_1_PASC_case):
    df1 = analysis_1_COVID_negative_control.select('person_id', 'observation_period', 'data_partner_id', 'age', 'index_date', 'long_covid')
    df2 = analysis_1_PASC_case.select('person_id', 'observation_period_post_covid', 'data_partner_id', 'age_at_covid', 'index_date', 'long_covid')
    df2 = df2.withColumnRenamed('observation_period_post_covid', 'observation_period')
    df2 = df2.withColumnRenamed('age_at_covid', 'age')

    df1_sites = df1.select(F.collect_set('data_partner_id').alias('data_partner_id')).first()['data_partner_id']    
    df2_sites = df2.select(F.collect_set('data_partner_id').alias('data_partner_id')).first()['data_partner_id']    
    df1 = df1.filter(df1.data_partner_id.isin(df2_sites))
    df2 = df2.filter(df2.data_partner_id.isin(df1_sites))
    
    df = df1.union(df2)
    df = df.withColumn('age', df.age.cast('int'))
    df = df.withColumn('2020_index_date', F.lit('2020-01-01'))
    df = df.withColumn('index_date_numberofdays_from_20200101', F.datediff('index_date', '2020_index_date'))
    
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0ab2f17b-94f6-4f86-988b-e49c020e9d9f"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype=Input(rid="ri.foundry.main.dataset.4f161901-2489-46e9-b59a-9bbcdec5834c"),
    analysis_1_PASC_case=Input(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def analysis_1_COVID_positive_control(visit_occurrence, analysis_1_PASC_case, Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype):
    # COVID_first_poslab_or_diagnosis_date as index date
    df1 = visit_occurrence
    df2 = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype
    df3 = analysis_1_PASC_case.select('person_id')
    df1 = df1.groupBy('person_id').agg(F.max('visit_start_date').alias('latest_visit_date'))

    df = df2.join(df1, 'person_id', 'left')
    df = df.withColumn('index_date', F.col('COVID_first_poslab_or_diagnosis_date'))
    df = df.withColumn('2021oct_index_date', F.lit('2021-10-01'))

    # Make the is_long_COVID_dx_site column
    df2 = df2.filter(df2.Long_COVID_diagnosis_post_covid_indicator == 1)
    long_covid_dx_sites = df2.select(F.collect_set('data_partner_id').alias('data_partner_id')).first()['data_partner_id']    
    df = df.withColumn('is_long_COVID_dx_site', F.when(df.data_partner_id.isin(long_covid_dx_sites), 1).otherwise(0))
    

    # Select those with at least one visit >= 45 days of index date
    df = df.filter(F.datediff(F.col('latest_visit_date'), F.col('COVID_first_poslab_or_diagnosis_date')) >= 45)

    # From a site that is reporting U09.9 in their N3C data
    df = df.filter(df.is_long_COVID_dx_site == 1)

    # At least one visit Oct.1, 2021 or later
    df = df.filter(F.datediff(F.col('latest_visit_date'), F.col('2021oct_index_date')) >= 0)

    # Age >= 18
    df = df.filter(df.age_at_covid >= 18)

    # exclude PASC case
    df = df.join(df3, 'person_id', 'left_anti')

    # Long COVID control label
    df = df.withColumn('long_covid', F.lit(0))

    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ed878439-adf1-44c1-b96e-3b45eb3b6a2d"),
    analysis_1_COVID_positive_control=Input(rid="ri.foundry.main.dataset.0ab2f17b-94f6-4f86-988b-e49c020e9d9f"),
    analysis_1_PASC_case=Input(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68")
)
def analysis_1_COVID_positive_control_pre_matching(analysis_1_COVID_positive_control, analysis_1_PASC_case):
    df1 = analysis_1_COVID_positive_control.select('person_id', 'observation_period_post_covid', 'data_partner_id', 'age_at_covid', 'index_date', 'long_covid')
    df2 = analysis_1_PASC_case.select('person_id', 'observation_period_post_covid', 'data_partner_id', 'age_at_covid', 'index_date', 'long_covid')

    
    df1_sites = df1.select(F.collect_set('data_partner_id').alias('data_partner_id')).first()['data_partner_id']    
    df2_sites = df2.select(F.collect_set('data_partner_id').alias('data_partner_id')).first()['data_partner_id']    
    df1 = df1.filter(df1.data_partner_id.isin(df2_sites))
    df2 = df2.filter(df2.data_partner_id.isin(df1_sites))
    
    
    df = df1.union(df2)
    df = df.withColumn('age_at_covid', df.age_at_covid.cast('int'))
    df = df.withColumn('2020_index_date', F.lit('2020-01-01'))
    df = df.withColumn('index_date_numberofdays_from_20200101', F.datediff('index_date', '2020_index_date'))
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype=Input(rid="ri.foundry.main.dataset.4f161901-2489-46e9-b59a-9bbcdec5834c")
)
def analysis_1_PASC_case(Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype):
    # COVID positive
    # Now we only have threshold 0.75, and would change the threshold after sensitivity analysis
    # COVID_first_poslab_or_diagnosis_date as index date
    df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype
    df = df.withColumn('index_date', F.col('COVID_first_poslab_or_diagnosis_date'))
    df = df.filter((df.Long_COVID_diagnosis_post_covid_indicator == 1) | (df.Long_COVID_clinic_visit_post_covid_indicator == 1) | (df.LC_u09_computable_phenotype_threshold_75 == 1))
    df = df.filter(df.age_at_covid >= 18)
    # Long COVID case label
    df = df.withColumn('long_covid', F.lit(1))
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1e5e00da-adbf-4c93-8c3d-1a1caf99c4f6"),
    analysis_1_COVID_negative_control_matching=Input(rid="ri.foundry.main.dataset.b29fdc92-4983-44c5-853f-c3117d55cf86"),
    analysis_1_COVID_positive_control_matching=Input(rid="ri.foundry.main.dataset.7aa4122a-d05e-4e3a-999a-88e069107fbd"),
    analysis_1_PASC_case=Input(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68")
)
def analysis_1_PASC_case_matched(analysis_1_PASC_case, analysis_1_COVID_negative_control_matching, analysis_1_COVID_positive_control_matching):
    df1 = analysis_1_PASC_case
    df2 = analysis_1_COVID_negative_control_matching
    df3 = analysis_1_COVID_positive_control_matching
    df2 = (df2.filter(df2.long_covid == 1)).select('person_id')
    df3 = df3.filter(df3.long_covid == 1).select('person_id')

    result = (df1.join(df2, 'person_id', 'inner')).join(df3, 'person_id', 'inner')
    return result
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7d7a7b20-d395-41e5-9804-f9e8bfa34e4f"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype=Input(rid="ri.foundry.main.dataset.4f161901-2489-46e9-b59a-9bbcdec5834c")
)
def analysis_2_PASC_case_cohort_2a(Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype):
    
    # COVID_first_poslab_or_diagnosis_date as index date
    df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype
    df = df.filter((df.Long_COVID_diagnosis_post_covid_indicator == 1) | (df.Long_COVID_clinic_visit_post_covid_indicator == 1))

    # Age >= 18
    df = df.filter(df.age_at_covid >= 18)
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e3640a26-eac1-43b7-b012-261f6dbbd2f3"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype=Input(rid="ri.foundry.main.dataset.4f161901-2489-46e9-b59a-9bbcdec5834c")
)
def analysis_2_PASC_case_cohort_2b(Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype):
    
    # COVID_first_poslab_or_diagnosis_date as index date
    df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_with_computable_phenotype
    df = df.filter((df.Long_COVID_diagnosis_post_covid_indicator == 1) | (df.Long_COVID_clinic_visit_post_covid_indicator == 1) | (df.LC_u09_computable_phenotype_threshold_75 == 1))

    # Age >= 18
    df = df.filter(df.age_at_covid >= 18)
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dfd52b0d-1b4b-49d1-a420-0f3df44e0f8d"),
    analysis_2_PASC_case_cohort_2a=Input(rid="ri.foundry.main.dataset.7d7a7b20-d395-41e5-9804-f9e8bfa34e4f"),
    cci_score_covid_positive=Input(rid="ri.foundry.main.dataset.0d64bb7b-0e57-4c26-8c41-18a3b793fc00")
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
    Output(rid="ri.foundry.main.dataset.4db8d51f-f165-43c0-a98f-31971c43c059"),
    analysis_2a=Input(rid="ri.foundry.main.dataset.dfd52b0d-1b4b-49d1-a420-0f3df44e0f8d")
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
    plt.close()
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
    Output(rid="ri.foundry.main.dataset.df6207be-25db-47f6-893e-ae6c8eb96f3f"),
    analysis_2a=Input(rid="ri.foundry.main.dataset.dfd52b0d-1b4b-49d1-a420-0f3df44e0f8d")
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
    plt.close()
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
    analysis_2a_xgboost_cv=Input(rid="ri.foundry.main.dataset.df6207be-25db-47f6-893e-ae6c8eb96f3f")
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
    Output(rid="ri.foundry.main.dataset.f251c730-78fb-4044-8c57-96c16e3c2011"),
    analysis_2_PASC_case_cohort_2b=Input(rid="ri.foundry.main.dataset.e3640a26-eac1-43b7-b012-261f6dbbd2f3"),
    cci_score_covid_positive=Input(rid="ri.foundry.main.dataset.0d64bb7b-0e57-4c26-8c41-18a3b793fc00")
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
    Output(rid="ri.foundry.main.dataset.40938b04-d7e0-4669-9b4e-57d4be39e92a"),
    analysis_2b=Input(rid="ri.foundry.main.dataset.f251c730-78fb-4044-8c57-96c16e3c2011")
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
    plt.close()
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
    Output(rid="ri.foundry.main.dataset.41c8204d-b51b-4689-b2e0-2d9d25962b11"),
    analysis_2b=Input(rid="ri.foundry.main.dataset.f251c730-78fb-4044-8c57-96c16e3c2011")
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
    plt.close()
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
    analysis_2b_xgboost_cv=Input(rid="ri.foundry.main.dataset.41c8204d-b51b-4689-b2e0-2d9d25962b11")
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

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3ca69c7a-76bc-4a5f-ab5c-8f94f5709789"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198")
)
# [Logic Liaison Template] CCI Score (LDS) (40b7f6fe-9486-4b68-82ad-e8118b47738c): v6
def cci_score(Logic_Liaison_All_patients_summary_facts_table_lds, Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_):

    #user determined template input
    #when True - applies index date and calculates CCI score for confirmed covid positive patients before and up to their covid index date 
    #when False - no index date applied and calculates CCI score for all patients based on indicators noted at any time in their EHR 
    use_only_confirmed_covid_positive = False

    if use_only_confirmed_covid_positive:
        df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ \
            .select('person_id', *[c for c in Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_.columns if '_indicator' in c])

        #calculating CCI components BEFORE and up to covid index
        MI = F.col('MYOCARDIALINFARCTION_before_or_day_of_covid_indicator') * 1
        CHF = F.col('CONGESTIVEHEARTFAILURE_before_or_day_of_covid_indicator') * 1
        PVD = F.col('PERIPHERALVASCULARDISEASE_before_or_day_of_covid_indicator') * 1
        CVD = F.col('CEREBROVASCULARDISEASE_before_or_day_of_covid_indicator') * 1
        DEM = F.col('DEMENTIA_before_or_day_of_covid_indicator') * 1
        CPD = F.col('CHRONICLUNGDISEASE_before_or_day_of_covid_indicator') * 1
        RD = F.col('RHEUMATOLOGICDISEASE_before_or_day_of_covid_indicator') * 1
        PEP = F.col('PEPTICULCER_before_or_day_of_covid_indicator') * 1
        LIV = F.when(F.col('MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator')==1, F.col('MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator') * 3) \
            .when(F.col('MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator')==0, F.col('MILDLIVERDISEASE_before_or_day_of_covid_indicator') * 1) \
            .otherwise(0)
        DIA = F.when(F.col('DIABETESCOMPLICATED_before_or_day_of_covid_indicator')==1, F.col('DIABETESCOMPLICATED_before_or_day_of_covid_indicator') * 2) \
            .when(F.col('DIABETESCOMPLICATED_before_or_day_of_covid_indicator')==0, F.col('DIABETESUNCOMPLICATED_before_or_day_of_covid_indicator') * 1) \
            .otherwise(0)
        HEM = F.col('HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator') * 2
        REN = F.col('KIDNEYDISEASE_before_or_day_of_covid_indicator') * 2
        #MALIGNANT CANCER concept set covers Leukemia and Lymphoma specified in the Charlson Comorbidity Index calculation
        CAN = F.when(F.col('METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator')==1, F.col('METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator') * 6) \
            .when(F.col('METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator')==0, F.col('MALIGNANTCANCER_before_or_day_of_covid_indicator') * 2) \
            .otherwise(0)
        HIV = F.col('HIVINFECTION_before_or_day_of_covid_indicator') * 6

        #calculate CCI score for patients, age not included as it is not explicitly included in the reference calculation used and is often already accounted for as a covariate in most studies
        df = df.withColumn('CCI_score_up_through_index_date', F.lit(MI + CHF + PVD + CVD + DEM + CPD + RD + PEP + LIV + DIA + HEM + REN + CAN + HIV).cast(IntegerType())) \
            .select('person_id', 'CCI_score_up_through_index_date')

    else:
        df = Logic_Liaison_All_patients_summary_facts_table_lds \
            .select('person_id', *[c for c in Logic_Liaison_All_patients_summary_facts_table_lds.columns if '_indicator' in c])

        #calculating CCI components all-time
        MI = F.col('MYOCARDIALINFARCTION_indicator') * 1
        CHF = F.col('CONGESTIVEHEARTFAILURE_indicator') * 1
        PVD = F.col('PERIPHERALVASCULARDISEASE_indicator') * 1
        CVD = F.col('CEREBROVASCULARDISEASE_indicator') * 1
        DEM = F.col('DEMENTIA_indicator') * 1
        CPD = F.col('CHRONICLUNGDISEASE_indicator') * 1
        RD = F.col('RHEUMATOLOGICDISEASE_indicator') * 1
        PEP = F.col('PEPTICULCER_indicator') * 1
        LIV = F.when(F.col('MODERATESEVERELIVERDISEASE_indicator')==1, F.col('MODERATESEVERELIVERDISEASE_indicator') * 3) \
            .when(F.col('MODERATESEVERELIVERDISEASE_indicator')==0, F.col('MILDLIVERDISEASE_indicator') * 1) \
            .otherwise(0)
        DIA = F.when(F.col('DIABETESCOMPLICATED_indicator')==1, F.col('DIABETESCOMPLICATED_indicator') * 2) \
            .when(F.col('DIABETESCOMPLICATED_indicator')==0, F.col('DIABETESUNCOMPLICATED_indicator') * 1) \
            .otherwise(0)
        HEM = F.col('HEMIPLEGIAORPARAPLEGIA_indicator') * 2
        REN = F.col('KIDNEYDISEASE_indicator') * 2
        #MALIGNANT CANCER concept set covers Leukemia and Lymphoma specified in the Charlson Comorbidity Index calculation
        CAN = F.when(F.col('METASTATICSOLIDTUMORCANCERS_indicator')==1, F.col('METASTATICSOLIDTUMORCANCERS_indicator') * 6) \
            .when(F.col('METASTATICSOLIDTUMORCANCERS_indicator')==0, F.col('MALIGNANTCANCER_indicator') * 2) \
            .otherwise(0)
        HIV = F.col('HIVINFECTION_indicator') * 6

        #calculate CCI score for patients, age not included as it is not explicitly included in the reference calculation used and is often already accounted for as a covariate in most studies
        df = df.withColumn('CCI_score', F.lit(MI + CHF + PVD + CVD + DEM + CPD + RD + PEP + LIV + DIA + HEM + REN + CAN + HIV).cast(IntegerType())) \
            .select('person_id', 'CCI_score')

    return df

#################################################
## Global imports and functions included below ##
#################################################

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0d64bb7b-0e57-4c26-8c41-18a3b793fc00"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198")
)
# [Logic Liaison Template] CCI Score (LDS) (40b7f6fe-9486-4b68-82ad-e8118b47738c): v6
def cci_score_covid_positive(Logic_Liaison_All_patients_summary_facts_table_lds, Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_):

    #user determined template input
    #when True - applies index date and calculates CCI score for confirmed covid positive patients before and up to their covid index date 
    #when False - no index date applied and calculates CCI score for all patients based on indicators noted at any time in their EHR 
    use_only_confirmed_covid_positive = True

    if use_only_confirmed_covid_positive:
        df = Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ \
            .select('person_id', *[c for c in Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_.columns if '_indicator' in c])

        #calculating CCI components BEFORE and up to covid index
        MI = F.col('MYOCARDIALINFARCTION_before_or_day_of_covid_indicator') * 1
        CHF = F.col('CONGESTIVEHEARTFAILURE_before_or_day_of_covid_indicator') * 1
        PVD = F.col('PERIPHERALVASCULARDISEASE_before_or_day_of_covid_indicator') * 1
        CVD = F.col('CEREBROVASCULARDISEASE_before_or_day_of_covid_indicator') * 1
        DEM = F.col('DEMENTIA_before_or_day_of_covid_indicator') * 1
        CPD = F.col('CHRONICLUNGDISEASE_before_or_day_of_covid_indicator') * 1
        RD = F.col('RHEUMATOLOGICDISEASE_before_or_day_of_covid_indicator') * 1
        PEP = F.col('PEPTICULCER_before_or_day_of_covid_indicator') * 1
        LIV = F.when(F.col('MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator')==1, F.col('MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator') * 3) \
            .when(F.col('MODERATESEVERELIVERDISEASE_before_or_day_of_covid_indicator')==0, F.col('MILDLIVERDISEASE_before_or_day_of_covid_indicator') * 1) \
            .otherwise(0)
        DIA = F.when(F.col('DIABETESCOMPLICATED_before_or_day_of_covid_indicator')==1, F.col('DIABETESCOMPLICATED_before_or_day_of_covid_indicator') * 2) \
            .when(F.col('DIABETESCOMPLICATED_before_or_day_of_covid_indicator')==0, F.col('DIABETESUNCOMPLICATED_before_or_day_of_covid_indicator') * 1) \
            .otherwise(0)
        HEM = F.col('HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator') * 2
        REN = F.col('KIDNEYDISEASE_before_or_day_of_covid_indicator') * 2
        #MALIGNANT CANCER concept set covers Leukemia and Lymphoma specified in the Charlson Comorbidity Index calculation
        CAN = F.when(F.col('METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator')==1, F.col('METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator') * 6) \
            .when(F.col('METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator')==0, F.col('MALIGNANTCANCER_before_or_day_of_covid_indicator') * 2) \
            .otherwise(0)
        HIV = F.col('HIVINFECTION_before_or_day_of_covid_indicator') * 6

        #calculate CCI score for patients, age not included as it is not explicitly included in the reference calculation used and is often already accounted for as a covariate in most studies
        df = df.withColumn('CCI_score_up_through_index_date', F.lit(MI + CHF + PVD + CVD + DEM + CPD + RD + PEP + LIV + DIA + HEM + REN + CAN + HIV).cast(IntegerType())) \
            .select('person_id', 'CCI_score_up_through_index_date')

    else:
        df = Logic_Liaison_All_patients_summary_facts_table_lds \
            .select('person_id', *[c for c in Logic_Liaison_All_patients_summary_facts_table_lds.columns if '_indicator' in c])

        #calculating CCI components all-time
        MI = F.col('MYOCARDIALINFARCTION_indicator') * 1
        CHF = F.col('CONGESTIVEHEARTFAILURE_indicator') * 1
        PVD = F.col('PERIPHERALVASCULARDISEASE_indicator') * 1
        CVD = F.col('CEREBROVASCULARDISEASE_indicator') * 1
        DEM = F.col('DEMENTIA_indicator') * 1
        CPD = F.col('CHRONICLUNGDISEASE_indicator') * 1
        RD = F.col('RHEUMATOLOGICDISEASE_indicator') * 1
        PEP = F.col('PEPTICULCER_indicator') * 1
        LIV = F.when(F.col('MODERATESEVERELIVERDISEASE_indicator')==1, F.col('MODERATESEVERELIVERDISEASE_indicator') * 3) \
            .when(F.col('MODERATESEVERELIVERDISEASE_indicator')==0, F.col('MILDLIVERDISEASE_indicator') * 1) \
            .otherwise(0)
        DIA = F.when(F.col('DIABETESCOMPLICATED_indicator')==1, F.col('DIABETESCOMPLICATED_indicator') * 2) \
            .when(F.col('DIABETESCOMPLICATED_indicator')==0, F.col('DIABETESUNCOMPLICATED_indicator') * 1) \
            .otherwise(0)
        HEM = F.col('HEMIPLEGIAORPARAPLEGIA_indicator') * 2
        REN = F.col('KIDNEYDISEASE_indicator') * 2
        #MALIGNANT CANCER concept set covers Leukemia and Lymphoma specified in the Charlson Comorbidity Index calculation
        CAN = F.when(F.col('METASTATICSOLIDTUMORCANCERS_indicator')==1, F.col('METASTATICSOLIDTUMORCANCERS_indicator') * 6) \
            .when(F.col('METASTATICSOLIDTUMORCANCERS_indicator')==0, F.col('MALIGNANTCANCER_indicator') * 2) \
            .otherwise(0)
        HIV = F.col('HIVINFECTION_indicator') * 6

        #calculate CCI score for patients, age not included as it is not explicitly included in the reference calculation used and is often already accounted for as a covariate in most studies
        df = df.withColumn('CCI_score', F.lit(MI + CHF + PVD + CVD + DEM + CPD + RD + PEP + LIV + DIA + HEM + REN + CAN + HIV).cast(IntegerType())) \
            .select('person_id', 'CCI_score')

    return df

#################################################
## Global imports and functions included below ##
#################################################

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

@transform_pandas(
    Output(rid="ri.vector.main.execute.1012c1bb-ccfe-4fdd-9ca8-a982ab0168a9"),
    analysis_1_COVID_negative_control=Input(rid="ri.foundry.main.dataset.cabcd0ef-fb38-471c-a325-493a9ca7b458"),
    analysis_1_COVID_positive_control=Input(rid="ri.foundry.main.dataset.0ab2f17b-94f6-4f86-988b-e49c020e9d9f"),
    analysis_1_PASC_case=Input(rid="ri.foundry.main.dataset.42e7f154-baae-479c-aa65-f8ad830f7c68")
)
def test_no_intersection(analysis_1_COVID_positive_control, analysis_1_PASC_case, analysis_1_COVID_negative_control):
    df1 = analysis_1_COVID_positive_control.select('person_id','age_at_covid')
    df2 = analysis_1_PASC_case.select('person_id', 'first_COVID_ED_only_start_date')
    df3 = analysis_1_COVID_negative_control.select('person_id', 'state')

    result1 = df1.join(df2, 'person_id', 'inner')
    result2 = df1.join(df3, 'person_id', 'inner')
    result3 = df2.join(df3, 'person_id', 'inner')
    print(result1.count())
    print(result2.count())
    print(result3.count())
    

