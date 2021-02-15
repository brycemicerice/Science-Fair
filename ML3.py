import re
import os
import csv
import numpy as np
import arff
import pandas as pd
import csv2arff
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import PredictionOutput

from scipy.stats import pearsonr, spearmanr

jvm.start()

project_data = pd.read_csv('C:\Users\qwert\Downloads\dataset.csv')
first = project_data.columns[16:1213]
gma_verbal_project_data = pd.read_csv(r'C:\Users\qwert\Downloads\OUTPUT.csv')
project_data = project_data.merge(gma_verbal_project_data, how='inner',on='ID')
second = project_data.columns[16:1213]
if first.all() == second.all():
    print("same data")
else:
    print("different data")
feature_data = project_data.iloc[:,16:1213]
feature_data = feature_data.drop(columns=['isMale_evidence_mean','isMale_evidence_stddev','isMale_evidence_var','isMale_evidence_median','isMale_evidence_min','isMale_evidence_max','isMale_evidence_range'])

gender = gma_verbal_project_data['gender']
male_mask = gender == 1
female_mask = gender == 2


labels = ['obsExtra_x', 'obsAgree_x','obsConsc_x', 'obsES_x','obsOpen_x','obsIntel_x','obsHire_x','standard_score_gma','gma_wo_cube_questions','gma_w_alternatives','verbal_partial_credit_score','verbal_nonpartial_credit_score']

for label in labels:
    df = pd.concat((feature_data, project_data[label]), axis=1)
    df.to_csv('df.csv', index=False)
    os.system('csv2arff df.csv df.arff')

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("df.arff")
    data.class_is_last()

    best_spearman_corr = -np.inf
    best_leaf_instance_param = None
    for min_leaf_instances in [150, 250, 350, 500]:  
        cls = Classifier(classname="weka.classifiers.trees.M5P")
        cls.options = ["-M", str(min_leaf_instances)]
        pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
        evl = Evaluation(data)
        evl.crossvalidate_model(cls, data, 5, Random(1), pout)

        with open("prediction.csv", "wb") as outfile:
            buffer_content = pout.buffer_content().strip(' ')
            csv_content = re.sub(' *\n *', '\n', buffer_content)
            csv_content = re.sub(' +', ',', csv_content)
            outfile.write(csv_content)

        pred_df = pd.read_csv('prediction.csv')

        spearman_corr = spearmanr(pred_df['actual'], pred_df['predicted'])[0]

        if spearman_corr > best_spearman_corr:
            best_spearman_corr = spearman_corr
            best_leaf_instance_param = min_leaf_instances

    cls.options = ["-M", str(best_leaf_instance_param)]
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 5, Random(1), pout)

    prediction_file_name = label+"_M5P_prediction__M"+str(best_leaf_instance_param)+".csv"

    with open(prediction_file_name, "wb") as outfile:
            buffer_content = pout.buffer_content().strip(' ')
            csv_content = re.sub(' *\n *', '\n', buffer_content)
            csv_content = re.sub(' +', ',', csv_content)
            outfile.write(csv_content)

    final_df = pd.read_csv(prediction_file_name)
    pearson_corr = pearsonr(final_df['actual'], final_df['predicted'])[0]
    spearman_corr = spearmanr(final_df['actual'], final_df['predicted'])[0]
    male_spearman_corr = spearmanr(final_df['actual'][male_mask], final_df['predicted'][male_mask])[0]
    female_spearman_corr = spearmanr(final_df['actual'][female_mask], final_df['predicted'][female_mask])[0]
    subtracted = abs(male_spearman_corr - female_spearman_corr)
    print("\nData: " +label)
    print("Best Leaf Instance Parameter: " + str(best_leaf_instance_param))
    print("Pearson Correlation: "+str(pearson_corr))
    print("Spearman Correlation: "+str(spearman_corr))
    print("Male Spearman Correlation: "+str(female_spearman_corr))
    print("Female Spearman Correlation: "+str(male_spearman_corr))
    print("Different between M & F Correlations: "+str(subtracted))

jvm.stop()