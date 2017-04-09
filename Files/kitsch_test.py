# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 14:25:20 2017

@author: Sugu
"""

import os
import re
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import xlrd
import pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from CalculateMetrics import CalculateMetrics


def evaluation_metrics(clasification_report_list):
    precision_list = []
    recall_list = []
    fscore_list = []
    for clasification_report in clasification_report_list:
        lines = clasification_report.split('\n')
        average_metrics = lines[6].split()
#        print average_metrics
        precision_list.append(float(average_metrics[3]))
        recall_list.append(float(average_metrics[4]))
        fscore_list.append(float(average_metrics[5]))
    return float(sum(precision_list))/len(precision_list), float(sum(recall_list))/len(recall_list), float(sum(fscore_list))/len(fscore_list)
    
def print_metrics(clasification_report_list, accuracy_score_list):
    average_precision, average_recall, average_fscore = evaluation_metrics(clasification_report_list)
    overall_accuracy = float(sum(accuracy_score_list))/len(accuracy_score_list)
    print("Average Precision: ", average_precision)
    print("Average Recall: ", average_recall)
    print("Average Fscore: ", average_fscore)
    print("Overall Accuracy: ", overall_accuracy)
    return average_precision, average_recall, average_fscore, overall_accuracy

def json_to_csv():
    df=pd.read_json("train.json")
    df.to_csv("results.csv", sep='\t', encoding='utf-8')

def xls_to_txt(filename,output_file):
        x =  xlrd.open_workbook(filename)#, encoding_override = "utf-8")
        x1 = x.sheet_by_index(0)        
        
        f = open(output_file, 'wb')
        for rownum in xrange(0,x1.nrows):
            f.write(u','.join([re.sub(r'\s+', r' ', i) if isinstance(i, basestring) else str(float(i)) for i in x1.row_values(rownum, 0, 15)]).encode('utf-8').strip()+ '\n')
        f.close()
        
def cleanup(data):
        cleantext = data.replace(",","")        #remove commas
        cleaner = re.compile('<.*?>')           #remove tags
        cleantext= re.sub(cleaner,'', cleantext)        
        ascii = set(string.printable) 
        cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
        cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
        cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
        cleantext = cleantext.translate(None, string.digits)
        stemmer = PorterStemmer()
        cleantextlist = [stemmer.stem(i) for i in cleantext.lower().split()]      #stem the word  
        cleantext = ' '.join(cleantextlist)
        stop = set(stopwords.words('english')) - set(('and', 'or', 'not'))
        cleantextlist = [i for i in cleantext.lower().split() if i not in stop]      #remove stopwords except few exceptions  
        cleantext = ' '.join(cleantextlist)
        if len(cleantextlist) == 0:
            cleantext = "EMPTY"
        return cleantext

def read_file(filename):
        rel_path = filename
        script_dir= os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(script_dir, rel_path)
        f = open(abs_file_path, "r")
        lines = f.read().split("\n")
        num_lines = len(lines)
#        print num_lines
        data = []
        labels = []
        building_id = []
        manager_id = []
        description_list = []
        le_building = preprocessing.LabelEncoder()
        le_manager = preprocessing.LabelEncoder()
        description_file = open('description_cleaned.txt', 'wb')
        for i in range(1,num_lines-1):
            all_cols = lines[i].split(",")
            building_id.append(all_cols[8])
            manager_id.append(all_cols[12])
            description = cleanup(all_cols[9])
            description_list.append(description)
            description_file.write(description)
            description_file.write("\n")
        description_file.close()
        le_building.fit(building_id)
        le_manager.fit(manager_id)
#        print len(le_manager.classes_)
   
        for i in range(1,num_lines-1):
            cols = lines[i].split(",")
#            print(i)
            temp = list(map(float, cols[1:7]))
            temp.append(float(le_building.transform(cols[8])))
            temp.append(float(le_manager.transform(cols[12])))
            data.append(temp)
            labels.append(re.sub(r'\s+', r' ', cols[-1]))
        f.close()
        return data, description_list, labels
        
#xls_to_txt("training.xlsx", "training_csv.txt")
data, description_list, labels = read_file("training_csv_small.csv")
print('File read complete')

#calculate_metrics = CalculateMetrics()
#description_list = calculate_metrics.read_file('description_cleaned.txt')

number_of_folds = 2
subset_size = len(data)/number_of_folds
linear_svm_report = []
linear_svm_accuracy_list = []
dt_report = []
dt_accuracy_list = []
knn_report = []
knn_accuracy_list = []
ab_report = []
ab_accuracy_list = []
dt_description_report = []
dt_description_accuracy_list = []

for i in range(number_of_folds):
    test_data = data[i*subset_size:][:subset_size]
    train_data = data[:i*subset_size] + data[(i+1)*subset_size:]
    test_data_labels = labels[i*subset_size:][:subset_size]
    train_data_labels = labels[:i*subset_size] + labels[(i+1)*subset_size:]

    test_description = description_list[i*subset_size:][:subset_size]
    train_description = description_list[:i*subset_size] + description_list[(i+1)*subset_size:]

    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_description)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_description)
    
    dt_description_classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, train_data_labels)
    predictions = dt_description_classifier.predict(test_tfidf_matrix)
    dt_description_report.append(classification_report(test_data_labels, predictions))
    dt_description_accuracy_list.append(accuracy_score(test_data_labels, predictions))
    
    dt_classifier = tree.DecisionTreeClassifier().fit(train_data, train_data_labels)
    predictions = dt_classifier.predict(test_data)
    dt_report.append(classification_report(test_data_labels, predictions))
    dt_accuracy_list.append(accuracy_score(test_data_labels, predictions))
#    
#    knn_classifier = KNeighborsClassifier(n_neighbors=3).fit(train_data, train_data_labels)
#    predictions = knn_classifier.predict(test_data)
#    knn_report.append(classification_report(test_data_labels, predictions))
#    knn_accuracy_list.append(accuracy_score(test_data_labels, predictions))
#    
#    ab_classifier = AdaBoostClassifier().fit(train_data, train_data_labels)
#    predictions = ab_classifier.predict(test_data)
#    ab_report.append(classification_report(test_data_labels, predictions))
#    ab_accuracy_list.append(accuracy_score(test_data_labels, predictions))
#    
#    svm_classifier = svm.LinearSVC().fit(train_data, train_data_labels)
#    predictions = svm_classifier.predict(test_data)
#    linear_svm_report.append(classification_report(test_data_labels, predictions))
#    linear_svm_accuracy_list.append(accuracy_score(test_data_labels, predictions))
#    
print 'Decision tree'
print_metrics(dt_report, dt_accuracy_list)
#print_metrics(linear_svm_report, linear_svm_accuracy_list)
#print_metrics(knn_report, knn_accuracy_list)
#print_metrics(ab_report, ab_accuracy_list)
#
print 'Decision tree description'
print_metrics(dt_description_report, dt_description_accuracy_list)
