# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 14:25:20 2017

@author: Sugu
"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from KitschPreprocess import KitschPreprocess


kitsch_preprocess = KitschPreprocess()     
#xls_to_txt("training.xlsx", "training_csv.txt")
data, description_list, display_address_list, street_address_list, labels = kitsch_preprocess.read_file("training_csv.csv")
print('File read complete')

#calculate_metrics = CalculateMetrics()
#description_list = calculate_metrics.read_file('description_cleaned.txt')

number_of_folds = 10
subset_size = len(data)/number_of_folds
dt_report = []
dt_accuracy_list = []
dt_description_report = []
dt_description_accuracy_list = []
dt_display_address_report = []
dt_display_address_accuracy_list = []
dt_street_address_report = []
dt_street_address_accuracy_list = []

predictions_list = []

for i in range(number_of_folds):
    test_data = data[i*subset_size:][:subset_size]
    train_data = data[:i*subset_size] + data[(i+1)*subset_size:]
    """
    numerical data split
    """
    test_data_labels = labels[i*subset_size:][:subset_size]
    train_data_labels = labels[:i*subset_size] + labels[(i+1)*subset_size:]
    
    """
    street address split
    """
    test_street_address = street_address_list[i*subset_size:][:subset_size]
    train_street_address = street_address_list[:i*subset_size] + street_address_list[(i+1)*subset_size:]

    """
    description data split
    """
    test_description = description_list[i*subset_size:][:subset_size]
    train_description = description_list[:i*subset_size] + description_list[(i+1)*subset_size:]

    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_description)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_description)
    
    print("Fold " + str(i) + " description classifier starts")
    dt_description_classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, train_data_labels)
    predictions = dt_description_classifier.predict(test_tfidf_matrix)
#    dt_description_report.append(classification_report(test_data_labels, predictions))
#    dt_description_accuracy_list.append(accuracy_score(test_data_labels, predictions))
    predictions_list.append(predictions)
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_street_address)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_street_address)
    
    print("Fold " + str(i) + " street address classifier starts")
    dt_street_address_classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, train_data_labels)
    predictions = dt_street_address_classifier.predict(test_tfidf_matrix)
#    dt_street_address_report.append(classification_report(test_data_labels, predictions))
#    dt_street_address_accuracy_list.append(accuracy_score(test_data_labels, predictions))
    predictions_list.append(predictions)
    
    print("Fold " + str(i) + " numerical classifier starts")
    dt_classifier = tree.DecisionTreeClassifier().fit(train_data, train_data_labels)
    predictions = dt_classifier.predict(test_data)
#    dt_report.append(classification_report(test_data_labels, predictions))
#    dt_accuracy_list.append(accuracy_score(test_data_labels, predictions))
    predictions_list.append(predictions)
    
    print("Fold " + str(i) + " classification report starts")
    predictions = kitsch_preprocess.majority_vote(predictions_list)
    dt_report.append(classification_report(test_data_labels, predictions))
    dt_accuracy_list.append(accuracy_score(test_data_labels, predictions))

print 'Decision tree'
kitsch_preprocess.print_metrics(dt_report, dt_accuracy_list)
#print 'Decision tree description'
#print_metrics(dt_description_report, dt_description_accuracy_list)
#print 'Decision tree street address'
#print_metrics(dt_street_address_report, dt_street_address_accuracy_list)
