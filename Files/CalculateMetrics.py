# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:03:48 2017

@author: Suganya
"""

import os
#from random import shuffle
#from preprocess import Preprocess

class CalculateMetrics:
    def calculate_metrics(self,classifier, test_set, class_label):
        """
        To compare labels from test set vs predicted labels
        """
        #precision, recall, F1score
        results = classifier.classify_many([fs for (fs, l) in test_set])
        tp = [l == class_label and r == class_label for ((fs, l), r) in zip(test_set, results)]
        fp = [l != class_label and r == class_label for ((fs, l), r) in zip(test_set, results)]
        #tn = [l != class_label and r != class_label for ((fs, l), r) in zip(test_set, results)]
        fn = [l == class_label and r != class_label for ((fs, l), r) in zip(test_set, results)]
        classified_correct = [l == r for ((fs, l), r) in zip(test_set, results)]
        precision, recall, f1score, overall_accuracy = self.calculate_values(classified_correct, tp, fp, fn)
        return precision, recall, f1score, overall_accuracy
        
    def calculate_values(self, classified_correct, tp, fp, fn):
        precision_denominator = (sum(tp) + sum(fp))
        recall_denominator = (sum(tp) + sum(fn))
        if(precision_denominator != 0.0):
            precision = float(sum(tp)) / precision_denominator
        else:
            #precision = 'NAN'
            precision = 0.0
        if(recall_denominator != 0.0):
            recall = float(sum(tp)) / (sum(tp) + sum(fn))
        else:
            #recall = 'NAN'
            recall = 0.0
        if(precision+recall != 0.0):
            f1score = float(2*precision*recall) / float(precision + recall)
        else:
#            f1score = 'NAN'
            f1score = 0.0
        if classified_correct:
            overall_accuracy = float(sum(classified_correct)) / len(classified_correct)
        else:
            overall_accuracy = 0.0
        return precision, recall, f1score, overall_accuracy
        
    """
    To compare predicted labels vs actual labels
    """
    def metrics(self, class_label, predicted_labels, actual_labels):
        assert(len(predicted_labels) == len(actual_labels))
        tp = [l == class_label and r == class_label for (l, r) in zip(actual_labels, predicted_labels)]
        fp = [l != class_label and r == class_label for (l, r) in zip(actual_labels, predicted_labels)]
        fn = [l == class_label and r != class_label for (l, r) in zip(actual_labels, predicted_labels)]
        n =  [l == class_label for l in actual_labels]
        class_accuracy = float(sum(tp)) / sum(n)
        overall_classified_correct = [l == r for (l,r) in zip(actual_labels, predicted_labels)]
        precision, recall, f1score, overall_accuracy = self.calculate_values(overall_classified_correct, tp, fp, fn)
        return precision, recall, f1score, class_accuracy, overall_accuracy
        
    def read_file(self, filename):
        rel_path = filename
        script_dir= os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(script_dir, rel_path)
        f = open ( abs_file_path )
        description_list = []
#        labels = []
#        descriptions=[]
#        preprocess = Preprocess()
   
        for line in f.readlines():
            cols = line.split("\t")
#            cols[0] = preprocess.cleanup(cols[0])      #write to a file new cleaned things 
            words_filtered=[]   #remove words less than 2 letters in length
            words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
#            descriptions.append((words_filtered,cols[1]))
            description_list.append(words_filtered)
#            labels.append(cols[1])
        f.close()
#        shuffle(descriptions)
        return  description_list
        
