# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 16:18:54 2017

@author: Sugu
"""
import os
import re
import xlrd
import pandas as pd
from sklearn import preprocessing
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.stats import mode

class KitschPreprocess:
    def majority_vote(self, predictions_list):
        predictions = [mode([i, j, k]).mode[0] for i, j, k in zip(predictions_list[0], predictions_list[1], predictions_list[2])]
        return predictions
    
    def evaluation_metrics(self, clasification_report_list):
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
        
    def print_metrics(self, clasification_report_list, accuracy_score_list):
        average_precision, average_recall, average_fscore = self.evaluation_metrics(clasification_report_list)
        overall_accuracy = float(sum(accuracy_score_list))/len(accuracy_score_list)
        print("Average Precision: ", average_precision)
        print("Average Recall: ", average_recall)
        print("Average Fscore: ", average_fscore)
        print("Overall Accuracy: ", overall_accuracy)
        return average_precision, average_recall, average_fscore, overall_accuracy
    
    def json_to_csv(self):
        df=pd.read_json("train.json")
        df.to_csv("results.csv", sep='\t', encoding='utf-8')
    
    def xls_to_txt(self, filename,output_file):
            x =  xlrd.open_workbook(filename)#, encoding_override = "utf-8")
            x1 = x.sheet_by_index(0)        
            
            f = open(output_file, 'wb')
            for rownum in xrange(0,x1.nrows):
                f.write(u','.join([re.sub(r'\s+', r' ', i) if isinstance(i, basestring) else str(float(i)) for i in x1.row_values(rownum, 0, 15)]).encode('utf-8').strip()+ '\n')
            f.close()
            
    def cleanup(self, data):
            cleantext = data.replace(",","")        #remove commas
            cleaner = re.compile('<.*?>')           #remove tags
            cleantext= re.sub(cleaner,'', cleantext)        
            ascii = set(string.printable) 
            cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
            cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
            cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
    #        cleantext = cleantext.translate(None, string.digits)
#            stemmer = PorterStemmer()
#            cleantextlist = [stemmer.stem(i) for i in cleantext.lower().split()]      #stem the word  
#            cleantext = ' '.join(cleantextlist)
            stop = set(stopwords.words('english')) - set(('and', 'or', 'not'))
            cleantextlist = [i for i in cleantext.lower().split() if i not in stop]      #remove stopwords except few exceptions  
            cleantext = ' '.join(cleantextlist)
            if len(cleantextlist) == 0:
                cleantext = "EMPTY"
            return cleantext
    
    def read_file(self, filename):
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
            features_list = []
            display_address_list = []
            street_address_list = []
            le_building = preprocessing.LabelEncoder()
            le_manager = preprocessing.LabelEncoder()
    #        le_created = preprocessing.LabelEncoder()
            for i in range(1,num_lines-1):
                all_cols = lines[i].split(",")
                building_id.append(all_cols[8])
                manager_id.append(all_cols[12])
                description = self.cleanup(all_cols[9])
                features = self.cleanup(all_cols[11])
                display_address = self.cleanup(all_cols[10])
                street_address = self.cleanup(all_cols[13])
                description_list.append(description)
                features_list.append(features)
                display_address_list.append(display_address)
                street_address_list.append(street_address)
            le_building.fit(building_id)
            le_manager.fit(manager_id)
    #        print len(le_manager.classes_)
       
            for i in range(1,num_lines-1):
                cols = lines[i].split(",")
    #            print(i)
                temp = list(map(float, cols[1:7]))
                temp.append(float(cols[7]))
                temp.append(float(le_building.transform(cols[8])))
                temp.append(float(le_manager.transform(cols[12])))
                data.append(temp)
                labels.append(re.sub(r'\s+', r' ', cols[-1]))
            f.close()
            return data, description_list, display_address_list, street_address_list, labels