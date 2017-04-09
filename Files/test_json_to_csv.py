# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 12:02:51 2017

@author: Sugu
"""

import pandas as pd

df=pd.read_json("train.json")

df.to_csv("results.csv", sep='\t', encoding='utf-8')


