#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:33:20 2022

@author: alexandre
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from aix360.algorithms.rbm import FeatureBinarizer, BooleanRuleCG,FeatureBinarizerFromTrees
from aix360.algorithms.rbm import LogisticRuleRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import re





class DMNDecisionActivity:

    def __init__(self, model, activity, attribute, shift_no, traces):
        self.model = model
        self.activity = activity
        self.attribute = attribute
        self.shift_no = shift_no
        self.traces = traces

    def __str__(self):
        return str(self.activity) + '---->' + str(self.attribute) + '(' + str(self.shift_no) + ')'


class PossibleModel:

    MAX_DEPTH = 5
    # file = open('rresults__dt_' + str(MAX_DEPTH) + '.csv', 'w')
    # file.write('model,type,no_lines,no_labels,no_leaves,no_nodes,acc_dt,f1_dt,acc_rf,f1_rf,acc_lrr,'
    #            'f1_lrr,no_rules_lrr,terms_lrr,acc_br,f1_br,no_rules,terms_br\n')

    def __init__(self, top_dap, daaps, object_nos, keys_y):
        print('New model with TOP DAP:', top_dap)
        print('DAAPS:')
        self.daaps = set()
        for daap in daaps:
            print(daap)
            self.daaps.add(daap)
        print('Object Traces:', len(object_nos), object_nos)
        
        self.top_dap = top_dap
        self.object_nos = object_nos

        self.rn = np.random.randint(0, 100000000)
        
        self.keys_y = keys_y
        
        
    def add_daaotp(self, daap,keys):
       
        self.keys_y.append(keys)
        
        
        self.daaps.add(daap)
        
    def build_model2(self,just_correlation):
        
        self.create_training_set2()
        rf = gv = None
        explan = []
        info_string = ''
        #print('self.y', self.y)
        if not is_numeric_dtype(self.y.iloc[:, 0]):
            if len(self.y.iloc[:, 0].unique()) > len(self.y) / 2:
                #print('this is Y',self.y.iloc[:, 0].unique(), self.y )
                print('Bad y-ratio:', len(self.y.iloc[:,0].unique()), 'over', len(self.y))
                return 0, None
                
        
        if just_correlation==True:
            if len(self.X.columns) > 0:
                print('Good y-ratio:', len(self.y.iloc[:, 0].unique()), 'over', len(self.y))
                self.X = pd.get_dummies(self.X, drop_first=True)
                if is_numeric_dtype(self.y.iloc[:, 0]):
                    accuracy = mutual_info_regression(self.X, self.y.iloc[:, 0],random_state=42)
                    
                else:
                    accuracy = mutual_info_classif(self.X, self.y.iloc[:, 0],random_state=42)
                    
                accuracy = max(accuracy)
            else:
                print('Not enough variables')
                return 0, None
        elif just_correlation ==False:
            accuracy = 0
            if len(self.X.columns) > 0:
                print('X:', self.X.columns)
                print('y:', self.y.columns)
                
       
                
                if len(self.y.iloc[:, 0].unique()) <  2:
                    print(self.y.iloc[:, 0].unique())
                    #breakpoint()
                    
                    return accuracy, (gv, explan)
                if is_numeric_dtype(self.y.iloc[:, 0]) and self.y.iloc[:, 0].dtype != bool:
                    self.y.iloc[:, 0], bins = pd.qcut(self.y.iloc[:, 0], q=2, labels=['lower', 'higher'], retbins=True)
                    replace_dict = {'lower': 'lower_'+str(bins[1]), 'higher': 'higher_'+str(bins[1])}
                    self.y.iloc[:, 0] = self.y.iloc[:, 0].replace(replace_dict)
                    
                
                
                all_rules = []
           
                if 1 > 0:
                    """Here we make the categorical X columns and label them into dummies"""
                    colCat = [x for x in self.X.columns if x in self.X.select_dtypes(['object'])]
                    
                    le = LabelEncoder()
                    for col in colCat:
                        
                        self.X[col] = le.fit_transform(self.X[col])
                    #print(len(colCat))
                    #print(len(self.X.columns))
                    if len(colCat) == len(self.X.columns) :
                    
                    
                        print('number of rows', len(self.X))
                        new_row = random.sample(range(0, 1000000000000000000), len(self.X))
                        self.X = self.X.assign(new=new_row)
                    #print('here',self.X)
                    #breakpoint()
                     
                    y_labels = self.y.iloc[:, 0].unique()
                    
                    
                    
                    """Here we labelize and encode the y_labels"""
                    if len(y_labels) <= 2:
                        le = LabelEncoder()
                        y_br = le.fit_transform(self.y.iloc[:, 0])
                    
                    else:
                        le = LabelBinarizer()
                        y_br = le.fit_transform(self.y.iloc[:, 0])
                        y_outcomes = np.zeros((len(y_br), len(y_labels)))
                        y_outcomes_br = np.zeros((len(y_br), len(y_labels)))
                    
                    accuracy_br = accuracy_lrr = auc_br = f1_br = f1_lrr = auc_lrr = 0
                    no_rules = no_rules_lrr = 0
                    terms_br = terms_lrr = 0
                    """
                    fb = FeatureBinarizer(negations=False,returnOrd=True)   
                    x_br, x_br_std = fb.fit_transform(self.X)
                    print(y_labels)
                    
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', -1)
                    
                    #print('hello',self.X)
                   
                    #breakpoint()
                    #print('What is X problem', x_br.columns)
                    #print(self.X.axes)
                    
                    #print('What is the Y problem?', y_br)
                    
                    
                    if len(y_labels) <= 2:
                        #Here we built all the rules of the binary predictive model (was with "")
                        br = BooleanRuleCG(lambda0=0.00005, lambda1=0.00001)
                        br.fit(x_br, y_br)
                        
                        accuracy_br = accuracy_score(y_br, br.predict(x_br))
                        f1_br = f1_score(y_br, br.predict(x_br))
                        all_rules.extend(br.explain()['rules'])
                        
                        "#Here we built all the rules of the logistic predictive model (was with ")
                        #Original
                        #lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
                        lrr = LogisticRuleRegression(lambda0=0.00005, lambda1=0.00001, useOrd=True)
                        
                        lrr.fit(x_br, y_br,x_br_std)
                        dfx = lrr.explain()
                        accuracy_lrr = accuracy_score(y_br, lrr.predict(x_br, x_br_std))
                        f1_lrr = f1_score(y_br, lrr.predict(x_br, x_br_std))
                        
                        print('Labels:', self.y[self.top_dap.small_string()].unique())
                       
                        print(self.y.iloc[:, 0].value_counts())
                        #breakpoint()
                        print('Accuracy BR:', accuracy_br)
                        print('Accuracy LRR:', accuracy_lrr)
                        print('BR rules:\n', br.explain()['rules'])
                        
                        for rule in br.explain()['rules']:
                            sep_terms = re.split('AND|OR', rule)
                            terms_br += len(sep_terms)
                            
                        if 'rule/numerical feature' in lrr.explain().columns:
                            for rule in lrr.explain()['rule/numerical feature']:
                                sep_terms = re.split('AND|OR', rule)
                                terms_lrr += len(sep_terms)
                        
                        print('Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:')
                        print(br.explain()['rules'])
                        explan.append('Label proportions:\n')
                        explan.append(str(self.y.iloc[:, 0].value_counts()))
                        explan.append('\n\nBoolean Rule Column Generation:\n')
                        for rule in br.explain()['rules']:
                           explan.append(rule + '\n')
                        explan.append('\n\nLogistic Rule Regression:\n')
                        no_rules = len(br.explain()['rules'])
                        explan.extend(str(lrr.explain()))
                        no_rules_lrr = len(lrr.explain())
                        
                    #In your current artificial event logs you will not go through this normally    
                    else:
                        explan = []
                        sum_acc = sum_f1 = 0
                        sum_acc_lrr = sum_f1_lrr = sum_auc_lrr = 0
                        for i, label in enumerate(sorted(y_labels)):
                            print('Label ', i, ':', label)
                            
                            y_to_use = y_br[:, i]
                            br = BooleanRuleCG()  # lambda0=1e-6, lambda1=1e-6
                            br.fit(x_br, y_to_use)
                            accuracy_br = accuracy_score(y_to_use, br.predict(x_br))
                            f1_br = f1_score(y_to_use, br.predict(x_br))
                            y_outcomes_br[:, i] = br.predict(x_br)
                            for rule in br.explain()['rules']:
                                all_rules.append('Label: ' + label + ' - ' + rule)
                                
                            lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
                            lrr.fit(x_br, y_to_use, x_br_std)
                            accuracy_lrr = accuracy_score(y_to_use, lrr.predict(x_br, x_br_std))
                            y_outcomes[:, i] = lrr.predict_proba(x_br, x_br_std)
                            f1_lrr = f1_score(y_to_use, lrr.predict(x_br, x_br_std))
                            
                            explan.extend('\n\n##########################################\n')
                            explan.append('Label: ' + label + ' (' + str(np.sum(y_to_use)) + ')\n')
                            explan.extend('##########################################\n')
                            explan.append('\nBoolean Rule Column Generation:\n')
                            for rule in br.explain()['rules']:
                                explan.append(rule + '\n')
                            
                            explan.extend('\n\n-------------------------------------------------')
                            explan.append('\n\nLogistic Rule Regression:\n')
                            explan.extend(str(lrr.explain()))
                            print('Starting br:')
                            print(br.explain())
                            
                            for rule in br.explain()['rules']:
                               #print('This works')
                               sep_terms = re.split('AND|OR', rule)
                               terms_br += len(sep_terms)
                            print(lrr.explain())
                            if 'rule/numerical feature' in lrr.explain().columns:
                                for rule in lrr.explain()['rule/numerical feature']:
                                    #print('This works (2)')
                                    sep_terms = re.split('AND|OR', rule)
                                    terms_lrr += len(sep_terms)
                            
                            print('Accuracy BR:', accuracy_br)
                            print('Accuracy LRR:', accuracy_lrr)
                            print('BR rules:\n', br.explain()['rules'])
                            print('LRR rules\n:', lrr.explain())
                            
                            sum_acc += accuracy_br
                            sum_acc_lrr += accuracy_lrr
                            sum_f1 += f1_br
                            no_rules += len(br.explain()['rules'])
                            no_rules_lrr += len(lrr.explain())
                        
                        accuracy_br = sum_acc / len(y_labels)
                        accuracy_lrr = sum_acc_lrr / len(y_labels)
                        f1_lrr = f1_score(self.y.iloc[:, 0], le.inverse_transform(y_outcomes), average='micro')
                        f1_br = f1_score(self.y.iloc[:, 0], le.inverse_transform(y_outcomes_br), average='micro')
                        for i, label in enumerate(sorted(y_labels)):
                            print('Label ', i, ':', label)
                        #print(y_labels)
                        #print(self.y.columns)
                        #breakpoint()
                        """
                    """Now we make a random forest classifier"""
                    self.X = pd.get_dummies(self.X, drop_first=True)
                    print(y_labels)
                    rf = RandomForestClassifier()
                    rf.fit(self.X, self.y.iloc[:, 0])
                    accuracy = accuracy_score(self.y.iloc[:, 0], rf.predict(self.X))
                    
                    if len(y_labels) > 2:
                        f1_rf = f1_score(self.y.iloc[:, 0], rf.predict(self.X), average='micro')
                    else:
                        print('other path',y_labels)
                        """If the labels are boolean transform them to boolean"""
                        if self.y.iloc[:, 0].dtype == bool:
                            y_labels = [True, False]
                        f1_rf = f1_score(self.y.iloc[:, 0], rf.predict(self.X), average='binary', pos_label=y_labels[0])
                        
                    rf_2 = DecisionTreeClassifier(max_depth=3)
                    rf_2.fit(self.X, self.y.iloc[:, 0])
                    
                    print('##### NODES (3):', rf_2.tree_.node_count)
                    print('##### MAX DEPTH (3):', rf_2.tree_.max_depth)
                    accuracy_2 = accuracy_score(self.y.iloc[:, 0], rf_2.predict(self.X))
                    
                    if len(y_labels) > 2:
                        f1_rf2 = f1_score(self.y.iloc[:, 0], rf_2.predict(self.X), average='micro')
                    else:
                        f1_rf2 = f1_score(self.y.iloc[:, 0], rf_2.predict(self.X), average='binary', pos_label=y_labels[0])
                    print('Accuracy: (2)', accuracy_2, 'difference', (accuracy-accuracy_2))
                    info_string = f'{str(self)},classification,{len(self.X)},{len(y_labels)},{rf_2.tree_.n_leaves}' \
                                  f',{rf_2.tree_.node_count},{accuracy_2},{f1_rf2},{accuracy},{f1_rf},{accuracy_lrr},' \
                                  f'{f1_lrr},{no_rules_lrr},{terms_lrr},{accuracy_br},{f1_br},{no_rules},{terms_br}\n'
                    
                    rf = rf_2
                    
                gv = (rf, accuracy_2, accuracy_br, self.X.columns,sorted(y_labels), info_string, all_rules)
                #gv =(rf,accuracy_2,self.X.columns,sorted(y_labels), info_string, all_rules)
                print('\n****Built a RF!****\n')
                print('Accuracy:', accuracy)
            
            else:
                print('Not enough variables')
                return 0, None
        
        return accuracy, (gv, explan)
               
                            
                    
                    
                   
                

        
                
                
               
                
               
                

    
        
      
    
    
 
    
    def create_training_set2(self):
        self.data = {}
        self.expand_dataset2(self.top_dap, list(self.object_nos))
        print('Just finished the X dataset \n  ')
        
        for daap in self.daaps:
            self.expand_dataset2(daap, self.keys_y)
            print('Just finished the Y dataset\n ')
        
        
        df = pd.DataFrame.from_dict(self.data)
    
        
        print(df.to_markdown())
        
        
        self.y = df[[self.top_dap.small_string()]]
        self.X = df.drop([self.top_dap.small_string()], axis=1)



        to_remove = []
        for var in self.X.columns:
            #print('This',var)
            if is_numeric_dtype(self.X[var]):
                continue
            print(len(self.X[var].unique()))
            if len(self.X) / 2 < len(self.X[var].unique()) or len(self.X[var].unique()) == 1:
                to_remove.append(var)
        self.X = self.X.drop(to_remove, axis=1)
        
        #return print('Create training set')
        
    def expand_dataset2(self, daap, keys):
             print('Expanding', daap)
             print(daap.values_per_object_trace)
             print(self.object_nos)
                 
             subset = []
             if type(keys[-1]) == list:
                 maxobjectsintrace= len(keys[-1])
                 addedobjects = 0
                 
                     
                 for object_no in keys:
                   if  maxobjectsintrace > addedobjects:
                             #print('maxobjectsintrace', maxobjectsintrace)
                             if type(object_no) != list:
                                 try: 
                                     subset.append(daap.values_per_object_trace[object_no][0])
                                     addedobjects = addedobjects+1
                                    
                                 except:
                                     #print("An exception occurred")
                                     #print('total',addedobjects)
                                     continue
                             
                             else:
                                 for object_no2 in object_no:
                                     try: 
                                         subset.append(daap.values_per_object_trace[object_no2][0])
                                         addedobjects = addedobjects+1
                                         
                                     except:
                                     
                                         continue
                   else:
                        continue
             else:
                 for object_no in keys:
                     if type(object_no) != list:
                         try: 
                             subset.append(daap.values_per_object_trace[object_no][0])
                             
                         except:
                             #print("An exception occurred")
                             continue
                
                         
                     
                     
             
             self.data[daap.small_string()] = subset  
        
            
    def contains(self, other):
        if self.top_dap.attribute != other.top_dap.attribute:
            return False
        if self.object_nos != other.object_nos:
            return False
        for daap in other.daaps:
            if daap not in self.daaps:
               return False
        return True  
    
    def __str__(self):
        return str(self.top_dap) + ' ' + str(len(self.daaps)) + ' ' + str(len(self.object_nos)) + ' ' + str(self.rn)
    


class DMNModel:

    class Edge:

        def __init__(self, source, target, weight=0, correlation=0):
            self.source = source
            self.target = target
            self.weight = weight
            self.correlation = correlation

    def __init__(self, label, object_nos, daotp):
        self.label = label
        self.object_nos = object_nos
        self.daotp = daotp
        self.trained_models = set()
        self.__edges = set()
        self.top_activity = None
        self.aaps = {}
        self.nodes = set()
        self.attributes = set()
        self.found_models = set()
        self.sub_models = set()
        self.super_model = None
        self.pairs = {}
        
    def are_connected(self, top_dap, d_activity, dmn_attribute, shift_no):
        print('Connected?', top_dap, d_activity.name,"for",dmn_attribute,'at', shift_no)
        for edge in self.__edges:
            if edge.source.activity == d_activity and edge.source.attribute == dmn_attribute \
                and edge.source.no_shift == shift_no and edge.target.activity == top_dap.activity \
                    and edge.target.attribute == top_dap.attribute and edge.target.no_shift == top_dap.no_shift:
                return True
            if edge.target.activity == d_activity and edge.target.attribute == dmn_attribute \
                and edge.target.no_shift == shift_no and edge.source.activity == top_dap.activity \
                    and edge.source.attribute == top_dap.attribute and edge.source.no_shift == top_dap.no_shift:
                return True
        return False
    
    def add_node(self, daap):
      for node in self.nodes:
          if node.activity == daap.activity and node.attribute == daap.attribute and node.shift_no == daap.no_shift:
              return node

      decision_activity = DMNDecisionActivity(self, daap.activity, daap.attribute, daap.no_shift, daap.values_per_object_trace.keys())
      self.nodes.add(decision_activity)
      self.pairs[daap] = decision_activity
    
    def add_edge(self, n1, n2, no_objects, corr):
        self.add_node(n1)
        self.add_node(n2)
        edge = DMNModel.Edge(n1, n2, no_objects, corr)
        self.__edges.add(edge) 
        
    def get_edges(self):
        return self.__edges

    def get_no_edges(self):
        return len(self.__edges)
    
    def clone(self, label, object_nos):
     new_model = DMNModel(label, object_nos, self.top_activity)

     for daap, activity in self.aaps.items():
         new_model.nodes.add(daap)

     for attribute in self.attributes:
         new_model.attributes.add(attribute)

     for edge in self.__edges:
         new_model.add_edge(edge.source, edge.target, edge.weight, edge.correlation)

     for activity, dmn_attr in self.found_models:
         new_model.found_models.add((activity, dmn_attr))

     return new_model
    
    
    
    
    
    