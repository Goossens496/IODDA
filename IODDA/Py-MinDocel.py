
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:38:25 2022

@author: alexandre
"""
"""import relevant packages"""

#%%

import pandas as pd
from DOCEL_classes import Activity, Shift, AttributeShiftSequence, DMNActivityAttributeObjectTypePair
from DMNModel import DMNModel, PossibleModel
import ast
import re
from graphviz import Digraph
from sklearn import tree
import os
from subprocess import check_call


#%%

""" Function to read a DOCEL log  """ 

def read_docel_log(path):
    
    """ Start by loading in the excel file and all its sheets. """ 
    
    df_sheet_map = pd.read_excel(path, sheet_name=None,index_col=1)
    attributes = set()
    values = dict()

    for sheet in df_sheet_map:
        """  We also drop the 'Unnamed: 0' column """ 
        
        df_sheet_map[sheet]=df_sheet_map[sheet].drop("Unnamed: 0",axis=1)
        """ We create a list of attributes as well """ 
        for col in df_sheet_map[sheet].columns:
            attributes.add(col)
            values[col] = df_sheet_map[sheet][col].unique()
                
        remove = ['Publishers','Authors','Books','Activity','EID','OID','Timestamp']
        attributes = attributes - set(remove)
        for k in remove:
            values.pop(k, None)  
        
        """ We replace all the NaN with / """ 
        
        df_sheet_map[sheet]=df_sheet_map[sheet].fillna('/')
        
        print(sheet + ' has been stored in the dataframe: df_sheet_map[' + '\'' + sheet + '\''
              + ']:')
        
        
    return df_sheet_map, attributes, values

#%%


""" Function to discover_shifts """ 

def discover_shifts(df_sheet_map, activities, attribute_values_dict, object_dict):
    
    for ot in object_types_list:
        object_dict[ot] = []
        
        
        """  For every object (which in this case is an index) """ 
        for odx in df_sheet_map[ot].index:
           
           print(df_sheet_map['Events'].loc[df_sheet_map['Events'][ot].str.contains(fr'\b{odx}\b', regex=True, case=False)].index)
           
           """da stands for dynamic attribute"""
           da_set = set()
           """ We create a list of indices with all the events (endices) where the object is present """ 
           endices = df_sheet_map['Events'].loc[df_sheet_map['Events'][ot].str.contains(fr'\b{odx}\b', regex=True, case=False)].index
           if len(endices) > 0:
               object_dict[ot].append(odx)
           for edx in  endices:
               """ for every event we need to extract the name of the event to create the shifts """ 
               event_name = df_sheet_map['Events'].loc[edx]['Activity']
               """  We assume static attributes are created by the first event """ 
               if edx == endices[0]:
                   """  Here we need extract to for sure all the static attributes """ 
                   """  We loop over all static attributes sa """ 
                   for sa in df_sheet_map[ot].columns:
                       """  We create a new shift with event, static attribute, object ID and object Type """ 
                       new_shift = Shift(edx,event_name, sa, 0, odx, ot)
                       new_shift.attribute_value = df_sheet_map[new_shift.object_type].loc[new_shift.object_no][new_shift.attribute]
                    
                       
                       activities[event_name].add_shift(odx,new_shift)
                       
                       activities[event_name].events[sa].append(new_shift.attribute_value)
                       print(new_shift.__str__())
                    
                       
               for sa in df_sheet_map[ot].columns:
                 activities[event_name].values[sa] = attribute_values_dict[sa]
                
                
                       
                       
                           
                   
               """  Check the dynamic attributes """ 
               for da in dynamic_attributes_dict[ot]:
                   print('problem', ot,odx,edx)
                   print(da)
                  
                   ID=list(df_sheet_map[da].columns.values.tolist())[-1]
                  
                  
                  
                   """  Voeg shift toe indien event en object dezelfde zijn als hetgeen we nu door loopen """ 
                   """  Maak een variabele aan die op deze voorwaarde filtert """ 
                   da_match_bool= df_sheet_map[da].loc[(df_sheet_map[da][ID] == odx)  &
                                       (df_sheet_map[da]['EID'] == edx)]
                   print(da_match_bool)
                   if da_match_bool.empty == False :
                       new_shift = Shift(edx,event_name, da, 0, odx, ot)
                       new_shift.attribute_value = da_match_bool.iloc[0][0]
                       """ for Quantity we need to make a dictionary of the values"""
                       activities[event_name].add_shift(odx,new_shift)
                      
                       activities[event_name].events[da].append(new_shift.attribute_value)
                       #Here we add the dynamic attributes that exist until then in a set
                       da_set.add(da)
                       
                       print(new_shift.__str__())
                   if da_match_bool.empty == True :
                       activities[event_name].events[da].append(' " ')
                      
                
                    
               #Here we add all the values of all that attribute to that event        
               for da in da_set:
                   activities[event_name].values[da] = attribute_values_dict[da]
                   
  #%%
                 
""" Small function to extract object number if there are multiple in one cell """
def find_between(s, first, last):
    if  ',' in s:
        regex = rf'{first}(.*?){last}'
        return re.findall(regex, s)
    else:
        object_no = [s]
        return object_no
                   
#%%

""" Function to find the related nodes of a model """

def find_related_nodes(level, new_model, ass_d, top_daotp, shifting_object_traces):
    
    print('shifting_object_traces', shifting_object_traces)
    
    "Initialize f with False"
    f = False
    
    for (dmn_act, dmn_att) in new_model.trained_models:
        if dmn_act.first == top_daotp.activity and dmn_act.second == top_daotp.attribute:
            f = True
            break
        
    space = ''
    for i in range(1, level):
        space += '|_'
       
        
    
    "if first condition is met and the amount of object traces is larger than the minimum amount per object type"
    if not f and len(top_daotp.values_per_object_trace) > min_object_trace[top_daotp.object_type]:   
       print('Number of objects:', len(top_daotp.values_per_object_trace), 'Minimum number of objects:', min_object_trace[top_daotp.object_type])
       
       print('\n','start loop')
       
       pairs = set()
       total_mapping = dict()
       
       for dmn_attr in top_daotp.activity.input_attributes:
           
           for d_activity in activities.values():
               
               if dmn_attr in d_activity.setting_attributes:
                  
                   print('\n')
                   print('Creating ass - ', d_activity.name, 'Attribute', dmn_attr)
                    
                   """ Create a new attribute shift sequence for that activity_attribute combination""" 
                   new_ass = AttributeShiftSequence(d_activity.shifts_per_attribute[dmn_attr])
                   print('Object_Type', new_ass.object_type)
                
                   shifting_object_traces_to_be_used = list()
                   no_shifts_distribution = {}
                   object_no2_list = list()
                   mapping2= dict()
                   
                   """ Now we loop for every object trace to calculate the correlations"""
                   for object_no in sorted(shifting_object_traces):
                       
               
                       print('Analyzing trace of object_no:', object_no,'with value', top_daotp.values_per_object_trace[object_no])
                       earliest = ass_d.get_earliest_appearance_of(d_activity, object_no)
                       """This prints the earliest event where object_no appears for that attribute"""
                       print('Earliest event:',earliest)
                       
                       if earliest == None:
                            continue
                       
                      
                       """ Now we need to match the attribute with the correct object of that event by using the object type attribute""" 
                       
                       object_nos_string = df_sheet_map['Events'].loc[df_sheet_map['Events'].index == 'e' + str(earliest),new_ass.object_type].iloc[0]

                       object_nos = find_between(object_nos_string, '\'', '\'')
                       
                       """Here we check if the attribute is created before earliest event if that is the case we calculate the correlation with that attribute"""
                       
                       """ For every object that is found we analyze its shifts and see whether these are compatible"""
                       
                       for object_no2 in object_nos:
                           print("Input attribute",  dmn_attr, "belongs to object:", object_no2 )
                           shifts_in_object_trace = new_ass.get_ordered_shifts_of_object_trace_before(object_no2, earliest)
                           """ if len(shifts_in_object_trace) == 0: then the attribute of object 2 was not created before attribute of object 1 """
                           
                           if len(shifts_in_object_trace) == 0:
                               print('Attribute 2 is not created before attribute 1')
                               break
                              
                           
                           for i in range(1, len(shifts_in_object_trace)+1):
                           
                               if i not in no_shifts_distribution.keys():
                                   
                                   no_shifts_distribution[i] = set()
                               no_shifts_distribution[i].add(object_no2)
                               
                           
                               
                           if len(no_shifts_distribution.keys()) > 0:
                               
                               shifting_object_traces_to_be_used.append(object_no)
                               print('--Add', dmn_attr, 'of object', object_no2, 'through', d_activity.name)
                               object_no2_list.append(object_no2)
                               
                               if object_no2 not in mapping2.keys():
                                   mapping2[object_no2] = set()
                               mapping2[object_no2].add(object_no)
                               
                           
                           
                   print('Distro:', no_shifts_distribution)    
                  
                  
                   shifts_to_consider = set()
                              
                   """We now know the shift distribution of object no 2 and we now check if it has enough shifts if so we continue the analysis"""
                    
                   for no_shift, object_nos in no_shifts_distribution.items():
                        print('Number of objects for each number of shifts',len(object_nos), object_nos,'minimum object trace', min_object_trace[new_ass.object_type],'number of shift', no_shift  )
                        if len(object_nos) >= min_object_trace[new_ass.object_type] and no_shift > 0:
                            shifts_to_consider.add(no_shift)
                            
                   """shift_no is the frequency of the shift """
                   
                   for shift_no in shifts_to_consider:
                   
                        print("Going for shift", shift_no, 'with #object traces:', len(no_shifts_distribution[shift_no]))
                        
                        if new_model.are_connected(top_daotp, d_activity, dmn_attr, shift_no):
                            continue
                        object_no3_list=list()
                        object_no4_list= list()
                        values_per_object_trace = {}
                       
                        for object_nox in no_shifts_distribution[shift_no]:
                            
                            
                            earliest = ass_d.get_earliest_appearance_of(top_daotp.activity, object_nox)
                            
                            print('earliest of object no2',object_nox,': ', earliest)
                            
                            shifts = new_ass.get_ordered_shifts_of_object_trace_before(object_nox, earliest)
                            shift=shifts[shift_no-1]
                            earliest1= shift.get_pos_no()
                            
                            values_per_object_trace[object_nox] = []
                            
                            values_per_object_trace[object_nox].append(shift.attribute_value)
                           
                            
                            print('map',mapping2[object_nox])
                            
                            for object_no in mapping2[object_nox]:
                            
                                earliest2 = ass_d.get_earliest_appearance_of(top_daotp.activity, object_no)
                                print('earliest',object_no,earliest2)
                                if earliest1 < earliest2:
                                    
                                    object_no3_list.append(object_nox)
                                    object_no4_list.append(object_no)
                                
                 
                                
                        
                        
                       
                        print('map',mapping2)
                        correlation = 0
                        
                        print(d_activity.name, dmn_attr, values_per_object_trace, shift_no, correlation, new_ass.object_type)
                        
                        
                        
                        new_daaotp = DMNActivityAttributeObjectTypePair(d_activity, dmn_attr, values_per_object_trace, shift_no, correlation, new_ass.object_type)
                        
                    
                        """ So we check the object traces of the attribute we want to calculate the correlation with """
                        
                        """ X is the activity of input attribute of object no 2, Y is the activity and attribute of object no 1 """
                       
                        
                        print('object_no3_list', object_no3_list)
                        
                        print('object_no4_list',object_no4_list)
                       
                        pm = PossibleModel(new_daaotp, set([top_daotp]), object_no3_list, object_no4_list)
                        
                        correlation, rf = pm.build_model2(True)
                        
                        
                        
                        
                        
                        
                        print('correlation', correlation)
                      
                        
                        
                        """Here we add the input to the pairs if the correlation is high enough and some other conditions"""
                        if correlation > min_correlation and len(values_per_object_trace.values()) > 1 \
                            and len(no_shifts_distribution[shift_no]) > min_trace_proportion :
                                
                                
                                new_daaotp.correlation = correlation
                                """mapping is a dictionary with keys the objects of the potential subdecision and value are the objects of the top_decision"""
                                
                                mapping = {object_no3_list[i]: object_no4_list[i] for i in range(len(object_no4_list))}
                                res = {}
                                for key in object_no4_list:
                                    
                                    for value in object_no3_list: 
                                        if key in res.keys():
                                            res[key].append(value)
                                            
                                        else:
                                            res[key] = []
                                            res[key].append(value)
                                            
                                        object_no3_list.remove(value)
                                        break
 
                             
                                
                                """total mapping is a dictionary with keys the subdecision and value the mapping object defined above """
                                
                                total_mapping[new_daaotp] = res
                                pairs.add(new_daaotp)
                                
                                
        
                        
   
       print('total_mapping',total_mapping)
    
       pairs = sorted(pairs)
       print("#Pairs:", len(pairs))
       
       
       
          
       possible_models = find_possible_models3(top_daotp, pairs,total_mapping)
       print('\n')
       
      
       to_remove = set()
       
    
       for pm in possible_models:
           for pm2 in possible_models:
               if pm != pm2:
                   if pm.contains(pm2):
                       to_remove.add(pm2)
                       
       possible_models = possible_models.difference(to_remove)
       
      
       
       
       for possible_model in possible_models:
           print('Possible model:', possible_model)
           
           if len(possible_model.object_nos) > min_trace_proportion:
               
               accuracy, pred_model = possible_model.build_model2(False)
               
               if accuracy > min_support:
                   top_daotp.classifier = pred_model
                   inter = set(possible_model.object_nos).intersection(new_model.object_nos)
                   print(possible_model.object_nos)
                   print(new_model.object_nos)
                   print(len(new_model.object_nos), 'vs', len(set(possible_model.object_nos)))
                   
                   if len(inter) == len(new_model.object_nos):
                       models_string =''
                       for daap in possible_model.daaps:
                           models_string += daap.small_string() + ', '
                       print("\n--------------------------------\n(1) Extending model with " + models_string)
                     
                       
                       add_nodes(level, new_model, top_daotp, possible_model,total_mapping)
                   

                   else:
                        print("\n--------------------------------\n(2) Extending model with " + str(possible_model.daaps))
                        
                        newer_model = new_model.clone(new_model.label + "_"
                                                      + str(len(possible_model.object_nos)), possible_model.object_nos)
                        models.add(newer_model)
                        new_model.sub_models.add(newer_model)
                        newer_model.super_model = new_model
                        add_nodes(level, newer_model, top_daotp, possible_model,total_mapping)
                       
                       
                   
               
     
        
     
#%%
def add_nodes(level, dmn_model, top_daotp, possible_model,total_mapping):

    
    for daap in possible_model.daaps:
        """Here I need to make a new objectlist with the mapping object to provide the correct objects"""
        object1list=list()
        remove = list()
        
        for obj in top_daotp.values_per_object_trace.keys():
            try:
             object1list.append(total_mapping[daap][obj][0])
            except:
                remove.append(obj)
        for obj in remove:
            del top_daotp.values_per_object_trace[obj]
        

        print(object1list)
        print(daap)
        print(list(top_daotp.values_per_object_trace.keys()))
        print('Level:', level)
        #breakpoint()
        """Here I had to make a list out of dmn_model.object_nos since it is originally a set"""
        new_possible_model = PossibleModel(daap, set([top_daotp]), object1list,list(top_daotp.values_per_object_trace.keys()))
        print(len(object1list))
        print(len(top_daotp.values_per_object_trace.keys()))
        
        corr, rf = new_possible_model.build_model2(True)
        print('Correlation:', corr)
        
        if corr > min_correlation:
            dmn_model.add_edge(daap, top_daotp, len(possible_model.object_nos), corr)
    
    for daap in possible_model.daaps:
        space = ""
        for le in range(1, level):
            space += "|_"
        print('\n\n', space, 'Going down the rabbit hole for', daap.__str__())    
        print(len(possible_model.object_nos))
  
        found_ass = find_ass(daap.values_per_object_trace.keys(), daap.no_shift, daap.activity, daap.attribute)
     
        print(daap.values_per_object_trace.keys())
        find_related_nodes(level+1, dmn_model, found_ass, daap, daap.values_per_object_trace.keys())
    
      
        
        
    
#%%
#%%
def find_ass(shifting_object_traces, no_shift, activity, dmn_attr):
    shifts_for_attr = set()
    
    for object_no in shifting_object_traces:
        if len(activity.get_shifts_in_object_trace(dmn_attr, object_no)) >= no_shift:
            shifts_for_attr.add(activity.get_shifts_in_object_trace(dmn_attr, object_no)[no_shift - 1])
    print('Found ass:', len(shifts_for_attr))
    
    return AttributeShiftSequence(shifts_for_attr)


#%%   
       
#%%
def find_possible_models3(top_daotp,pairs,total_mapping):
    possible_models = set()
    
    for daaotp in pairs:
        print('\nChecking for possible models', daaotp)
        mapping = total_mapping[daaotp]
        print('mapping', mapping)
        
        to_add = set()
        
        for possible_model in possible_models:
            print('possible model:', possible_model)
            if daaotp not in possible_model.daaps:
               
                intersecting_object_traces = set(mapping.keys()).intersection(possible_model.object_nos)
                print('Intersecting_traces:', len(intersecting_object_traces))
                
                #If the length of intersecting traces and possibles_models.objectnos is equal they must cover the same ground
                if len(intersecting_object_traces) == len(possible_model.object_nos):
                    
                    
                    object2_list = []
                    for key in intersecting_object_traces:
                        object2_list = [*object2_list,*mapping[key]]
                   
                    print(object2_list)       
                    possible_model.add_daaotp(daaotp, object2_list)
 
                elif len(intersecting_object_traces) == len(daaotp.values_per_object_trace.keys()):
                    
                    object2_list = []
                    object1_list = [] 
                    for key in intersecting_object_traces:
                        object2_list = [*object2_list,*mapping[key]]
                        
                        for i in range(len(mapping[key])):
                            object1_list.append(key)
                    print(object2_list)        
                    new_possible_model = PossibleModel(top_daotp, possible_model.daaps, object1_list, object2_list)
                    possible_model.add_daaotp(daaotp, object2_list)
                    
                    if len(daaotp.values_per_object_trace.keys()) > min_trace_proportion:
                        to_add.add(new_possible_model)
                
                elif len(intersecting_object_traces) > 0:
            
                        object2_list = []
                        object1_list = [] 
                        for key in intersecting_object_traces:
                            object2_list = [*object2_list,*mapping[key]]
                            
                            for i in range(len(mapping[key])):
                                object1_list.append(key)
                        print(object2_list)  
                        if len(intersecting_object_traces) >= len(possible_model.object_nos) * min_deviation:
                            possible_model.add_daaotp(daaotp, object2_list)
                            possible_model.object_nos = intersecting_object_traces
                            
                        else:
                            print('1', object1_list)
                            print('2',object2_list)
                            new_possible_model = PossibleModel(top_daotp, possible_model.daaps, object1_list, object2_list) 
                            if len(daaotp.values_per_object_trace.keys()) > min_trace_proportion:
                                possible_model.add_daaotp(daaotp, object2_list)
                            to_add.add(new_possible_model)
                
                
             
                
        
        
        possible_models.update(to_add)
        
        
        
        #set(mapping.values()) is a set of all the object traces of the top decision that are covered by the subdecision
        not_covered = not_covered_by_models(possible_models, set(mapping.keys()))
        if len(not_covered) > min_trace_proportion:
           object2_list = []
           object1_list = [] 
           for key in not_covered:
               object2_list = [*object2_list,*mapping[key]]
               
               for i in range(len(mapping[key])):
                   object1_list.append(key)
           
            
                
           print('New not covered:', len(not_covered))
           print(object1_list)
           new_possible_model = PossibleModel(top_daotp, set([daaotp]), object1_list, object2_list)
           possible_models.add(new_possible_model)
        
    
    return possible_models


#%%      
        
                 
                     
                                   
                       
                       
                       
                       
                       
                     
            
                   
               
               

    
    
    
    
                  
 #%%   

def not_covered_by_models(possible_models, object_nos):      
    not_covered = set()
    for object_no in object_nos:
        contained = False
        for possible_model in possible_models:
            if object_no in possible_model.object_nos:
                contained = True
                break
        if not contained:
            not_covered.add(object_no)
    return not_covered

#%%       
#%%




"""Parameters related to attributes and objects of the event log"""



object_types_list = ['Authors','Books','Publishers']

dynamic_attributes_dict = {'Authors':[],
                           'Books':['Publication_Status', 'Review Score', 'Quality','Compliant'],'Publishers':[]}


attributes_dict  = {'Authors':['Name','Number of published books','Author Specialty Genre'],
                           'Books':['Genre', 'Number_of_pages','Quality','Publication_Status','Review Score','Compliant'],
                           'Publishers':['Name', 'Publisher Specialty Genre']}

"""
attributes_dict  = {'Authors':['Number of published books'],
                           'Books':['Quality','Publication_Status','Review Score','Compliant'],
                           'Publishers':[]}
"""
#%%

"""Hyperparameters"""
#Hoeveel shifts er moeten voorkomen vooraleer dat die shift combinatie interessant genoeg wordt geacht
min_shift = 0.15

#Proportie die bepaalt wanneer er genoeg object_nos gedekt zijn door de shifts
min_trace_proportion = 0.2
#Minimum correlatie dat twee variabelen moeten hebben vooraleer ze interessant genoeg worden geacht
min_correlation = 0.08

#Variabele die kijkt hoeveel van de intersecting traces gedekt moet zijn om te mogen toevoegen aan het model
#Dus hoe kleiner hoe meer verschillende modellen zou ik zeggen
min_deviation = 0.03

#Minimum accuracy dat uw predictief model moet hebben om toegevoegd te worden
min_support = 0.2

#%%
"""  Read in the excel file with all its sheets. Provide the path to the excel file"""
df_sheet_map, attributes, attribute_values_dict = read_docel_log('artificial log.xlsx')


df_sheet_map['Authors']['Name'] = df_sheet_map['Authors']['Name'].astype('str')
df_sheet_map['Authors']['Number of published books'] = df_sheet_map['Authors']['Number of published books'].astype('int64')
df_sheet_map['Authors']['Author Specialty Genre'] = df_sheet_map['Authors']['Author Specialty Genre'].astype('str')

df_sheet_map['Books']['Genre'] = df_sheet_map['Books']['Genre'].astype('str')
df_sheet_map['Books']['Number_of_pages'] = df_sheet_map['Books']['Number_of_pages'].astype('int64')

df_sheet_map['Publishers']['Name'] = df_sheet_map['Publishers']['Name'].astype('str')
df_sheet_map['Publishers']['Publisher Specialty Genre'] = df_sheet_map['Publishers']['Publisher Specialty Genre'].astype('str')


df_sheet_map['Publication_Status']['Publication_Status'] = df_sheet_map['Publication_Status']['Publication_Status'].astype('str')
df_sheet_map['Quality']['Quality'] = df_sheet_map['Quality']['Quality'].astype('str')
df_sheet_map['Review Score']['Review Score'] = df_sheet_map['Review Score']['Review Score'].astype('int64')
df_sheet_map['Compliant']['Compliant'] = df_sheet_map['Compliant']['Compliant'].astype('str')

print(attributes)
print(df_sheet_map)



#%%

""" Use a list of unique activities and create the object activities """ 

"""Object dictionary is a dictionary with as key object type and all of its objects as its value"""
object_dict = {}

activities= {}
for act in df_sheet_map['Events'].Activity.unique().tolist():
     #print(act)
     new_act = Activity(act)
     activities[act] = new_act
     
     for attribute in attributes:
        #print(attribute)
        new_act.events[attribute] = []
        new_act.shifts_per_attribute[attribute] = []

#%%
        
"""Now we discover shifts"""

discover_shifts(df_sheet_map,activities, attribute_values_dict, object_dict)


#%%

      
""" To discover potential input and output attributes""" 

 
for activity in activities.values():
    activity.analyse_attributes(min_shift,object_dict)
    
#%%

""" min_obj_trace is a dictionary with keys object type and as value the minimum amount of objects for which a ATAOTS should be valid"""    
min_object_trace = {}

for key in object_dict.keys():
    min_object_trace[key]= len(object_dict[key]) * min_trace_proportion
    print('Minimum object_traces of object type ', key, ': ', min_object_trace[key])

models = set()
 
#%%

""" Now we start building the actual models"""

print('\nBuilding model(s) for activities:\n')   

for activity_name, top_activity in activities.items():
         print('Activity:', activity_name)
         
         """For every attribute that is set by the activity"""
         for attribute in top_activity.setting_attributes:
             for key in attributes_dict.keys():
                 if attribute in attributes_dict[key]:
                     Object_Type= key
             print('Attribute:', attribute, 'of Object Type', Object_Type)
             shifting_object_traces = top_activity.get_shift_object_trace_nos(attribute)
             
            
             """ shift distribution is a dictionary with as keys the number of shifts and the values the length of object traces that have that amount of changes in them"""
             shift_distribution = {}
             
             for object_no in shifting_object_traces:
                 no_shifts_in_obj_trace = len(top_activity.get_shifts_in_object_trace(attribute, object_no))
                 if no_shifts_in_obj_trace > 0:
                     if no_shifts_in_obj_trace in shift_distribution.keys():
                         shift_distribution[no_shifts_in_obj_trace] += 1
                     else:
                         shift_distribution[no_shifts_in_obj_trace] = 1
                         
             print(shift_distribution)
            
          
             for no_shifts in range(1,max(shift_distribution.keys())+1):     
                 
                 

                 for shift in shift_distribution.keys():
                    if shift >= no_shifts:
                        distribution_value = shift_distribution[shift]
                        break

                 
                 
                 print('number of shifts',no_shifts, distribution_value)
                 shifts_of_attribute = set()
                 values_per_object_trace = {}
                 
                 subset_of_object_traces = set()

                 for object_no in shifting_object_traces:
                     no_shifts_in_object_trace = len(top_activity.get_shifts_in_object_trace(attribute, object_no))
                     
                     if no_shifts_in_object_trace >= no_shifts:
                         subset_of_object_traces.add(object_no)
                         
                         values_per_object_trace[object_no] = []
                         
                         shift_at_position = top_activity.get_shifts_in_object_trace(attribute, object_no)
                         
                         
                         for shift in shift_at_position:
                             shifts_of_attribute.add(shift)
                             values_per_object_trace[object_no].append(shift.attribute_value)
                             
                             
                     
                        
                     """Here we only keep the values of the last shift """       
                     if object_no in values_per_object_trace.keys():
                            del values_per_object_trace[object_no][0:no_shifts-1]
                            
                            #print('olaaaa', values_per_object_trace[object_no])
                             
                 print('All values of the object traces', values_per_object_trace)  
                
                 if len(subset_of_object_traces) < min_object_trace[Object_Type]:  
                     continue
                 
                 ass = AttributeShiftSequence(shifts_of_attribute)
                 
                 
                 
                 print("------------------------------------------------------")
                 print(top_activity.name + " shifts " + str(no_shifts) + " time(s) setting attribute: " + str(attribute) + " in " + str(len(subset_of_object_traces)) + " object traces")
                 print("------------------------------------------------------")
                 print('\n')
                 
                 """ We create a DMNActivityAttributeObjectTypePair """
                 daaotp = DMNActivityAttributeObjectTypePair(top_activity, attribute, values_per_object_trace, no_shifts, 1, Object_Type)
                 
                 """ top activity has X shifts of attribute in Y object traces of object Type"""
                 model_label = top_activity.name + '(' + str(no_shifts) +')->' + str(attribute) \
                          + '_' + str(len(subset_of_object_traces)) + ' of Object Type ' + Object_Type 
                          
                 """ We create a potentially new model with a subset of object traces, some DMNactivittyAttributeObjectType paires and a label """  
                 
                 new_model = DMNModel(model_label, subset_of_object_traces,  daaotp)
                 models.add(new_model)
                 
                 
                
                 find_related_nodes(1, new_model, ass, daaotp, subset_of_object_traces)
                 
                 
                 
                 
                 
                 
                 
                 
                 
#This whole part works!!!                 

trace_clusters = []
keep = set()
top_models = set()

model_strings = []

for model in models:
    

    if model.get_no_edges() > 1:
        
        print(model.label, 'has a trace cluster of size', len(model.object_nos))
        model_strings.append(model.label + ' has a trace cluster of size ' + str(len(model.object_nos)))
        trace_clusters.append(model.object_nos)
        keep.add(model)
        if model.super_model == None:
            top_models.add(model)
for model_s in sorted(model_strings):
    print(model_s)

models = models.intersection(keep)

results_file = open('results_models.csv', 'w')
results_file.write('model_no,model,type,no_lines,no_labels,no_leaves,no_nodes,acc_dt,f1_dt,acc_rf,f1_rf,acc_lrr,'
           'f1_lrr,no_rules_lrr,terms_lrr,acc_br,f1_br,no_rules,terms_br\n')

for mi, model in enumerate(models):
    g = Digraph('G', filename='./output/model_' + str(mi), format='png',strict=True)
    # nxg = nx.DiGraph()
    g.graph_attr.update(rankdir='BT')
    
    g.attr('node',shape='note')
    sources= list()
    targets = list()
    annotations=list()
    PossibleModel.MAX_DEPTH = 8
    

    for edge in model.get_edges():
        sources.append(edge.source.small_string2())
        targets.append(edge.target.small_string2())
      
    difference = list(set(sources) - set(targets))
    print('difference',difference)
    
    
    
    for edge in model.get_edges():
        
        if edge.target.classifier != None:
            rf, accu, accu_b, xs, ys, info_string, br_rules = edge.target.classifier[0]
            print(edge.correlation)
            print(model)
            print(rf)
            print(xs)
            print(ys)
            print('model_name',str(edge.target))
            
            results_file.write(str(mi)+',' + info_string)
            file_name = './output/' + str(mi)+"-"+str(edge.target)+"_"+str(round(accu, 2)) +"_tcs_" +str(len(model.object_nos))
            file_name_rule = './output/' + str(mi)+"-"+str(edge.target)+"_"+str(round(accu_b, 2)) +"_tcs_" +str(len(model.object_nos))
            rule_file = open(file_name_rule +'.txt', 'w')
            for line in edge.target.classifier[1]:
                rule_file.write(line)
            rule_file.close()
            tree.export_graphviz(rf,
                                 out_file=file_name + '.dot',
                                 feature_names=xs,
                                 class_names=ys,
                                 filled=True)
            
            check_call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png'])
            os.remove(file_name + '.dot')
            

        edge_contained = ''
        for rule in br_rules:
            if str(edge.source) in rule or str(edge.target) in rule:
                edge_contained += rule + '\n'
        if edge_contained == '':
            g.edge(edge.source.small_string2(), edge.target.small_string2(),label=str(edge.weight),color='black',splines='true')
            g.edge(edge.source.small_string2(), edge.source.small_string3(),color='black',arrowhead='none',style='dashed',splines='true')
            g.edge(edge.target.small_string2(), edge.target.small_string3(),color='black',arrowhead='none',style='dashed',splines='true')
            annotations.append(edge.source.small_string3())
            annotations.append(edge.target.small_string3())
            
            
          
            
        else:
            g.attr('edge', color='blue')
            
            g.edge(str(edge.source), str(edge.target), label=edge_contained)
           
    

    
    for node in difference:
        g.node(node,style='rounded',shape='ellipse')
    
    for node in targets:
        g.node(node, shape='box')
        
    for node in annotations:
        g.node(node,color='transparent', image='textannotation2.png',fontsize='11')
 
    print(g)   
   
    

    
    
        
        
        
    g.view()

results_file.close()
print('#Models retained:', len(models))
print('#Top models:', len(top_models))
                         
     #%%                    
                         
                         

                         
                         
                         
                         
                         
                         
                         
                         
                    