#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:45:18 2022

@author: alexandre
"""

from collections import OrderedDict
from functools import total_ordering


class AttributeShiftSequence:

    MAXINT = 2000000

    def __init__(self, shifts):
        self.object_traces = {}
        self.event_maps = {}
        self.object_type = 'Type'

        for shift in shifts:
            self.event_maps[shift.event_name] = set()
            self.object_type = shift.object_type
            
            if shift.object_no in self.object_traces.keys():
                self.object_traces[shift.object_no].add(shift)
            else:
                self.object_traces[shift.object_no] = set([shift])

        for et in self.event_maps.keys():
            self.event_maps[et] = {}
            for et2 in self.event_maps.keys():
                if et != et2:
                    self.event_maps[et][et2] = 0

        for object_no, shifts in self.object_traces.items():
            self.calculate_orders(shifts)



    def calculate_orders(self, shifts):

        for et in self.event_maps.keys():
            min_e1 = self.MAXINT

            for shift in shifts:
                if shift.event_name == et:
                    if shift.get_pos_no() < min_e1:
                        min_e1 = shift.get_pos_no()

            for et2 in self.event_maps.keys():
                if et != et2:
                    min_e2 = self.MAXINT
                    for shift in shifts:
                        if shift.event_name == et2:
                            if shift.get_pos_no() < min_e2:
                                min_e2 = shift.get_pos_no()

                    if min_e1 < min_e2 != self.MAXINT and min_e1 != self.MAXINT:
                        self.event_maps[et][et2] += 1
                        
    def get_earliest_appearance_of(self, activity, object_no):
        earliest_pos_no = self.MAXINT
        if object_no not in self.object_traces:
            return earliest_pos_no

        for shift in self.object_traces[object_no]:
            #print('Shift:', shift)
            if shift.get_pos_no() < earliest_pos_no:
                earliest_pos_no = shift.get_pos_no()
        return earliest_pos_no
    
    def get_ordered_shifts(self, object_no):
        shift_dict = {}
        for shift in self.object_traces[object_no]:
            shift_dict[shift.get_pos_no()] = shift
        return OrderedDict(sorted(shift_dict.items(), key=lambda t: t[0])).values()

    def get_ordered_shifts_of_object_trace_before(self, object_no, position):
        shifts = []
        if object_no in self.object_traces.keys():
            for shift in self.get_ordered_shifts(object_no):
                if shift.get_pos_no() < position:
                    shifts.append(shift)
        return shifts


class Activity:

    def __init__(self, name):
        self.name = name
        self.shifts = {}
        self.values = {}
        self.events = {}
        self.input_attributes = set()
        self.setting_attributes = set()
        self.shifts_per_attribute = {}
        
    def add_shift(self, object_no, shift):
      if object_no not in self.shifts.keys():
          self.shifts[object_no] = []
      self.shifts[object_no].append(shift)
      self.shifts_per_attribute[shift.attribute].append(shift)
      
    def analyse_attributes(self, shift_ratio, object_dict):
        if len(self.shifts) == 0:
            print(self.name, 'does not have enough shifts')
            return

        print('\n', self.name)
        to_remove = set()
        
      
        
        # Key is an object_id
        length_key = 0
        for key in self.shifts.keys():
         length_key = length_key + len(self.shifts[key])
       
        
        # Here I count the number of events
        number_events = 0
        for key in self.events.keys():
            if number_events < len(self.events[key]):
                number_events = len(self.events[key])
        
        print('#Total number of events:', number_events)
        
        
    
        """Here you remove all the attributes that are not discriminative"""
        for attribute in self.values.keys():
            # print(attribute, len(self.values[attribute]))
            # print(self.values[attribute])
            if len(self.values[attribute]) > 1:
                #print('Attribute', attribute, 'is added as input')
                self.input_attributes.add(attribute)
            else:
                to_remove.add(attribute)
        print('To remove:', to_remove)

        for v in to_remove:
            self.values.pop(v)
            # self.events.pop(v)
            # self.shifts_per_attribute.pop(v)

        for attribute in self.shifts_per_attribute.keys():
            print(attribute)
            
            
            """For each attribute check its object type and check how many objects there exist for that type"""
            #if  self.shifts_per_attribute[attribute]:
                #print("List is filled")
                #print(self.shifts_per_attribute[attribute][0].object_type)
                #print(len(object_dict[self.shifts_per_attribute[attribute][0].object_type]))
                
         
            
            events = self.events[attribute]
           
            
            
            print('#Events:', len(events))

            min_shift = round(shift_ratio * len(events))
            print('Min shift:', min_shift)
            print('#Shifts for attribute:', len(self.shifts_per_attribute[attribute]))
            if len(self.shifts_per_attribute[attribute]) > min_shift:
                print("->>>>", attribute, " passed minShift")
                self.setting_attributes.add(attribute)

        print('Input attributes:', self.input_attributes)
        print('Setting attributes:', self.setting_attributes)
        
        
    def get_shift_object_trace_nos(self, attribute):
        object_traces = set()
        for shift in self.shifts_per_attribute[attribute]:
            object_traces.add(shift.object_no)
        return object_traces
    
    def get_shifts_in_object_trace(self, attribute, object_no):
        shifts = []
        for shift in self.shifts_per_attribute[attribute]:
            if shift.object_no == object_no:
                shifts.append(shift)
        return shifts
        

class Shift:
    def __init__(self, event, event_name, attribute, attribute_value, object_no, object_type):
        self.event = event
        self.event_name= event_name
        self.attribute = attribute
        self.object_no, = object_no,
        self.object_type = object_type
        self.attribute_value = attribute_value

    def __str__(self):
        return 'Shift of Attribute ' + str(self.attribute) + ' in object_no ' + str(self.object_no) + ' of object type ' \
               + str(self.object_type) + ' by event ' + str(self.event + ' to value ' + str(self.attribute_value))
               
    def get_pos_no(self):
        filtered_str_iterable = filter(str.isdigit, self.event)
        pos_no = ''.join(filtered_str_iterable)
        return int(pos_no)
    
@total_ordering
class DMNActivityAttributeObjectTypePair:

    def __init__(self, activity, attribute, values_per_object_trace, no_shift, correlation, object_type):
        self.activity = activity
        self.attribute = attribute
        self.values_per_object_trace = values_per_object_trace
        self.no_shift = no_shift
        self.correlation = correlation
        self.classifier = None
        self.object_type = object_type
        
    def __str__(self):
        return self.small_string().replace(':', '')
     

    def small_string(self):
        return str(self.activity.name) + '--' + str(self.attribute) + '_shift-' + str(self.no_shift)  + '--Object_Type:--' + str(self.object_type)
    
    def small_string2(self):
        return str(self.attribute) + '_shift-' + str(self.no_shift) 
    
    def small_string3(self):
        return  '  Activity- '+ str(self.activity.name) +'\l' + '\\n  Object_Type-' + str(self.object_type) +'\l'

    def __lt__(self, other):
        return len(self.values_per_object_trace) > len(other.values_per_object_trace)