# -*- coding: utf-8 -*-
"""
Implements the Oslo model

Also contains other functions to facilitate tasks related to this model
"""

import numpy as np
import random 
import matplotlib.pyplot as plt
import time
from numba import njit
import pickle

#%%

@njit
def critical(size, slopes, thresholds):
    for i in range(size):
        if slopes[i] > thresholds[i]:
            return True
    return False


@njit
def method(slopes, heights, thresholds, size, steady):
    avalanche = 0 # if no avalanchhe occurs, size is 0
        
    while critical(size, slopes, thresholds): # loop until all sites are relaxed
        
        for i in np.arange(0, size):
            if slopes[i] > thresholds[i]: # avalanche occurs
                avalanche += 1
                thresholds[i] = random.randint(1,2) # np.random.choice([1,2], p=self._prob) # new threshold slope after an avalanche
                if i == 0: # left-most site
                    slopes[i] -= 2
                    heights[i] -= 1
                    slopes[i+1] += 1
                    heights[i+1] += 1
                elif i == size - 1:
                    slopes[i] -= 1
                    heights[i] -= 1
                    slopes[i-1] += 1
                    steady = True # system is in steady state when it first overflows
                else:
                    slopes[i] -= 2
                    heights[i] -= 1
                    slopes[i+1] += 1
                    heights[i+1] += 1
                    slopes[i-1] += 1
        
    return slopes, heights, thresholds, avalanche, steady
        
   
#%%
    #a = model._slopes
    """
    avalanche = 0 # if no avalanchhe occurs, size is 0
    while any(model._slopes > model._thresholds): # loop until all sites are relaxed
        #start = min(np.where(model._slopes > model._thresholds)[0])
        #end = max(np.where(model._slopes > model._thresholds)[0]) + 1
        #np.argmax(self._slopes > self._thresholds) # this is the index where the for loop should start 
        #for i in np.arange(start, model._size): # change this to the first position of the if statement
        #for i in np.arange(start, end):
    
        for i in np.arange(0, model._size):
            if model._slopes[i] > model._thresholds[i]: # avalanche occurs
                if i == 0: # left-most site
                    model._slopes[i] -= 2
                    model._heights[i] -= 1
                    model._slopes[i+1] += 1
                    model._heights[i+1] += 1
                elif i == model._size - 1:
                    model._slopes[i] -= 1
                    model._heights[i] -= 1
                    model._slopes[i-1] += 1
                    model._steady = True # system is in steady state when it first overflows
                else:
                    model._slopes[i] -= 2
                    model._heights[i] -= 1
                    model._slopes[i+1] += 1
                    model._heights[i+1] += 1
                    model._slopes[i-1] += 1
    
                model._thresholds[i] = model.choose_threshold() #random.randint(1,2) # np.random.choice([1,2], p=self._prob) # new threshold slope after an avalanche
                avalanche += 1
    model._avalanches.append(avalanche) # update avlanche list
    """
#%%

class oslo:
    def __init__(self, L, p):
        self._size = L
        self._prob = p # list of values
        self._heights =  np.zeros(L) # stores the heights of all sites
        self._slopes = np.zeros(L) # stores the slopes of all sites
        self._thresholds = np.zeros(L) # stores the threshold slopes of all sites
        self._avalanches = [] # stores the avalanche sizes
        self._steady = False # not in steady state initially
        self._average = 0
    
    def choose_threshold(self):
        x = random.random()
        if self._prob[0] > x:
            return 1
        else:
            return 2
        
    """
    Randomly choose a threshold slope for each site
    """
    
    def initialize(self): 
        for i in range(self._size):
            self._thresholds[i] = self.choose_threshold() #random.randint(1,2) #np.random.choice([1,2], p=self._prob) # 1 or 2 with equal probability
            
    """
    Add grain at the left-most site
    """
    
    def drive(self): 
        self._slopes[0] += 1
        self._heights[0] += 1
    """
    def critical(self):
        for i in range(self._size):
            if self._slopes[i] > self._thresholds[i]:
                return True
        return False
    """
    
    """
    Relax all sites which have a slope > threshold slope
    Continue looping over all sites until they are all relaxed
    A later avalanche can lead to a new avalanche further up the site!
    """ 
   
    def relax(self): 
        avalanche = 0 # if no avalanchhe occurs, size is 0
        #while self.critical():
        while any(self._slopes > self._thresholds): # loop until all sites are relaxed
            start = min(np.where(self._slopes > self._thresholds)[0])
            end = max(np.where(self._slopes > self._thresholds)[0]) + 1
            #np.argmax(self._slopes > self._thresholds) # this is the index where the for loop should start 
            #for i in np.arange(start, self._size): # change this to the first position of the if statement
            for i in np.arange(start, end):
                if self._slopes[i] > self._thresholds[i]: # avalanche occurs
                    if i == 0: # left-most site
                        self._slopes[i] -= 2
                        self._heights[i] -= 1
                        self._slopes[i+1] += 1
                        self._heights[i+1] += 1
                    elif i == self._size - 1:
                        self._slopes[i] -= 1
                        self._heights[i] -= 1
                        self._slopes[i-1] += 1
                        self._steady = True # system is in steady state when it first overflows
                    else:
                        self._slopes[i] -= 2
                        self._heights[i] -= 1
                        self._slopes[i+1] += 1
                        self._heights[i+1] += 1
                        self._slopes[i-1] += 1
                    self._thresholds[i] = self.choose_threshold() #random.randint(1,2) # np.random.choice([1,2], p=self._prob) # new threshold slope after an avalanche
                    avalanche += 1
            """
            To make the code more efficient, instead of starting from site 1 in each for loop
            We find the first position where an avalanche will occur 
            """
        # np.append(self._avalanches, avalanche) # use only for njit
        self._avalanches.append(avalanche) # update avlanche list
       
    def test(self):
        self._slopes, self._heights, self._thresholds, avalanche, self._steady = method(self._slopes, self._heights, self._thresholds, self._size, self._steady)
        self._avalanches.append(avalanche)
    """
    Initialize the system in the empty configuration and iterate over the specified number of times
    """
    def run(self, iterations):
        self.initialize()
        iterations_steady = [] # these iterations have reached steady state
        for i in range(iterations):
            self.drive()
            self.relax()
            """
            If in steady state, starts calculating the average until all iterations are over
            """
            if self._steady == True:
                self._average += self._heights[0]
                iterations_steady.append(i+1)
        self._average = self._average / (iterations - (iterations_steady[0])) # calculate average

    def count_threshold_slopes(self):
        self.initialize()
        p_2 = 0
        p_1 = 0
        tc = 0
        while self._steady != True:
            self.drive()
            self.relax()
            tc +=  1
            thresholds = self._thresholds.tolist()
            count_2 = thresholds.count(2)
            #print(count_2)
            p_2 += count_2 / self._size
            count_1 = thresholds.count(1)
            p_1 += count_1 / self._size
        p_2 = p_2 / tc
        p_1 = p_1 / tc
        return p_1, p_2
    
    def count_slopes(self, extra_iterations):
        self.initialize()
        tc = 0
        average_slopes = []
        while self._steady != True:
            self.drive()
            self.relax()
            tc +=  1
        for i in range(extra_iterations):
            average_slope = sum(self._slopes) / self._size
            self.drive()
            self.relax()
            average_slopes.append(average_slope)
        average = sum(average_slopes) / extra_iterations
        return average
        
    def count_reccurent_configurations(self, theoretical_value, extra_iterations):
        self.initialize()
        configurations = [] # heights
        while self._steady != True:
            self.drive()
            self.relax()
        while len(set(configurations)) != theoretical_value:
            configurations.append(tuple(self._heights))
            self.drive()
            self.relax()
            print(len(set(configurations)))
        print("Match theoretical value")
        for i in range(extra_iterations): # to confirm there are no other reccurent configurations
            configurations.append(tuple(self._heights))
            self.drive()
            self.relax()
        if len(set(configurations)) == theoretical_value:
            print("There is no new reccurent configuration")
        else:
            print("There are new recurrent configurations")
            
    def get_heights(self):
        print("heights =", self._heights)
    
    def get_avalanches(self):
        print("Avalanche sizes =", self._avalanches)
        
    def get_average(self):
        print("Average height at site 1 =", self._average)

    """
    Return the total height (height at site 1) over all iterations
    """
    def find_total_height(self, extra_iterations):
        tic = time.time()
        total = []
        self.initialize()
        tc = 0 # initial value
        while self._steady != True:
            self.drive()
            self.relax()
            #relax(self)
            #self.relax()
            total.append(self._heights[0]) # append the height of site 1
            tc += 1
            # print("time", tc)
        #print("ended transient")
        for i in range(extra_iterations):
            self.drive()
            self.relax()
            #relax(self)
            #self.relax()
            total.append(self._heights[0]) # append the height of site 1
        toc = time.time()
        print("time taken = ", toc - tic)
        return total, tc
    
"""
Calculate 
1. average height after reaching steady state 
2. standard deviation of the height
3. height probability
for 1 system size
"""
def measure(system, extra_iterations):
    """
    Pick only the first realisation
    """
    #heights = system.iloc[0] # heights at all sites
    heights = system[0][0] # heights at all sites
    #tc = system.iloc[-1][0] # cross-over time
    tc = system[-1][0] # cross-over time
    heights_steady = np.array(heights[int(tc)+1:int(tc)+extra_iterations+1]) # in steady state
    T = len(heights_steady)
    
    def find_av_height():
        av_height = sum(heights_steady) / T
        sigma_height = np.std(heights_steady) / T
        return av_height, sigma_height
    
    def find_sigma():
        sigma = np.sqrt(sum(heights_steady **2) / T - (sum(heights_steady) / T) ** 2)
        return sigma
    
    def find_prob():
        prob_hist = np.histogram(heights_steady, bins=np.arange(min(heights_steady), max(heights_steady)+2), density=True) 
        return prob_hist
    
    av_height, sigma_height = find_av_height()
    sigma = find_sigma()
    h_prob, h = find_prob()
    
    return av_height, sigma, h_prob, h, sigma_height

#%%
"""
Calculate height probability
"""
def process_log_bin_data(system, extra_iterations):
    avalanches = system[1][0] # first realization
    tc = system[-1][0]
    avalanches_steady = avalanches[int(tc)+1:int(tc)+extra_iterations+1]
    
    def find_prob():
        prob_hist = np.histogram(avalanches_steady, bins=np.arange(min(avalanches_steady), max(avalanches_steady)+2), density=True) 
        return prob_hist
    
    s_prob, s = find_prob()
    
    return avalanches_steady, s_prob, s

def find_av_kmoment(system, extra_iterations, k):
    avalanches = system[1][0] # first realization
    tc = system[-1][0]
    avalanches_steady = avalanches[int(tc)+1:int(tc)+extra_iterations+1]
    T = len(avalanches_steady)
    
    av_kmoment = sum(np.array(avalanches_steady, dtype=np.float64) ** k) / T
    sigma = np.std(np.array(avalanches_steady, dtype=np.float64) ** k) / T
    
    return av_kmoment, sigma
    
    
"""
oslo4_heights = []
oslo4_avalanches = []
oslo4_tcs = []

print("L = ", 4)
for i in range(20):
    print("iteration = ", i)
    oslo4 = oslo(4, [0.5,0.5])
    oslo4_height, oslo4_tc = oslo4.find_total_height(1000000) # extra 1M
    oslo4_heights.append(oslo4_height)
    oslo4_avalanches.append(oslo4._avalanches)
    oslo4_tcs.append(oslo4_tc)
    
#%%

arr4 = np.array([oslo4_heights, oslo4_avalanches, oslo4_tcs], dtype=object)

pickle.dump(arr4, open("files/task3a/oslo4.pkl", "wb"))

#%%
oslo8_heights = []
oslo8_avalanches = []
oslo8_tcs = []

print("L = ", 8)
for i in range(20):
    print("iteration = ", i)
    oslo8 = oslo(8, [0.5,0.5])
    oslo8_height, oslo8_tc = oslo8.find_total_height(1000000) # extra 1M
    oslo8_heights.append(oslo8_height)
    oslo8_avalanches.append(oslo8._avalanches)
    oslo8_tcs.append(oslo8_tc)

arr8 = np.array([oslo8_heights, oslo8_avalanches, oslo8_tcs], dtype=object)

pickle.dump(arr8, open("files/task3a/oslo8.pkl", "wb"))


oslo16_heights = []
oslo16_avalanches = []
oslo16_tcs = []

print("L = ", 16)
for i in range(20):
    print("iteration = ", i)
    oslo16 = oslo(16, [0.5,0.5])
    oslo16_height, oslo16_tc = oslo16.find_total_height(1000000) # extra 1M
    oslo16_heights.append(oslo16_height)
    oslo16_avalanches.append(oslo16._avalanches)
    oslo16_tcs.append(oslo16_tc)
    
arr16 = np.array([oslo16_heights, oslo16_avalanches, oslo16_tcs], dtype=object)

pickle.dump(arr16, open("files/task3a/oslo16.pkl", "wb"))

oslo32_heights = []
oslo32_avalanches = []
oslo32_tcs = []

print("L = ", 32)
for i in range(20):
    print("iteration = ", i)
    oslo32 = oslo(32, [0.5,0.5])
    oslo32_height, oslo32_tc = oslo32.find_total_height(1000000) # extra 1M
    oslo32_heights.append(oslo32_height)
    oslo32_avalanches.append(oslo32._avalanches)
    oslo32_tcs.append(oslo32_tc)
    
arr32 = np.array([oslo32_heights, oslo32_avalanches, oslo32_tcs], dtype=object)

pickle.dump(arr32, open("files/task3a/oslo32.pkl", "wb"))

oslo64_heights = []
oslo64_avalanches = []
oslo64_tcs = []

print("L = ", 64)
for i in range(20):
    print("iteration = ", i)
    oslo64 = oslo(64, [0.5,0.5])
    oslo64_height, oslo64_tc = oslo64.find_total_height(1000000) # extra 1M
    oslo64_heights.append(oslo64_height)
    oslo64_avalanches.append(oslo64._avalanches)
    oslo64_tcs.append(oslo64_tc)

arr64 = np.array([oslo64_heights, oslo64_avalanches, oslo64_tcs], dtype=object)

pickle.dump(arr64, open("files/task3a/oslo64.pkl", "wb"))

oslo128_heights = []
oslo128_avalanches = []
oslo128_tcs = []

print("L = ", 128)
for i in range(20):
    print("iteration = ", i)
    oslo128 = oslo(128, [0.5,0.5])
    oslo128_height, oslo128_tc = oslo128.find_total_height(1000000) # extra 1M
    oslo128_heights.append(oslo128_height)
    oslo128_avalanches.append(oslo128._avalanches)
    oslo128_tcs.append(oslo128_tc)

arr128 = np.array([oslo128_heights, oslo128_avalanches, oslo128_tcs], dtype=object)

pickle.dump(arr128, open("files/task3a/oslo128.pkl", "wb"))

oslo256_heights = []
oslo256_avalanches = []
oslo256_tcs = []

print("L = ", 256)
for i in range(20):
    print("iteration = ", i)
    oslo256 = oslo(256, [0.5,0.5])
    oslo256_height, oslo256_tc = oslo256.find_total_height(1000000) # extra 1M
    oslo256_heights.append(oslo256_height)
    oslo256_avalanches.append(oslo256._avalanches)
    oslo256_tcs.append(oslo256_tc)
    
arr256 = np.array([oslo256_heights, oslo256_avalanches, oslo256_tcs], dtype=object)

pickle.dump(arr256, open("files/task3a/oslo256.pkl", "wb"))

#%% 

oslo512_heights = []
oslo512_avalanches = []
oslo512_tcs = []

print("L = ", 512)
for i in range(20):
    print("iteration = ", i)
    oslo512 = oslo(512, [0.5,0.5])
    oslo512_height, oslo512_tc = oslo512.find_total_height(1000000) # extra 1M
    oslo512_heights.append(oslo512_height)
    oslo512_avalanches.append(oslo512._avalanches)
    oslo512_tcs.append(oslo512_tc)
    
arr512 = np.array([oslo512_heights, oslo512_avalanches, oslo512_tcs], dtype=object)

pickle.dump(arr512, open("files/task3a/oslo512.pkl", "wb"))

#%%

oslo1024_heights = []
oslo1024_avalanches = []
oslo1024_tcs = []

print("L = ", 1024)
for i in range(20):
    print("iteration = ", i)
    oslo1024 = oslo(1024, [0.5,0.5])
    oslo1024_height, oslo1024_tc = oslo1024.find_total_height(1000000) # extra 1M
    oslo1024_heights.append(oslo1024_height)
    oslo1024_avalanches.append(oslo1024._avalanches)
    oslo1024_tcs.append(oslo1024_tc)
    
arr1024 = np.array([oslo1024_heights, oslo1024_avalanches, oslo1024_tcs], dtype=object)

pickle.dump(arr1024, open("files/task3a/oslo1024.pkl", "wb"))
#%%

oslo1024_heights = []
oslo1024_avalanches = []
oslo1024_tcs = []

print("L = ", 1024)
for i in range(20):
    oslo1024 = oslo(1024, [0.5,0.5])
    oslo1024_height, oslo1024_tc = oslo1024.find_total_height(1000000) # extra 1M
    oslo1024_heights.append(oslo1024_height)
    oslo1024_avalanches.append(oslo1024._avalanches)
    oslo1024_tcs.append(oslo1024_tc)
    
arr1024 = np.array([oslo1024_heights, oslo1024_avalanches, oslo1024_tcs], dtype=object)

pickle.dump(arr1024, open("files/task3a/oslo1024.pkl", "wb"))

#%%



#%%

a = pickle.load(open("files/task3a/oslo256.pkl", "rb"))

#%%

oslo512 = oslo(512, [0.5,0.5])
oslo512_height, oslo512_tc = oslo512.find_total_height(5000000) # extra 1M

arr512 = np.array([oslo512_height, oslo512_tc], dtype=object)

pickle.dump(arr512, open("files/task3a/oslo512xx.pkl", "wb"))
"""