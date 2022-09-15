# -*- coding: utf-8 -*-

import csv
import oslo as o
import numpy as np

#%% Save the data for all system sizes L=4,8,16,32,64,128,256,512; iterate 1M times

oslo4_1M = o.oslo(4, [0.5,0.5])
oslo4_1M_total = oslo4_1M.find_total_height(1000000)

oslo8_1M = o.oslo(8, [0.5,0.5])
oslo8_1M_total = oslo8_1M.find_total_height(1000000)

oslo16_1M = o.oslo(16, [0.5,0.5])
oslo16_1M_total = oslo16_1M.find_total_height(1000000)

oslo32_1M = o.oslo(32, [0.5,0.5])
oslo32_1M_total = oslo32_1M.find_total_height(1000000)

oslo64_1M = o.oslo(64, [0.5,0.5])
oslo64_1M_total = oslo64_1M.find_total_height(1000000)

oslo128_1M = o.oslo(128, [0.5,0.5])
oslo128_1M_total = oslo128_1M.find_total_height(1000000)

oslo256_1M = o.oslo(256, [0.5,0.5])
oslo256_1M_total = oslo256_1M.find_total_height(300000)

oslo512_1M = o.oslo(512, [0.5,0.5])
oslo512_1M_total = oslo512_1M.find_total_height(1000000)

#%%

t = np.arange(1, 1e6+1)

#%% Write files in folder task2a

with open('files/task2a/oslo4.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo4_1M_total) # row 2: total height

with open('files/task2a/oslo8.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo8_1M_total) # row 2: total height

with open('files/task2a/oslo16.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo16_1M_total) # row 2: total height    

with open('files/task2a/oslo32.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo32_1M_total) # row 2: total height
 
with open('files/task2a/oslo64.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo64_1M_total) # row 2: total height

with open('files/task2a/oslo128.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo128_1M_total) # row 2: total height

with open('files/task2a/oslo256.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo256_1M_total) # row 2: total height

with open('files/task2a/oslo512.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    writer.writerow(oslo512_1M_total) # row 2: total height
    
#%% Run 10 times for each system size

oslo4_totals = []
oslo4_tcs = []

for i in range(10):
    oslo4 = o.oslo(4, [0.5,0.5])
    oslo4_total, oslo4_tc = oslo4.find_total_height(100000) # extra 100k
    oslo4_totals.append(oslo4_total)
    oslo4_tcs.append(oslo4_tc)

oslo8_totals = []
oslo8_tcs = []

for i in range(10):
    oslo8 = o.oslo(8, [0.5,0.5])
    oslo8_total, oslo8_tc = oslo8.find_total_height(100000) # extra 100k
    oslo8_totals.append(oslo8_total)
    oslo8_tcs.append(oslo8_tc)

oslo16_totals = []
oslo16_tcs = []

for i in range(10):
    oslo16 = o.oslo(16, [0.5,0.5])
    oslo16_total, oslo16_tc = oslo16.find_total_height(100000) # extra 100k
    oslo16_totals.append(oslo16_total)
    oslo16_tcs.append(oslo16_tc)

oslo32_totals = []
oslo32_tcs = []

for i in range(10):
    oslo = o.oslo(32, [0.5,0.5])
    oslo_total, oslo_tc = oslo.find_total_height(100000) # extra 100k
    oslo32_totals.append(oslo_total)
    oslo32_tcs.append(oslo_tc)
    
oslo64_totals = []
oslo64_tcs = []

for i in range(10):
    oslo = o.oslo(64, [0.5,0.5])
    oslo_total, oslo_tc = oslo.find_total_height(100000) # extra 100k
    oslo64_totals.append(oslo_total)
    oslo64_tcs.append(oslo_tc)

oslo128_totals = []
oslo128_tcs = []

for i in range(10):
    oslo = o.oslo(128, [0.5,0.5])
    oslo_total, oslo_tc = oslo.find_total_height(100000) # extra 100k
    oslo128_totals.append(oslo_total)
    oslo128_tcs.append(oslo_tc)

oslo256_totals = []
oslo256_tcs = []

for i in range(10):
    oslo = o.oslo(256, [0.5,0.5])
    oslo_total, oslo_tc = oslo.find_total_height(100000) # extra 100k
    oslo256_totals.append(oslo_total)
    oslo256_tcs.append(oslo_tc)
        
oslo512_totals = []
oslo512_tcs = []

for i in range(10):
    oslo = o.oslo(512, [0.5,0.5])
    oslo_total, oslo_tc = oslo.find_total_height(100000) # extra 100k
    oslo512_totals.append(oslo_total)
    oslo512_tcs.append(oslo_tc)
    print("finish", i+1, "iterations")

#%%

t = np.arange(1, 1e6+1)

#%% Write files in folder task2b

with open('files/task2b/oslo4.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo4_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo4_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo8.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo8_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo8_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo16.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo16_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo16_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo32.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo32_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo32_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo64.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo64_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo64_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo128.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo128_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo128_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo256.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo256_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo256_tcs) # row 12: 10 x cross-over time

with open('files/task2b/oslo512.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo512_totals:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo512_tcs) # row 12: 10 x cross-over time
    
#%% Rerun again to do 1M extra in steady state again iterate 10 times

t = np.arange(1, 1e9+1)

oslo4_heights = []
oslo4_avalanches = []
oslo4_tcs = []

print("L = ", 4)
for i in range(10):
    print("iteration = ", i)
    oslo4 = o.oslo(4, [0.5,0.5])
    oslo4_height, oslo4_tc = oslo4.find_total_height(100000000) # extra 1M
    oslo4_heights.append(oslo4_height)
    oslo4_avalanches.append(oslo4._avalanches)
    oslo4_tcs.append(oslo4_tc)

oslo8_heights = []
oslo8_avalanches = []
oslo8_tcs = []

print("L = ", 8)
for i in range(30):
    oslo8 = o.oslo(8, [0.5,0.5])
    oslo8_height, oslo8_tc = oslo8.find_total_height(100000000) # extra 1M
    oslo8_heights.append(oslo8_height)
    oslo8_avalanches.append(oslo8._avalanches)
    oslo8_tcs.append(oslo8_tc)

oslo16_heights = []
oslo16_avalanches = []
oslo16_tcs = []

print("L = ", 16)
for i in range(30):
    oslo16 = o.oslo(16, [0.5,0.5])
    oslo16_height, oslo16_tc = oslo16.find_total_height(100000000) # extra 1M
    oslo16_heights.append(oslo16_height)
    oslo16_avalanches.append(oslo16._avalanches)
    oslo16_tcs.append(oslo16_tc)

oslo32_heights = []
oslo32_avalanches = []
oslo32_tcs = []

print("L = ", 32)
for i in range(30):
    oslo32 = o.oslo(32, [0.5,0.5])
    oslo32_height, oslo32_tc = oslo32.find_total_height(100000000) # extra 1M
    oslo32_heights.append(oslo32_height)
    oslo32_avalanches.append(oslo32._avalanches)
    oslo32_tcs.append(oslo32_tc)
    
oslo64_heights = []
oslo64_avalanches = []
oslo64_tcs = []

print("L = ", 64)
for i in range(30):
    oslo64 = o.oslo(64, [0.5,0.5])
    oslo64_height, oslo64_tc = oslo64.find_total_height(100000000) # extra 1M
    oslo64_heights.append(oslo64_height)
    oslo64_avalanches.append(oslo64._avalanches)
    oslo64_tcs.append(oslo64_tc)

oslo128_heights = []
oslo128_avalanches = []
oslo128_tcs = []

print("L = ", 128)
for i in range(30):
    oslo128 = o.oslo(128, [0.5,0.5])
    oslo128_height, oslo128_tc = oslo128.find_total_height(100000000) # extra 1M
    oslo128_heights.append(oslo128_height)
    oslo128_avalanches.append(oslo128._avalanches)
    oslo128_tcs.append(oslo128_tc)

oslo256_heights = []
oslo256_avalanches = []
oslo256_tcs = []

print("L = ", 256)
for i in range(30):
    oslo256 = o.oslo(256, [0.5,0.5])
    oslo256_height, oslo256_tc = oslo256.find_total_height(100000000) # extra 1M
    oslo256_heights.append(oslo256_height)
    oslo256_avalanches.append(oslo256._avalanches)
    oslo256_tcs.append(oslo256_tc)

oslo512_heights = []
oslo512_avalanches = []
oslo512_tcs = []

print("L = ", 512)
for i in range(10):
    oslo512 = o.oslo(512, [0.5,0.5])
    oslo512_height, oslo512_tc = oslo512.find_total_height(100000000) # extra 1M
    oslo512_heights.append(oslo512_height)
    oslo512_avalanches.append(oslo512._avalanches)
    oslo512_tcs.append(oslo512_tc)
    
#%%

import pickle

arr = np.array([[1, 2], [3, 4]])
pickle.dump(arr, open("files/task3a/oslo4.pkl", "w"))

#print (pickle.load(open("files/task3a/oslo4.pkl")))



#%%

with open('files/task3a/oslo4.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo4_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo4_tcs) # row 12: 10 x cross-over time
    for i in oslo4_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo8.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo8_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo8_tcs) # row 12: 10 x cross-over time
    for i in oslo4_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo16.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo16_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo16_tcs) # row 12: 10 x cross-over time
    for i in oslo16_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo32.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo32_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo32_tcs) # row 12: 10 x cross-over time
    for i in oslo32_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo64.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo64_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo64_tcs) # row 12: 10 x cross-over time
    for i in oslo64_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo128.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo128_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo128_tcs) # row 12: 10 x cross-over time
    for i in oslo128_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo256.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo256_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo256_tcs) # row 12: 10 x cross-over time
    for i in oslo256_avalanches:
        writer.writerow(i) # rows 13-22: avalanche

with open('files/task3a/oslo512.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(t) # row 1: number of grains added
    for i in oslo512_heights:
        writer.writerow(i) # rows 2-11: total height
    writer.writerow(oslo512_tcs) # row 12: 10 x cross-over time
    for i in oslo512_avalanches:
        writer.writerow(i) # rows 13-22: avalanche
        
