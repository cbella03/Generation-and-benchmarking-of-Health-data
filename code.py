"""
Created on Thu Feb 28 15:04:30 2019

@author: Carlo Bellati
"""

#PHASE 1
# MyHealthcare is a wearable device that generates n vital sign records of a person
# we need to import some modules first
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
import numpy as np
import math

def MyHealthcare(n):
    rand.seed(404)
    df = {'ts': [i for i in range(n)],
        'temp': [rand.randint(36, 39) for i in range(n)],
            'hr':[rand.randint(55, 100) for i in range(n)], 
             'pulse':[rand.randint(55, 100) for i in range(n)], 
             'bloodpr':[rand.choice([120, 121]) for i in range(n)], 
             'resrate': [rand.randint(11, 17) for i in range(n)], 
             'oxsat':[rand.randint(93, 100) for i in range(n)], 
             'ph': [round(7.1 + (rand.random()*0.5),1) for i in range(n)]}
    return df



#Example:
mydf = MyHealthcare(50)

# PHASE 2
#a)
# abnSignAnalytics is a function that returns the number of abnormal values, all abnormal records (and their relative timestamp) of a chosen
# variable between pulse and blood pressure.

def abnSignAnalytics(df, choice):
    d = {'pulse' : [60, 99], 'bloodpr' : [120, 120]}
    count = 0
    ls = []
    for record in range(len(df[choice])):
        if df[choice][record] > d[choice][1] or df[choice][record] < d[choice][0]:
            ls.append([df['ts'][record], df[choice][record]])
            count += 1
    return [choice, count, ls]

    
#b)
# frequencyAnalytics returns a list of the frequencies for pulse rate values in a given dataset

# the function needs to call a sorting algorithm. I have opted for a merge sort algorithm:
    
def mergesort(alist):
    if len(alist) > 1:
        mid = len(alist)//2
        lhalf = alist[:mid]
        rhalf = alist[mid:]
        mergesort(lhalf)
        mergesort(rhalf)
        i = 0
        j = 0
        k = 0
        while i < len(lhalf) and j < len(rhalf):
            if lhalf[i] < rhalf[j]:
                alist[k] = lhalf[i]
                i = i+1
            else:
                alist[k] = rhalf[j]
                j = j+1
            k = k+1
        while i < len(lhalf):
            alist[k] = lhalf[i]
            i = i+1
            k = k+1
        while j < len(rhalf):
            alist[k] = rhalf[j]
            j = j+1
            k = k+1

def frequencyAnalytics(df):
    values = list(df['pulse'])
    alist = []
    results = []
    count = 0
    for i in values:
        if i not in alist:
            alist.append(i)
    mergesort(alist)
    for i in alist:
        for j in values:
            if j==i:
                count+= 1
        results.append([i,count])
        count = 0
    return results


#alternatively I can implement a similar function by taking advantage of the Pandas functionalities.

def frequencyAnalytics2(df):
    mydf =pd.DataFrame(df) #this is necessary if df is a dictionary but not a pd.dataframe
    #I count all frequencies of pulse in df by using the value_counts method'
    #I obtain a new dataframe where all values are in the index and their frequencies in the 'pulse' column
    val_and_freq = pd.DataFrame(mydf['pulse'].value_counts())
    #however I want the pulse values to be treated as a series, not as an index. So I shift them into a new column
    #then I rename the columns to be more readable
    val_and_freq['pulse_vals'] = val_and_freq.index
    val_and_freq.rename(columns={'pulse':'frequencies'}, inplace=True)
    # I want to change the order of columns so that we have pulse values first and frequencies at the right
    columnsName = list(val_and_freq.columns)
    F, P = columnsName.index('frequencies'), columnsName.index('pulse_vals')
    columnsName[F], columnsName[P] = columnsName[P],columnsName[F]
    val_and_freq = val_and_freq[columnsName]
    #I sort the pulse values in ascending order
    val_and_freq = val_and_freq.sort_values(['pulse_vals'])
    #I use the iterrows method to transform the output into a list of lists
    results = []
    for row in val_and_freq.iterrows():
        index, data = row
        results.append(data.tolist())
    return results
    


#let's now plot the pulse values obtained from the first function:
mydf = MyHealthcare(50)
all_values = frequencyAnalytics(mydf)  #all pulse values and their relative frequency
# in order to plot the data in a histogram, I need to consider all data decomposed
all_values_per_fr = []
for i in all_values:
    for j in range(i[1]):
        all_values_per_fr.append(i[0])
        
plt.hist(all_values_per_fr, label='pulse values')
plt.xlabel('Pulse values')
plt.ylabel('Frequency')
plt.title('frequency of pulse values in mysample')
plt.legend()
plt.xticks(list(range(55,100, 5)))
plt.show()
        
# I can also plot the data in a bar chart (1 bar per each value).
# in the barchart I distinguish between normal and abnormal values  
plt.bar([j[0] for j in all_values] [2:23], [i[1] for i in all_values][2:23], label='normal pulse values')
plt.bar([[j[0] for j in all_values][i] for i in [0,1]], [[j[1] for j in all_values][i] for i in [0,1]], label='abnormal pulse values', color='r')
plt.xlabel('Pulse values')
plt.ylabel('Frequency')
plt.title('frequency of pulse values in mysample')
plt.legend()
plt.show()



#c) let's plot the results of the functions

abn_pulse = abnSignAnalytics(mydf, 'pulse') # result: a list which contains:name of variable (pulse); number of occurrances, all found values with relative timestamp
abnormal_pulse_vals = [i[1] for i in abn_pulse[2]] #the list of all pulse abnormal values (time stamps are not selected)
#I plot the abnormal values in a histogram. I choose 45 as a number of bins (45 are all possible values in the interval)
plt.hist(abnormal_pulse_vals, bins=45, label='abnormal values')
plt.xlim(54, 101)
plt.xticks([55, 57, 59, 100])


#PHASE 3
#a) 
 
#the first implementation 

def HealthAnalyzer(key, df):
    results = []
    for i in range(len(df['ts'])):
        if df['pulse'][i] == key:
            results.append([df['ts'][i], 
                           df['temp'][i], 
                           df['hr'][i], 
                            df['pulse'][i], 
                             df['bloodpr'][i], 
                              df['resrate'][i], 
                               df['oxsat'][i], df['ph'][i]])
    return results

#the second implementation 
    
# customized quick sort

def quickSort(alist, alist2):
   quickSort2(alist,alist2,0,len(alist)-1)
#alist is the list of the values to sort (here 'pulse' values)
# alist2 is the list of timestamps
def quickSort2(alist,alist2, first,last):
    if first<last:
        splitpoint = partition(alist, alist2, first,last)
        quickSort2(alist,alist2,first,splitpoint-1)
        quickSort2(alist,alist2, splitpoint+1,last)

def partition(alist,alist2,first,last):
   pivotvalue = alist[first]
   leftmark = first+1
   rightmark = last
   done = False
   while not done:
       while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
           leftmark = leftmark + 1
       while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1
       if rightmark < leftmark:
           done = True
       else:
           alist[leftmark], alist[rightmark] = alist[rightmark], alist[leftmark]
           alist2[leftmark], alist2[rightmark] = alist2[rightmark], alist2[leftmark]
           # when elements have to be swapped, the swapping takes place in both lists
   alist[first], alist[rightmark] = alist[rightmark], alist[first]
   alist2[first], alist2[rightmark] = alist2[rightmark], alist2[first]
   return rightmark

#the downside of this function, apart from its complexity, is that the final results
#are not displayed in a sorted fashion.
                
#the second implementation

def HealthAnalyzer2(df, item):
    ts = list(df['ts']) 
    pulse = list(df['pulse'])
    quickSort(pulse, ts)
    first = 0
    last = len(ts)-1
    ts_list = []
    results = []
    while first<=last:
        mid = (first+last)//2
        if pulse[mid] == item:
            if mid-1 >= first: 
                if pulse[mid-1] == item:
                    minus = mid-1
                    while pulse[minus] == item:
                        ts_list.append(ts[minus])
                        minus = minus-1
                        if minus< first:
                            break
            ts_list.append(ts[mid])
            if mid+1 <= last:
                if pulse[mid+1] == item:
                    plus = mid+1
                    while pulse[plus] == item:
                        ts_list.append(ts[plus])
                        plus = plus+1
                        if plus > last:
                            break
            break
        else:
            if item < pulse[mid]:
                last = mid-1
            else:
                first = mid+1
    for i in ts_list:
        results.append([mydf['ts'][i], mydf['temp'][i], mydf['hr'][i], mydf['pulse'][i], mydf['bloodpr'][i], mydf['resrate'][i], mydf['oxsat'][i], mydf['ph'][i]])
    return results

#the third implementation
    
def HealthAnalyzer3(df, item):
    #copy the dictionary into a dataframe
    df22 = pd.DataFrame(mydf)
    # use the .loc method to find all matching values
    found = df22.loc[df22['pulse'] == item]
    # if I want, I can save each row into a list and return a list of all these lists
    # as in the function seen before, by using the iterrows method
    results = []
    for row in found.iterrows():
        index, data = row
        results.append(data.tolist())
    return results
        


    
#c) let's plot the heart rate values for records of mydf having a pulse rate of 56
mydf = MyHealthcare(1000)    

myexample = HealthAnalyzer2(mydf, 56)
plt.hist([i[2] for i in myexample], label='heart rate values')
plt.xlabel('Heart rate')
plt.ylabel('Frequency')
plt.title('heart rate values when pulse rate is 56')
plt.legend()
plt.xticks(list(range(55, 105, 5)))
plt.show()


#PHASE 4
#a)

#I calculate the running time of MyHealthcare with the given inputs
import time
size = [1000, 5000, 7500, 10000]
elapsed_times = []
for i in size:
    start = time.time()
    MyHealthcare(i)
    elapsed = time.time()-start
    elapsed_times.append(elapsed)
 
#before plotting, I want to add some useful lines for comparison

last_value = (10000*math.log2(10000)) #this is the value of the rightmost dot I want to plot on the n*log(n) line.
dividend = last_value/0.06 # 0.06 is approximately the upper rightmost value on my plot

# I want to draw 3 lines: n, log(n) and n log(n)
n_logn_vals_ph4 = []
logn_vals_ph4 = []
n_vals_ph4= []
for i in size:
    a = (math.log2(i))*i
    b = math.log2(i)
    c = i/dividend
    n_logn_vals_ph4.append(a)
    logn_vals_ph4.append(b)
    n_vals_ph4.append(c)
n_logn_vals_ph4 = list(np.array(n_logn_vals_ph4)/dividend)
logn_vals_ph4 = list(np.array(logn_vals_ph4)/dividend)    
    
# plotting the data
plt.plot(size, elapsed_times, label='MyHealthcare', marker='o')
plt.title('running time of MyHealthcare function')
plt.xlabel('number of records')
plt.ylabel('running time')

plt.xticks(size)
plt.plot(size, n_logn_vals_ph4, linestyle='--', label='n(logn)')
plt.plot(size, logn_vals_ph4, linestyle='--', label='logn')
plt.plot(size, n_vals_ph4, linestyle='--', label='n')

plt.legend()
plt.show()

# I want also to benchmark all implemented functions and plot them together
# first I calculate the running time of all functions

size = [50000, 100000, 150000, 200000, 300000, 500000]
time_of_frAn = []
time_of_frAn2 = []
for i in size:
    mydf = MyHealthcare(i)
    start = time.time()
    frequencyAnalytics(mydf)
    elapsed = time.time()-start
    time_of_frAn.append(elapsed)
    start = time.time()
    frequencyAnalytics2(mydf)
    elapsed = time.time()-start
    time_of_frAn2.append(elapsed)

time_of_abnSign = []

for i in size:
    mydf = MyHealthcare(i)
    start = time.time()
    abnSignAnalytics(mydf, 'pulse')
    elapsed = time.time()-start
    time_of_abnSign.append(elapsed)

time_of_HealthAn = []

for i in size:
    mydf = MyHealthcare(i)
    start = time.time()
    HealthAnalyzer(98, mydf)
    elapsed = time.time()-start
    time_of_HealthAn.append(elapsed)

time_of_HealthAn2 = []

for i in [5000, 7000, 10000, 15000, 20000]:
    mydf = MyHealthcare(i)
    start = time.time()
    HealthAnalyzer2(mydf, 98)
    elapsed = time.time()-start
    time_of_HealthAn2.append(elapsed)

time_of_HealthAn3 = []

for i in size:
    mydf = MyHealthcare(i)
    start = time.time()
    HealthAnalyzer3(mydf, 98)
    elapsed = time.time()-start
    time_of_HealthAn3.append(elapsed)



#I redraw the lines for the new x values
last_value2 = (500000*math.log(500000))  #this is the value of the rightmost dot I want to plot on the n*log(n) line.
dividend = (500000*math.log(500000))/0.9 #0.9 is approximately the upper rightmost value on my plot
interval = [i for i in range(size[0], size[len(size)-1], 1500)]

n_logn_vals = [] #n(log(n)) line
logn_vals = [] #log(n) line
n_vals= [] #n line
for i in interval:
    a = (math.log2(i))*i  
    b = math.log2(i)   
    c = i/dividend  
    n_logn_vals.append(a)
    logn_vals.append(b)
    n_vals.append(c)
    
# i divide the results by a dividend
n_logn_vals = list(np.array(n_logn_vals)/dividend)
logn_vals = list(np.array(logn_vals)/dividend)


#I plot the lines 
plt.plot(interval, n_logn_vals, linestyle='--', label='n(logn)')
plt.plot(interval, logn_vals, linestyle='--', label='logn')
plt.plot(interval, n_vals, linestyle='--', label='n')
plt.legend()

# I plot the functions
plt.plot(size, time_of_frAn, label='frequencyAnalytics', linestyle='--', marker='o')
plt.legend()
plt.plot(size, time_of_frAn2, label='frequencyAnalytics2', linestyle='--', marker='o')
plt.legend()
plt.plot(size, time_of_abnSign, label='abnSignAnalytics', linestyle='--', marker='o')
plt.legend()
plt.plot(size, time_of_HealthAn, label='HealthAnalyzer', linestyle='--', marker='o')
plt.legend()
plt.plot([5000, 7000, 10000, 15000, 20000], time_of_HealthAn2, label='HealthAnalyzer2', linestyle='--', marker='o')
plt.legend()
plt.plot(size, time_of_HealthAn3, label='HealthAnalyzer3', linestyle='--', marker='o')
plt.legend()
plt.yticks(time_of_frAn2, time_of_abnSign, time_of_frAn, time_of_HealthAn, time_of_HealthAn2, time_of_HealthAn3)
plt.xticks(size)
plt.show()

    
