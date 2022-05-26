import pandas as pd
import numpy as np
import math

#https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9
#https://medium.datadriveninvestor.com/how-to-detect-support-resistance-levels-and-breakout-using-python-f8b5dac42f21

    
#bullish fractal
def is_support(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]
    cond2 = df['Low'][i] < df['Low'][i+1]
    cond3 = df['Low'][i+1] < df['Low'][i+2]
    cond4 = df['Low'][i-1] < df['Low'][i-2]
    return (cond1 and cond2 and cond3 and cond4)

# bearish fractal
def is_resistance(df,i):  
  cond1 = df['High'][i] > df['High'][i-1]   
  cond2 = df['High'][i] > df['High'][i+1]   
  cond3 = df['High'][i+1] > df['High'][i+2]   
  cond4 = df['High'][i-1] > df['High'][i-2]  
  return (cond1 and cond2 and cond3 and cond4)


# to make sure the new level area does not exist already
def is_far_from_level(value, levels, df):
    ave =  np.mean(df['High'] - df['Low'])
    return np.sum([abs(value-level)<ave for _,level in levels])==0

# a list to store resistance and support levels
def suprezlist(df):
    levels = []
    markers = []
    for i in range(2, df.shape[0] - 2):
        if is_support(df, i):
            l = df['Low'][i]    
            if is_far_from_level(l, levels, df):
                levels.append((i, l))
                markers.append(0)
        elif is_resistance(df, i):
            l = df['High'][i]
            if is_far_from_level(l, levels, df):
                levels.append((i, l))
                markers.append(1)
    return levels, markers

def has_breakout(levels, previous, last):
    try:
        for _, level in levels:
            cond1 = (previous['Open'] < level) 
            cond2 = (last['Open'] > level) and (last['Low'] > level)
        return (cond1 and cond2)
    except Exception as e:
        return False







