# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:33:07 2018

@author: deept
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
sns.set_style("white")
%matplotlib inline

x = np.linspace(start = -10, stop=10, num=1000)
y = stats.norm.pdf(x, loc=0, scale=1.5)

plt.plot(x,y)

#reading dataset
df = pd.read_csv("Data/bimodal_example.csv")
df.head(n=5)