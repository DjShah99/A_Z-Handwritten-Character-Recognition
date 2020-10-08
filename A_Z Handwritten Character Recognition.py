# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:09:00 2019

@author: dwijesh
"""
import pandas as pd

character = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

data = pd.read_csv('A_Z Handwritten Data.csv').values
x = data[:,1:]
y = data[:,0]

#Dataset divided into train and test in 75:25 ratio...

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

"""
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf_r=RandomForestClassifier(n_jobs=-1,n_estimators=26)
clf_r.fit(x_train,y_train)

"""
"""
# Matplotlib Plotting

import matplotlib.pyplot as plt
image = x_test[12000]
image.shape = (28,28)
plt.imshow(255-image,cmap='gray')

for i in range(0,26):
    if i == (y_test[12000]-1):
        character_display = character[i+1]

plt.title(character_display)
"""

#DecisionTreeClassifier
"""
from sklearn.tree import DecisionTreeClassifier
clf_d = DecisionTreeClassifier()
clf_d.fit(x_train,y_train)

"""
import matplotlib.pyplot as plt
image = x_test[12000]
image.shape = (28,28)
plt.imshow(255-image,cmap='gray')

for i in range(0,26):
    if i == (y_test[12000]-1):
        character_display = character[i+1]

plt.title(character_display)