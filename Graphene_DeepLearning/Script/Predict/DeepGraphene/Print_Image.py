# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 01:07:57 2018

@author: Herman Wu
"""

import numpy as np
import matplotlib.pyplot as plt

x=range(28200)

x=range(100)
plt.figure(figsize=(12,8))
plt.plot(x,a[0:100],label='VCN',color='red',linewidth=2)
plt.plot(x,c[0:100],label='RCN',color='blue',lw=2)
plt.plot(x,b1[0:100],label='CCN',color='green',lw=2)

plt.xlabel('Iteration (512 batch-size)')
plt.ylabel('Loss value')
plt.title('The Loss value during the training-process')

my_x_ticks = np.arange(0, 105, 5)
my_y_ticks = np.arange(0, 2.02, 0.2)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.ylim(0,2.0)
#plt.xlim(-1.2,1.2)
plt.legend(prop={'size': 25}) 
plt.grid() 
plt.savefig("D:\Working_Application\DropBox_File\Dropbox\DeepGraphene\Graphene_DeepLearning\Loss.png")
plt.show()
            
