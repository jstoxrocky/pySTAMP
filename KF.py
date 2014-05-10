#import modules
from pandas import DataFrame
import pandas as pd
import numpy as np

import pylab
import matplotlib.pyplot as plt

#import original functions
from StructTimeSeries_class import sts
from other_functions import *

#choose file
file_name = '/Users/joestox/Documents/datasets/koopman-commandeur/ukdrivers2.txt'
#file_name = '/Users/joestox/Documents/datasets/koopman-commandeur/norwayfinland.txt'
#file_name = '/Users/joestox/Documents/datasets/Durbin-Koopman-data/nile.txt'
#file_name = '/Users/joestox/Documents/datasets/koopman-commandeur/ukinflation.txt'
#file_name = '/Users/joestox/Documents/datasets/STAMP/rain.txt'


#upload file
try:
    xl = pd.ExcelFile(file_name, na_values=['na', 'NA'])
    sheets = xl.sheet_names
    df = xl.parse(sheets[0])
    print('xls')
except:
    df = DataFrame(pd.read_csv(file_name, na_values=['na', 'NA']))
    print('csv')

#select variable to send to KF algorithm
y = np.log(df['KSI'])
#y = np.log(df['Finnish_fatalities'])
#y = df['vol']
#y = 10000*df['inf']
#y = df['RainFort'][:-9]

#acf, freq = spectral_density(y)
#plt.plot(freq,acf)
#pylab.show()



#select components to model
comp_dic = {"irr": "var", "level": "var", "seasonal": ["fix", 12]}
#comp_dic = {"irr": "var", "level": "var", "slope": "var"}
#comp_dic = {"irr": "var", "level": "fix", "slope": "var"}
#comp_dic = {"irr": "var", "level": "var"}

#create StructTimeSeries object called a model
# sts(<univariate time series>, <dict of components and initial variances>)
model = sts(y,comp_dic,100) #structural time series object
df_filter = model.filter() #dataframe
df_smooth = model.smooth() #dataframes
params = model.params #dict
fit = model.fit #dict (gof=goodness of fit)
tests = model.diagnostics(15) #1st argument is #lags for Box-Ljung Test

#print(df_filter.head(3))
#print(df_smooth.head(3))

print(fit)
print(params)
print(tests)


#plt.plot(df_smooth["fs_level"])

#plt.plot(model.fs_m0)
#plt.plot(df_smooth["fs_level"] + df_smooth["fs_seasonal"])

fig, ax = plt.subplots()
ax.set_color_cycle(['blue', 'green', 'red', 'red'])
plt.plot(y)
plt.plot(df_smooth["fs_level"])
plt.plot(df_smooth["up"])
plt.plot(df_smooth["bottom"])


#plt.plot(df_smooth["fs_seasonal"][:12])
#plt.plot(df_smooth["s_obs_dis_var_level"])

#plt.plot(model.fs_m2)
#
#plt.plot(df_smooth["irr"])
#
#plt.plot(model.fs_m1)
#plt.plot(model.fs_m2)
#print(model.fs_m1)
#print(model.fs_m2)

pylab.show()

