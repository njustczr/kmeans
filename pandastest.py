import pandas as pd
import numpy as np
gdp = pd.DataFrame([[11.05,13.08,13.67, 15.05],[12.81, 14.30, 15.07, 16.62],[13.87, 15.52, 16.35, 17.87],[14.80, 16.62,17.36, 18.94]],\
                index=['2012','2013','2014','2015'],columns=['s1', 's2','s3', 's4'])
print gdp
print gdp.sum()
print gdp.sum(axis=1,skipna=False)
print gdp.describe()