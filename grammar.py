import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
from IPython import get_ipython
python_shell = get_ipython()

if python_shell == None:
    print ('None')
else:
    get_ipython().run_line_magic('matplotlib','inline')

#%matplotlib inline
#%pylab inline

start = datetime.datetime(2017,1,1)
end = datetime.date.today()

apple = web.DataReader("AAPL","yahoo",start,end)
microsoft = web.DataReader("MSFT","yahoo",start,end)
google = web.DataReader("GOOG","yahoo",start,end)
#type(apple)

print (apple.head())

stock = pd.DataFrame({"AAPL": apple["Adj Close"],
                    "MSFT": microsoft["Adj Close"],
                    "GOOG": google["Adj Close"]})

print (stock.head())
#apple["Adj Close"].plot(grid = True)
stock.plot(grid = True)
plt.show()
