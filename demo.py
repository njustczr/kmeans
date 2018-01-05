import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
tips = pd.read_csv('tips.csv')
sns.set(style="ticks")
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")
plt.show()