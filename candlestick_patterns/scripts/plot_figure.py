import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

data = {}
data = pd.DataFrame(data)

cm = 1 / 2.54
plt.figure(figsize=(14.8 * cm, 10.5 * cm))
ax = sns.barplot(data)
ax.set_xlabel("Data set")
ax.set_ylabel("Mean profitability score")
ax.set_title("Mean profitability score, filter by news")
plt.savefig("Figure.pdf")
