# import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = np.array([['8.87242722', '11111524.8', '0.25', '1.80092460', '45', 'D'],
                 ['7.24084124', '10621766.0', '0.125', '2.2857515', '48', 'A'],
                 ['4.83987333', '9085858.0', '0.0625', '2.6577979', '44', 'C']])
cols = ['Side Length', "Young's Modulus", 'Microsteps', 'Pitch', '# Teeth', 'Motor']

# data = np.array([['8.87242722', '11111524.8', '0.25', '1.80092460', '45'],
#                  ['7.24084124', '10621766.0', '0.125', '2.2857515', '48'],
#                  ['4.83987333', '9085858.0', '0.0625', '2.6577979', '44']])
# cols = ['Side Length', "Young's Modulus", 'Microsteps', 'Pitch', '# Teeth']

df = pd.DataFrame(data, columns=cols)

cats = df.select_dtypes(exclude=[int, float]).astype('category')
for col in cats.columns:
    codes = cats[col].cat.codes
    df[col] = codes

# fig = px.scatter_matrix(df)
# fig.update_traces(diagonal_visible=False, showupperhalf=False)
# fig.update_xaxes(tickformat=".3s", showgrid=False)
# fig.update_yaxes(tickformat=".3s", showgrid=False)

# fig.show()

g = sns.pairplot(df, diag_kind="kde", corner=True)
g.axes[-1].add_patch(patches.Circle((24, 25), 2,alpha=0.3, facecolor="green", edgecolor="black", linewidth=1, linestyle='solid'))
# g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.show()
