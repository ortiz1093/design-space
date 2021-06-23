import sklearn
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ansys_file = 'Ansys Data/ANSYS_Results_2021Mar11.csv'

col_names = [
    'WallThickness',
    'Width',
    'Height',
    'FloorThickness',
    'Length',
    'Deformation',
    'Mass'
]

df = pd.read_csv(ansys_file, header=0, names=col_names)

# Correlation matrix
corr = df.corr()

# Visulaize data
sns.set(style='ticks', color_codes=True, font_scale=0.75)
g = sns.pairplot(df, height=2, diag_kind='kde', kind='reg', corner=True)
g.fig.suptitle("Scatter Plot", y=1.08)
# plt.show()

# Separate dataframe into x and y
target_columns = ['Deformation', 'Mass']
X_df = df.drop(target_columns, axis=1)
Y_df = df[target_columns]

# Split data for training and testing


pass
