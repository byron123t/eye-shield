import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_data = {'r': [], 'g': [], 'b': []}
for file in os.listdir('data/colors'):
    print(file)
    npzfile = np.load(os.path.join('data', 'colors', file), allow_pickle=True)
    for i in npzfile['avg']:
        df_data['r'].append(i[0])
        df_data['g'].append(i[1])
        df_data['b'].append(i[2])
df = pd.DataFrame(df_data)

sns.set(style='ticks')
plt.figure(figsize=(16, 12))
g = sns.displot(data=df, x='g', y='b', kind='hist', binwidth=(1, 1), legend='auto', cbar=True)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'colors.pdf'))
