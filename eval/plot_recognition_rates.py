# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4], MUTED[5], MUTED[6], MUTED[7]])

df = pd.DataFrame({'Perspective': ['IU (Protect)', 'SS (GT)', 'SS (Protect)', 'SS (GT 45° Angle)', 'SS (Protect 45° Angle)', 'IU (Protect)', 'SS (GT)', 'SS (Protect)', 'SS (GT 45° Angle)', 'SS (Protect 45° Angle)'], 'Recognition Rate': [89.57, 90.37, 26.94, 100.0, 27.84, 94.05, 62.96, 5.876, 96.82, 14.67], 'Data Type': ['All', 'All', 'All', 'All', 'All', 'Text', 'Text', 'Text', 'Text', 'Text']})
sns.set(font_scale=1.9, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.barplot(data=df, x='Perspective', y='Recognition Rate', hue='Data Type', palette=PAL)
# sns.move_legend(g, 'upper right')
g.set(ylim=[0, 100])
plt.tight_layout()
sns.despine()
# plt.xticks(rotation=15)
plt.savefig(os.path.join('data/plots', 'recognition_rate.pdf'))
