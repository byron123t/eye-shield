# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import csv
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
CONVERT = {'strongly disagree': 0, 'somewhat disagree': 1, 'neither agree nor disagree': 2, 'somewhat agree': 3, 'strongly agree': 4}
INVERT = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}


infile = open('data/csvs/inperson.csv', 'r')
reader = csv.reader(infile)
index_dict = {}
user_data = {'Frequently': [], 'Complex': [], 'Easy to Use': [], 'Tech Support': [], 'Well Integrated': [], 'Inconsistency': [], 'Learn Quickly': [], 'Cumbersome': [], 'Confident': [], 'Learn a Lot': []}
privacy_dict = {}
demographics_dict = {}
attention_checks = {}
all_cronbach_data = {}

for i, user in enumerate(reader):
    if i == 0:
        continue
    if i == 1:
        k = 0
        for j, column in enumerate(user):
            column_id = '{} {}'.format(j, column)
            print(column_id)
            index_dict[j] = column_id
            if j < 17: # User data
                pass
            if j >= 17 and j < 29: # Timings
                pass
            if j >= 29 and j < 596: # Control scenarios
                if 'Timing - ' in column:
                    pass

    if i == 2:
        continue
    if i > 2:
        demographics_dict[i] = {}
        for j, column in enumerate(user):
            column_id = index_dict[j]
            col = column.strip().lower()
            if j >= 19 and j <= 28: # shoulder surf questionnaire
                if 'I think that I would like to use this system frequently' in column_id:
                    user_data['Frequently'].append(CONVERT[col])
                elif 'I found the system unnecessarily complex' in column_id:
                    user_data['Complex'].append(INVERT[CONVERT[col]])
                elif 'I thought the system was easy to use' in column_id:
                    user_data['Easy to Use'].append(CONVERT[col])
                elif 'I think that I would need the support of a technical person' in column_id:
                    user_data['Tech Support'].append(INVERT[CONVERT[col]])
                elif 'I found the various functions in this system were well integrated' in column_id:
                    user_data['Well Integrated'].append(CONVERT[col])
                elif 'I thought there was too much inconsistency in this system' in column_id:
                    user_data['Inconsistency'].append(INVERT[CONVERT[col]])
                elif 'I would imagine that most people would learn to use this system very quickly' in column_id:
                    user_data['Learn Quickly'].append(CONVERT[col])
                elif 'I found the system very cumbersome to use' in column_id:
                    user_data['Cumbersome'].append(INVERT[CONVERT[col]])
                elif 'I felt very confident using the system' in column_id:
                    user_data['Confident'].append(CONVERT[col])
                elif 'I needed to learn a lot of things before I could get going with this system' in column_id:
                    user_data['Learn a Lot'].append(INVERT[CONVERT[col]])
            if j >= 33 and j <= 34: # Demographics
                if 'Describe your gender' in column_id:
                    demographics_dict[i]['gender'] = col
                elif 'What is your age in years' in column_id:
                    if int(col) > 200:
                        age = 2022 - int(col)
                    else:
                        age = int(col)
                    demographics_dict[i]['age'] = age

df_data = {'Question': [], 'Score': []}
for question, answers in user_data.items():
    for j in answers:
        df_data['Question'].append(question)
        df_data['Score'].append(j)

print(df_data['Score'])

df = pd.DataFrame(df_data)
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=2.2, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.boxplot(data=df, x='Question', y='Score', palette=PAL)
g.set(ylim=[0, 4])
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'inperson_sus.pdf'))

male = 0
female = 0
ages = []
for key, val in demographics_dict.items():
    if 'gender' in val:
        if val['gender'] == 'male':
            male += 1
        elif val['gender'] == 'female':
            female += 1
        ages.append(val['age'])
ages = np.array(ages)
print(np.mean(ages), np.std(ages), max(ages), min(ages))
print(male, female)
