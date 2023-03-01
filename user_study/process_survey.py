# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import csv
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Note: Will not run without changes to code, due to deletion of IP address, location, and MTurk ID data columns from csv (to preserve anonymity of participants).

MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
CONVERT = {'strongly disagree': 0, 'somewhat disagree': 1, 'neither agree nor disagree': 2, 'somewhat agree': 3, 'strongly agree': 4}
INVERT = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}

infile = open('data/csvs/mturk.csv', 'r')
reader = csv.reader(infile)
index_dict = {}
user_data = {}
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
        user_data[i] = {'duration': 0, 'responses': {}, 'mturk_id': 0}
        privacy_dict[i] = {}
        demographics_dict[i] = {}
        for j, column in enumerate(user):
            column_id = index_dict[j]
            col = column.strip().lower()
            if j == 33 or j == 133 or j == 253 or j == 357 or j == 457 or j == 577 or j == 681 or j == 711 or j == 747:
                k = 0
            if 'The image above' in column_id or 'Please type the' in column_id or 'The video above' in column_id:
                k += 1
            if len(column) == 0:
                continue
            if j < 17: # User data
                if j == 5:
                    user_data[i]['duration'] = int(column)
                    user_data[i]['responses'] = {'image_pis': [], 'image_pii': [], 'image_ois': [], 'image_quality': [],
                                                 'text_pis': [], 'text_pii': [], 'text_ois': [], 'text_quality': [],
                                                 'video_pis': [], 'video_pii': [], 'video_ois': [], 'video_quality': [],
                                                 'image_pis_id': [], 'image_pii_id': [], 'image_ois_id': [],
                                                 'text_pis_id': [], 'text_pii_id': [], 'text_ois_id': [],
                                                 'video_pis_id': [], 'video_pii_id': [], 'video_ois_id': [],
                                                 'image_pis_timing': [], 'image_pii_timing': [], 'image_ois_timing': [],
                                                 'text_pis_timing': [], 'text_pii_timing': [], 'text_ois_timing': [],
                                                 'video_pis_timing': [], 'video_pii_timing': [], 'video_ois_timing': []}
            if j >= 26 and j < 29: # shoulder surf questionnaire
                if 'It bothers me when others peek' in column_id:
                    privacy_dict[i]['bothers'] = CONVERT[col]
                elif 'I peek at others' in column_id:
                    privacy_dict[i]['peek'] = CONVERT[col]
                elif 'I feel uncomfortable using my mobile' in column_id:
                    privacy_dict[i]['uncomfortable'] = CONVERT[col]
            if j >= 37 and j <= 136: # Protected images shouldersurfer
                if 'The image above is best described as a:' in column_id:
                    user_data[i]['responses']['image_pis'].append(col)
                    user_data[i]['responses']['image_pis_id'].append(k)
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['image_pis_timing'].append(float(col))
            if j >= 137 and j <= 256: # Protected images intendeduser
                if 'The image above is best described as a:' in column_id:
                    user_data[i]['responses']['image_pii'].append(col)
                    user_data[i]['responses']['image_pii_id'].append(k)
                elif 'I would be fine with looking' in column_id:
                    user_data[i]['responses']['image_quality'].append(CONVERT[col])
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['image_pii_timing'].append(float(col))
            if j >= 257 and j <= 356: # Original images shouldersurfer
                if 'The image above is best described as a:' in column_id:
                    user_data[i]['responses']['image_ois'].append(col)
                    user_data[i]['responses']['image_ois_id'].append(k)
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['image_ois_timing'].append(float(col))
            if j >= 361 and j <= 460: # Protected text intendeduser
                if 'Please type the' in column_id:
                    user_data[i]['responses']['text_pis'].append(col)
                    user_data[i]['responses']['text_pis_id'].append(k)
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['text_pis_timing'].append(float(col))
            if j >= 461 and j <= 580: # Protected text intendeduser
                if 'Please type the' in column_id:
                    user_data[i]['responses']['text_pii'].append(col)
                    user_data[i]['responses']['text_pii_id'].append(k)
                elif 'I would be fine with looking' in column_id:
                    user_data[i]['responses']['text_quality'].append(CONVERT[col])
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['text_pii_timing'].append(float(col))
            if j >= 581 and j <= 680: # Original text shouldersurfer
                if 'Please type the' in column_id:
                    user_data[i]['responses']['text_ois'].append(col)
                    user_data[i]['responses']['text_ois_id'].append(k)
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['text_ois_timing'].append(float(col))
            if j >= 685 and j <= 714: # Protected videos intendeduser
                if 'The video above is best described as a' in column_id:
                    user_data[i]['responses']['video_pis'].append(col)
                    user_data[i]['responses']['video_pis_id'].append(k)
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['video_pis_timing'].append(float(col))
            if j >= 715 and j <= 750: # Protected videos intendeduser
                if 'The video above is best described as a' in column_id:
                    user_data[i]['responses']['video_pii'].append(col)
                    user_data[i]['responses']['video_pii_id'].append(k)
                elif 'I would be fine with looking' in column_id:
                    user_data[i]['responses']['video_quality'].append(CONVERT[col])
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['video_pii_timing'].append(float(col))
            if j >= 751 and j <= 780: # Original videos shouldersurfer
                if 'The video above is best described as a' in column_id:
                    user_data[i]['responses']['video_ois'].append(col)
                    user_data[i]['responses']['video_ois_id'].append(k)
                elif 'Timing - Page Submit' in column_id:
                    user_data[i]['responses']['video_ois_timing'].append(float(col))
            if j >= 781 and j <= 782: # Demographics
                if 'Describe your gender' in column_id:
                    demographics_dict[i]['gender'] = col
                elif 'What is your age in years' in column_id:
                    if int(col) > 200:
                        age = 2022 - int(col)
                    else:
                        age = int(col)
                    demographics_dict[i]['age'] = age
            if j == 787: # MTurk ID
                user_data[i]['mturk_id'] = column

temp = user_data.copy()
durations = []
for user, val in user_data.items():
    durations.append(val['duration'])
    if val['duration'] < 250:
        print(user, val['mturk_id'], val['duration'])
        del privacy_dict[user]
        del demographics_dict[user]
        del temp[user]
user_data = temp
print(np.mean(durations))

def joint_sort(list1, list2):
    zipped_lists = zip(list1, list2)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    list1, list2 = [list(t) for t in tuples]
    return list1, list2


responses_dict = {}
ids_dict = {}
qualities_dict = {}
timings_dict = {}

for user, user_dict in user_data.items():
    for block, block_dict in user_dict['responses'].items():
        if '_id' in block:
            blockname = block.replace('_id', '')
            if blockname not in ids_dict:
                ids_dict[blockname] = []
            ids_dict[blockname].extend(block_dict)
        elif '_quality' in block:
            if block not in qualities_dict:
                qualities_dict[block] = []
            qualities_dict[block].extend(block_dict)
        elif '_timing' in block:
            if block not in timings_dict:
                timings_dict[block] = []
            timings_dict[block].extend(block_dict)
        else:
            if block not in responses_dict:
                responses_dict[block] = []
            responses_dict[block].extend(block_dict)

print(timings_dict)
print(responses_dict)

for key, responses in responses_dict.items():
    ids_dict[key], responses_dict[key] = joint_sort(ids_dict[key], responses)

df_data = {'Score': [], 'Question': []}

for key, val in privacy_dict.items():
    for key1, val1 in val.items():
        df_data['Score'].append(val1)
        df_data['Question'].append(key1.capitalize())

df = pd.DataFrame(df_data)
temp_df = df.where(df['Question'] == 'Bothers').dropna()
print(np.mean(temp_df))
temp_df = df.where(df['Question'] == 'Peek').dropna()
print(np.mean(temp_df))
temp_df = df.where(df['Question'] == 'Uncomfortable').dropna()
print(np.mean(temp_df))
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Question', y='Score', kind='box', legend='auto', palette=PAL)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'mturk_privacy.pdf'))

df_data = {'Time Spent (s)': [], 'Setting': [], 'Content': []}

for key, val in timings_dict.items():
    if 'pii' not in key:
        df_data['Time Spent (s)'].extend(val)
        if 'video' in key:
            df_data['Content'].extend(['Video'] * len(val))
        elif 'image' in key:
            df_data['Content'].extend(['Image'] * len(val))
        elif 'text' in key:
            df_data['Content'].extend(['Text'] * len(val))
        if 'ois' in key:
            df_data['Setting'].extend(['Original'] * len(val))
        elif 'pis' in key:
            df_data['Setting'].extend(['Protected'] * len(val))

df = pd.DataFrame(df_data)
PAL = sns.color_palette([MUTED[0], MUTED[1]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
g = sns.catplot(data=df, x='Content', y='Time Spent (s)', kind='box', hue='Setting', legend='auto', palette=PAL, order=['Text', 'Image', 'Video'], showfliers=False)
g.set(ylim=[0, 45])
plt.tight_layout()
sns.despine()
sns.move_legend(g, "upper center", ncol=2, title=None)
plt.savefig(os.path.join('data/plots', 'mturk_timings.pdf'))

df_data = {'Quality': [], 'Content': [], 'Privacy': []}

for key, val in qualities_dict.items():
    print(key, np.mean(np.array(val)))
for key, val in user_data.items():
    df_data['Privacy'].extend([(privacy_dict[key]['bothers'] + privacy_dict[key]['uncomfortable']) / 2.0] * len(val['responses']['image_quality']))
    df_data['Privacy'].extend([(privacy_dict[key]['bothers'] + privacy_dict[key]['uncomfortable']) / 2.0] * len(val['responses']['video_quality']))
    df_data['Privacy'].extend([(privacy_dict[key]['bothers'] + privacy_dict[key]['uncomfortable']) / 2.0] * len(val['responses']['text_quality']))
    df_data['Quality'].extend(val['responses']['image_quality'])
    df_data['Quality'].extend(val['responses']['video_quality'])
    df_data['Quality'].extend(val['responses']['text_quality'])
    df_data['Content'].extend(['Image'] * len(val['responses']['image_quality']))
    df_data['Content'].extend(['Video'] * len(val['responses']['video_quality']))
    df_data['Content'].extend(['Text'] * len(val['responses']['text_quality']))

df = pd.DataFrame(df_data)
temp_df = df.where(df['Privacy'] > 3.5).dropna()
temp_df = temp_df.where(temp_df['Content'] == 'Video').dropna()
print(np.mean(temp_df))
# df = df.where(df['Content'] == 'Image').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.displot(data=df, col='Content', x='Privacy', y='Quality', kind='hist', legend='auto', palette=PAL, binwidth=[0.5, 0.8], cbar=True, cbar_ax=2, col_order=['Text', 'Image', 'Video'])
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'mturk_qualities.pdf'))

def print_responses(key, responses_dict, ids_dict, contents):
    correctness_array = []
    prev = 0
    print('=================================')
    print(key)
    print('=================================')
    for i, item in enumerate(responses_dict[key]):
        if ids_dict[key][i] != prev:
            prev = ids_dict[key][i]
            print('=================================')
            cv2.destroyAllWindows() 
            print(contents[prev - 1])
            if 'video' in key:
                cv2.imshow('image', cv2.imread(os.path.join('data/mturk-shoulder-surf/video_original', contents[prev - 1])))
            elif 'text' in key:
                cv2.imshow('image', cv2.imread(os.path.join('data/mturk-shoulder-surf/rico_original', contents[prev - 1])))
            elif 'image' in key:
                cv2.imshow('image', cv2.imread(os.path.join('data/mturk-shoulder-surf/images_original', contents[prev - 1])))
            cv2.waitKey(1)
        print(item)
        error = True
        while error:
            try:
                correctness = input()
                if correctness == 'z':
                    last_annotation = correctness_array.pop()
                    print('Undo: {}'.format(responses_dict[key][i - 1]))
                    correctness = input()
                    while int(correctness) not in [0, 1]:
                        correctness = input()
                    correctness_array.append(int(correctness))
                    print(item)
                    correctness = input()
                    while int(correctness) not in [0, 1]:
                        correctness = input()
                if int(correctness) not in [0, 1]:
                    raise ValueError('Not 1 or 0')
                correctness_array.append(int(correctness))
                error = False
            except ValueError as e:
                print(e)
    print(correctness_array)
    return correctness_array
