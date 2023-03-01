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


def load_data(dataset_name):
    query1 = 'Text'
    query2 = 'Bounds'
    file = 'cloud_results_text_brightness'
    df_data = {'File': [], 'Dataset': [], 'Brightness': [], query1: [], query2: []}
    with open ('data/csvs/{}.csv'.format(file), 'r', encoding='utf8') as infile:
        reader = csv.reader(infile)
        for line in reader:
            filename = line[0]
            split = filename.split('/')
            df_data['File'].append(split[-1])
            df_data['Dataset'].append(split[-2])
            param_exists = False
            if '_33' in filename:
                df_data['Brightness'].append(1)
            elif '_66' in filename:
                df_data['Brightness'].append(2)
            else:
                df_data['Brightness'].append(3)
            df_data[query1].append(line[1])
            df_data[query2].append(line[2])
    df = pd.DataFrame(df_data)
    dataset_list = df.drop_duplicates(['Dataset']).dropna()['Dataset']
    for dataset in dataset_list:
        if dataset == dataset_name:
            print(dataset)
            temp_df = df.where(df['Dataset'] == dataset_name).dropna()
            darkest = temp_df.where(temp_df['Brightness'] == 1).dropna()
            moderate = temp_df.where(temp_df['Brightness'] == 2).dropna()
            brightest = temp_df.where(temp_df['Brightness'] == 3).dropna()
            print('darkest', len(darkest))
            print('moderate', len(moderate))
            print('brightest', len(brightest))
            print()


def run_text(dataset_name):
    all_missing_labels = load_data(dataset_name)
    # print(np.mean(all_missing_labels['Labels Retained (%)']))

    # df = pd.DataFrame(all_missing_labels)
    # temp_df = df.where(df['Parameter'].isin(['pixelate-1-32', 'pixelate-2-32', 'pixelate-3-32', 'pixelate-4-32', 'blur-1-8', 'blur-2-8', 'blur-3-8', 'blur-4-8']))
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.displot(data=temp_df, x='Labels Retained (%)', hue='Parameter', bins=9, kind='hist', multiple='dodge', legend='auto', palette=PAL)
    # sns.move_legend(g, 'upper right')
    # g.set(xlim=[0, 90])
    # plt.tight_layout()
    # sns.despine()
    # plt.savefig(os.path.join('data/plots', 'google_{}.pdf'.format(dataset_name)))

    # df = pd.DataFrame(all_missing_labels)
    # temp_df = df.where(df['Parameter'].isin(['pixelate-1-8', 'pixelate-1-16', 'pixelate-1-24', 'pixelate-1-32', 'blur-1-8', 'blur-1-16', 'blur-1-24', 'blur-1-32']))
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.displot(data=temp_df, x='Labels Retained (%)', hue='Parameter', kind='hist', multiple='dodge', legend='auto', palette=PAL)
    # g.set(xlim=[0, 90])
    # plt.tight_layout()
    # sns.despine()
    # plt.savefig(os.path.join('data/plots', 'google_cloud_text_param_{}.pdf'.format(dataset_name)))

run_text('dark-close')
run_text('dark-far')
run_text('dark-side')
run_text('light-close')
run_text('light-far')
run_text('light-side')
run_text('orig-close')
run_text('orig-far')
run_text('orig-side')