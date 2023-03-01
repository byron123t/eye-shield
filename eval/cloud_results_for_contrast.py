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


def find_differences(df, dataset, parameter, cloud_api):
    if cloud_api == 'label':
        query = 'Label'
    elif cloud_api == 'text':
        query = 'Text'
    df_hidden = df.where(df['Mode'] == 'Hidden').where(df['Dataset'] == dataset).where(df['Parameter'] == parameter).dropna()
    df_target = df.where(df['Mode'] == 'Target').where(df['Dataset'] == dataset).where(df['Parameter'] == parameter).dropna()
    df_orig = df.where(df['Mode'] == 'Original').where(df['Dataset'] == dataset).dropna()
    file_list = df_orig.drop_duplicates(['File']).dropna()['File']
    missing_labels = []
    total_labels = 0
    total_missing_labels = 0
    for file in file_list:
        found = df_orig.where(df_orig['File'] == file).dropna()
        missing_labels_file = 0
        for label in found[query]:
            newscore = df_hidden.where(df_hidden['File'] == file).where(df_hidden[query] == label).drop_duplicates(['File']).dropna()
            if newscore.empty:
                missing_labels_file += 1
                total_missing_labels += 1
            total_labels += 1
        missing_labels.append((1 - (missing_labels_file / len(found[query]))) * 100)
    print(total_missing_labels / total_labels, parameter)
    return missing_labels, total_missing_labels, total_labels


def load_data(cloud_api, dataset_name):
    if cloud_api == 'label':
        query1 = 'Label'
        query2 = 'Score'
        file = 'cloud_results_labels_contrast'
    elif cloud_api == 'text':
        query1 = 'Text'
        query2 = 'Bounds'
        file = 'cloud_results_text_contrast'
    df_data = {'File': [], 'Dataset': [], 'Parameter': [], 'Mode': [], query1: [], query2: []}
    parameters = ['pixelate-1-16-100', 'pixelate-1-32-100', 'pixelate-2-16-100', 'pixelate-2-32-100', 'pixelate-3-16-100', 'pixelate-3-32-100', 'pixelate-4-16-100', 'pixelate-4-32-100', 'blur-1-8-100', 'blur-1-16-100', 'blur-2-8-100', 'blur-2-16-100', 'blur-3-8-100', 'blur-3-16-100', 'blur-4-8-100', 'blur-4-16-100', 'pixelate-1-16-75', 'pixelate-1-32-75', 'pixelate-2-16-75', 'pixelate-2-32-75', 'pixelate-3-16-75', 'pixelate-3-32-75', 'pixelate-4-16-75', 'pixelate-4-32-75', 'blur-1-8-75', 'blur-1-16-75', 'blur-2-8-75', 'blur-2-16-75', 'blur-3-8-75', 'blur-3-16-75', 'blur-4-8-75', 'blur-4-16-75']
    with open ('data/csvs/{}.csv'.format(file), 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for line in reader:
            filename = line[0]
            split = filename.split('/')
            df_data['File'].append(split[-1])
            df_data['Dataset'].append(split[-2].replace('_downscaled', ''))
            param_exists = False
            for param in parameters:
                if param in filename:
                    df_data['Parameter'].append(param)
                    param_exists = True
            if not param_exists:
                df_data['Parameter'].append('None')
            if 'blurred' in filename or 'pixelated' in filename:
                df_data['Mode'].append('Target')
            elif 'hidden' in filename:
                df_data['Mode'].append('Hidden')
            else:
                df_data['Mode'].append('Original')
            df_data[query1].append(line[1])
            if cloud_api == 'label':
                df_data[query2].append(float(line[2]))
            elif cloud_api == 'text':
                df_data[query2].append(line[2])
    df = pd.DataFrame(df_data)
    dataset_list = df.drop_duplicates(['Dataset']).dropna()['Dataset']
    all_missing_labels = {'Labels Retained (%)': [], 'Dataset': [], 'Parameter': []}
    pixelation_only = 0
    pixelation_only_total = 0
    blurring_only = 0
    blurring_only_total = 0
    everything = 0
    everything_total = 0
    for dataset in tqdm(dataset_list):
        if dataset == dataset_name:
            print(dataset)
            for param in parameters:
                missing_labels, missing, total = find_differences(df, dataset, param, cloud_api)
                if 'blur' in param:
                    blurring_only += missing
                    blurring_only_total += total
                else:
                    pixelation_only += missing
                    pixelation_only_total += total
                everything += missing
                everything_total += total
                all_missing_labels['Labels Retained (%)'].extend(missing_labels)
                all_missing_labels['Dataset'].extend([dataset] * len(missing_labels))
                all_missing_labels['Parameter'].extend([param] * len(missing_labels))
    print(blurring_only / blurring_only_total)
    print(pixelation_only / pixelation_only_total)
    print(everything / everything_total)
    return all_missing_labels


def run_labels(dataset_name):
    all_missing_labels = load_data('label', dataset_name)
    print(np.mean(all_missing_labels['Labels Retained (%)']))

    df = pd.DataFrame(all_missing_labels)
    temp_df = df.where(df['Parameter'].isin(['pixelate-1-16-100', 'pixelate-2-16-100', 'pixelate-3-16-100', 'pixelate-4-16-100', 'blur-1-16-100', 'blur-2-16-100', 'blur-3-16-100', 'blur-4-16-100']))
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.displot(data=temp_df, x='Labels Retained (%)', hue='Parameter', kind='ecdf', legend='auto', palette=PAL, complementary=True)
    # sns.move_legend(g, 'upper right')
    g.set(xlim=[0, 90])
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join('data/plots', 'google_contrast100_{}.pdf'.format(dataset_name)))

    temp_df = df.where(df['Parameter'].isin(['pixelate-1-16-75', 'pixelate-2-16-75', 'pixelate-3-16-75', 'pixelate-4-16-75', 'blur-1-16-75', 'blur-2-16-75', 'blur-3-16-75', 'blur-4-16-75']))
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.displot(data=temp_df, x='Labels Retained (%)', hue='Parameter', kind='ecdf', legend='auto', palette=PAL, complementary=True)
    # sns.move_legend(g, 'upper right')
    g.set(xlim=[0, 90])
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join('data/plots', 'google_contrast75_{}.pdf'.format(dataset_name)))


def run_text(dataset_name):
    all_missing_labels = load_data('text', dataset_name)
    print(np.mean(all_missing_labels['Labels Retained (%)']))

    df = pd.DataFrame(all_missing_labels)
    temp_df = df.where(df['Parameter'].isin(['pixelate-1-32-100', 'pixelate-2-32-100', 'pixelate-3-32-100', 'pixelate-4-32-100', 'blur-1-8-100', 'blur-2-8-100', 'blur-3-8-100', 'blur-4-8-100']))
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.displot(data=temp_df, x='Labels Retained (%)', hue='Parameter', kind='ecdf', legend='auto', palette=PAL, complementary=True)
    # sns.move_legend(g, 'upper right')
    g.set(xlim=[0, 90])
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join('data/plots', 'google_contrast100_{}.pdf'.format(dataset_name)))
    
    temp_df = df.where(df['Parameter'].isin(['pixelate-1-32-75', 'pixelate-2-32-75', 'pixelate-3-32-75', 'pixelate-4-32-75', 'blur-1-8-75', 'blur-2-8-75', 'blur-3-8-75', 'blur-4-8-75']))
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.displot(data=temp_df, x='Labels Retained (%)', hue='Parameter', kind='ecdf', legend='auto', palette=PAL, complementary=True)
    # sns.move_legend(g, 'upper right')
    g.set(xlim=[0, 90])
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join('data/plots', 'google_contrast75_{}.pdf'.format(dataset_name)))


run_labels('div2kvalid')
run_text('ricovalid')
