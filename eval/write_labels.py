# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import csv
from tkinter.filedialog import SaveFileDialog
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SIMILARITY = True
if SIMILARITY:
    COUNT = 2
else:
    COUNT = 1

MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]

if SIMILARITY:
    df_data = {'Mode': [], 'Dataset': [], 'Downscale': [], 'Gridsize': [], 'Parameter': [], 'Similarity': [], 'Pair': [], 'Contrast': []}
else:
    df_data = {'Mode': [], 'Dataset': [], 'Downscale': [], 'Gridsize': [], 'Parameter': [], 'Similarity': [], 'Contrast': []}
for file in os.listdir('data/csvs'):
    if not file.startswith('performance-') and 'area' in file:
        with open(os.path.join('data/csvs', file), 'r') as infile:
            reader = csv.reader(infile)
            information = []
            similarity = []
            pair = []
            for i, row in enumerate(reader):
                if i > 0:
                    if SIMILARITY:
                        similarity.append(float(row[2]))
                        pair.append('Original')
                        similarity.append(float(row[8]))
                        pair.append('Target')
                    else:
                        similarity.append(float(row[1]) - float(row[6]))
                        # similarity.append(float(row[7]))
                        # similarity.append(float(row[6]))
                    if 'pixelateblur' in file:
                        df_data['Mode'].extend(['pixelateblur'] * COUNT)
                    elif 'blur' in file:
                        df_data['Mode'].extend(['blur'] * COUNT)
                    elif 'pixelate' in file:
                        df_data['Mode'].extend(['pixelate'] * COUNT)

                    if 'div2kvalid' in file:
                        df_data['Dataset'].extend(['div2kvalid'] * COUNT)
                    elif 'div2ktrain' in file:
                        df_data['Dataset'].extend(['div2ktrain'] * COUNT)
                    elif '480p' in file:
                        df_data['Dataset'].extend(['480p'] * COUNT)
                    elif '1080p' in file:
                        df_data['Dataset'].extend(['1080p'] * COUNT)
                    elif 'ricovalid' in file:
                        df_data['Dataset'].extend(['ricovalid'] * COUNT)

                    if 'area-0.25' in file:
                        df_data['Downscale'].extend([0.25] * COUNT)
                    elif 'area-0.33' in file:
                        df_data['Downscale'].extend([0.33] * COUNT)
                    elif 'area-0.2' in file:
                        df_data['Downscale'].extend([0.2] * COUNT)
                    elif 'area-0.5' in file:
                        df_data['Downscale'].extend([0.5] * COUNT)

                    if '-1-' in file:
                        df_data['Gridsize'].extend([1] * COUNT)
                    elif '-2-' in file:
                        df_data['Gridsize'].extend([2] * COUNT)
                    elif '-3-' in file:
                        df_data['Gridsize'].extend([3] * COUNT)
                    elif '-4-' in file:
                        df_data['Gridsize'].extend([4] * COUNT)

                    if '-32-' in file:
                        df_data['Parameter'].extend([32] * COUNT)
                    elif '-24-' in file:
                        df_data['Parameter'].extend([24] * COUNT)
                    elif '-16-' in file:
                        df_data['Parameter'].extend([16] * COUNT)
                    elif '-8-' in file:
                        df_data['Parameter'].extend([8] * COUNT)
                    
                    if '-100' in file:
                        df_data['Contrast'].extend([100] * COUNT)
                    elif '-75' in file:
                        df_data['Contrast'].extend([75] * COUNT)
                    else:
                        df_data['Contrast'].extend([127] * COUNT)
            # df_data['Entropy Original'].append(np.mean(information_original))
            # df_data['Entropy Target'].append(np.mean(information_target))
            df_data['Similarity'].extend(similarity)
            if SIMILARITY:
                df_data['Pair'].extend(pair)

for key, val in df_data.items():
    print(key, len(val))

orig_df = pd.DataFrame(df_data)

if SIMILARITY:

    # df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    # df = df.where(df['Parameter'] == 16).dropna()
    # df = df.where(df['Gridsize'] == 1).dropna()
    # PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # # lgnd = plt.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # sns.despine()
    # sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    # plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset.pdf'))

    # df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    # df = df.where(df['Downscale'] == 0.2).dropna()
    # df = df.where(df['Gridsize'] == 1).dropna()
    # PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # # lgnd = plt.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # sns.despine()
    # sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    # plt.savefig(os.path.join('data/plots', 'parameter_similarity_dataset.pdf'))

    # df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    # df = df.where(df['Downscale'] == 0.2).dropna()
    # df = df.where(df['Parameter'] == 16).dropna()
    # PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # # lgnd = plt.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # sns.despine()
    # sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    # plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset.pdf'))

    # df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    # df = df.where(df['Parameter'] == 16).dropna()
    # df = df.where(df['Gridsize'] == 1).dropna()
    # PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # # lgnd = plt.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # sns.despine()
    # sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    # plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset_original.pdf'))

    # df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    # df = df.where(df['Downscale'] == 0.2).dropna()
    # df = df.where(df['Gridsize'] == 1).dropna()
    # PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # # lgnd = plt.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # sns.despine()
    # sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    # plt.savefig(os.path.join('data/plots', 'parameter_similarity_dataset_original.pdf'))

    # df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    # df = df.where(df['Downscale'] == 0.2).dropna()
    # df = df.where(df['Parameter'] == 16).dropna()
    # PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    # sns.set(font_scale=1.5, style='ticks', palette=PAL)
    # plt.figure(figsize=(16, 12))
    # g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # # lgnd = plt.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # sns.despine()
    # sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    # plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset_original.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 100).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset100.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 100).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'parameter_similarity_dataset100.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Contrast'] == 100).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset100.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 100).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset_original100.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 100).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'parameter_similarity_dataset_original100.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Contrast'] == 100).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset_original100.pdf'))
    
    df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 75).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset75.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 75).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'parameter_similarity_dataset75.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Target').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Contrast'] == 75).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset75.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 75).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset_original75.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    df = df.where(df['Contrast'] == 75).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'parameter_similarity_dataset_original75.pdf'))

    df = orig_df.where(orig_df['Pair'] == 'Original').dropna()
    df = df.where(df['Downscale'] == 0.2).dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    df = df.where(df['Contrast'] == 75).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'], showfliers=False)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(0.99, 0.37), ncol=1, title=None, frameon=True)
    plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset_original75.pdf'))

else:

    orig_df = orig_df.where(orig_df['Dataset'] != 'ricovalid').dropna()
    df = orig_df.where(orig_df['Parameter'] == 16).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Downscale', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'])
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(1, 0.83), ncol=1, title=None)
    plt.savefig(os.path.join('data/plots', 'downscale_entropy_dataset.pdf'))

    df = orig_df.where(orig_df['Downscale'] == 0.2).dropna()
    df = df.where(df['Gridsize'] == 1).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Parameter', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'])
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(1, 0.83), ncol=1, title=None)
    plt.savefig(os.path.join('data/plots', 'parameter_entropy_dataset.pdf'))

    df = orig_df.where(orig_df['Downscale'] == 0.2).dropna()
    df = df.where(df['Parameter'] == 16).dropna()
    PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
    sns.set(font_scale=1.5, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, col='Mode', x='Gridsize', y='Similarity', hue='Dataset', kind='box', legend='auto', palette=PAL, hue_order=['ricovalid', '1080p', '480p', 'div2ktrain', 'div2kvalid'])
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "center right", bbox_to_anchor=(1, 0.83), ncol=1, title=None)
    plt.savefig(os.path.join('data/plots', 'gridsize_entropy_dataset.pdf'))

exit()

df = orig_df.where(orig_df['Dataset'] == 'div2kvalid').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Downscale', y='Similarity Original', hue='Mode', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'downscale_similarity_mode.pdf'))

df = orig_df.where(orig_df['Dataset'] == 'div2kvalid').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Gridsize', y='Similarity Original', hue='Mode', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'gridsize_similarity_mode.pdf'))

df = orig_df.where(orig_df['Mode'] == 'blur').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Downscale', y='Similarity Original', hue='Dataset', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset.pdf'))

df = orig_df.where(orig_df['Mode'] == 'blur').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Gridsize', y='Similarity Original', hue='Dataset', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset.pdf'))

df = orig_df.where(orig_df['Dataset'] == 'div2kvalid').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Downscale', y='Similarity Target', hue='Mode', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'downscale_similarity_mode_target.pdf'))

df = orig_df.where(orig_df['Dataset'] == 'div2kvalid').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Gridsize', y='Similarity Target', hue='Mode', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'gridsize_similarity_mode_target.pdf'))

df = orig_df.where(orig_df['Mode'] == 'blur').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Downscale', y='Similarity Target', hue='Dataset', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'downscale_similarity_dataset_target.pdf'))

df = orig_df.where(orig_df['Mode'] == 'blur').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Gridsize', y='Similarity Target', hue='Dataset', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'gridsize_similarity_dataset_target.pdf'))

df = orig_df.where(orig_df['Dataset'] == 'div2kvalid').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Downscale', y='Entropy Original', hue='Mode', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'downscale_entropyorig_mode.pdf'))

df = orig_df.where(orig_df['Dataset'] == 'div2kvalid').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Gridsize', y='Entropy Original', hue='Mode', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'gridsize_entropyorig_mode.pdf'))

df = orig_df.where(orig_df['Mode'] == 'pixelate').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Downscale', y='Entropy Original', hue='Dataset', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'downscale_entropyorig_dataset.pdf'))

df = orig_df.where(orig_df['Mode'] == 'pixelate').dropna()
PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4]])
sns.set(font_scale=1.5, style='ticks', palette=PAL)
plt.figure(figsize=(16, 12))
g = sns.catplot(data=df, x='Gridsize', y='Entropy Original', hue='Dataset', kind='box', legend='auto', palette=PAL)
# lgnd = plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join('data/plots', 'gridsize_entropyorig_dataset.pdf'))
