# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import csv
import pandas as pd
import matplotlib; 
matplotlib.use('agg');
import seaborn as sns
import matplotlib.pyplot as plt


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
PAL3 = sns.color_palette([MUTED[0], MUTED[2], MUTED[8], MUTED[3]])
PAL = sns.color_palette([MUTED[0], MUTED[1]])
PAL_MOBILE = sns.color_palette([MUTED[2], MUTED[3]])

BLUR_FLAG = False

ENERGY_DICT = {'Light': 'Low', 'High': 'Medium', 'Very High': 'High', 'Medium': 'Medium', 'Low': 'Low'}

def parse_file(filename, df_data):
    with open('data/{}'.format(filename), 'r') as infile:
        for line in infile:
            if len(line.strip()) > 0:
                if ' - ' in line:
                    split = line.strip().split(' - ')
                    if 'CPU Max Utilization' in cur_key:
                        if 'ios' in filename:
                            df_data['CPU Utilization (%)'].append(float(split[1]) / 6)
                        elif 'mac' in filename:
                            df_data['CPU Utilization (%)'].append(float(split[1]) / 8)
                        else:
                            df_data['CPU Utilization (%)'].append(float(split[1]))
                        df_data['Resolution'].append(int(split[0]))
                    elif 'Memory Max Usage' in cur_key:
                        if 'ios' in filename or 'mac' in filename:
                            df_data['Memory (MB)'].append(float(split[1]) / 100)
                        else:
                            df_data['Memory (MB)'].append(float(split[1]) / 100)
                    elif 'GPU Max Clock' in cur_key:
                        df_data['GPU Clock Speed'].append(float(split[1]))
                    elif 'GPU Utilization' in cur_key:
                        df_data['GPU Utilization'].append(float(split[1]))
                    elif 'GPU Max Read/Write' in cur_key:
                        df_data['GPU Read/Write'].append(float(split[1]))
                    elif 'FPS' in cur_key:
                        df_data['GPU FPS'].append(float(split[1]))
                    elif 'GPU Frame Time' in cur_key:
                        df_data['GPU Frame Speed'].append(float(split[1]))
                    elif 'Energy Impact' in cur_key:
                        df_data['Energy Impact'].append(ENERGY_DICT[split[1]])
                else:
                    cur_key = line.strip()
    return df_data


def parse_mobile_performance(filetype, df_data):
    for i in ['3088', '2560', '2532', '2400','1920','1600','1366','1280', '1024', '960', '854', '640', '512', '426', '256']:
        if filetype == 'out-penguin':
            with open('data/{}{}.txt'.format('out-penguin', i), 'r') as infile:
                for line in infile:
                    split = line.strip().split('microseconds')
                    time1 = split[0]
                    time2 = split[1].replace(split[0], '')
                    time3 = split[2].replace(split[1], '')
                    break
            with open('data/{}{}.txt'.format('out1-penguin', i), 'r') as infile:
                for line in infile:
                    time4 = line.strip().replace('microseconds', '')
                    break
            df_data['Latency (s)'].append((float(time1) + float(time2) + float(time3) + float(time4)) / 1000 / 1000)
            df_data['FPS'].append(1 / ((float(time1) + float(time2) + float(time3) + float(time4)) / 1000 / 1000))
            df_data['Mode'].append('blur')
            df_data['CPU Cores'].append(8)
            df_data['Hardware'].append('GPU')
            df_data['GPU Memory'].append(0)
            df_data['Device'].append('Android')
            df_data['GPU FPS'].append(0)
            df_data['GPU Frame Speed'].append(0)
            
        elif filetype == 'mac-perf-penguins':
            with open('data/{}{}.txt'.format('mac-perf-penguins', i), 'r') as infile:
                for line in infile:
                    df_data['Latency (s)'].append(float(line.strip()))
                    df_data['FPS'].append(1 / (float(line.strip())))
                    df_data['Mode'].append('blur')
                    df_data['CPU Cores'].append(8)
                    df_data['Hardware'].append('GPU')
                    df_data['GPU Memory'].append(0)
                    df_data['Device'].append('Mac')
                    df_data['GPU Clock Speed'].append(0)
                    df_data['GPU Read/Write'].append(0)
                    df_data['Energy Impact'].append('N/A')
                    df_data['GPU FPS'].append(0)
                    df_data['GPU Frame Speed'].append(0)
                    df_data['GPU Utilization'].append(0)
                    break
                
        elif filetype == 'ios-perf-penguins':
            with open('data/{}{}.txt'.format('ios-perf-penguins', i), 'r') as infile:
                for line in infile:
                    df_data['Latency (s)'].append(float(line.strip()))
                    df_data['FPS'].append(1 / (float(line.strip())))
                    df_data['Mode'].append('blur')
                    df_data['CPU Cores'].append(8)
                    df_data['Hardware'].append('GPU')
                    df_data['GPU Memory'].append(0)
                    df_data['Device'].append('iOS')
                    df_data['GPU Clock Speed'].append(0)
                    df_data['GPU Read/Write'].append(0)
                    df_data['GPU FPS'].append(0)
                    df_data['GPU Frame Speed'].append(0)
                    df_data['GPU Utilization'].append(0)
                    break
                
    return df_data



df_data = {'Resolution': [], 'Mode': [], 'Memory (MB)': [], 'CPU Utilization (%)': [], 'CPU Cores': [], 'Latency (s)': [], 'FPS': [], 'Hardware': [], 'GPU Memory': [], 'Device': [], 'GPU Utilization': [], 'GPU Frame Speed': [], 'GPU Clock Speed': [], 'GPU Read/Write': [], 'GPU FPS': [], 'Energy Impact': []}
for file in os.listdir('data/csvs'):
    if file.startswith('performance-') and 'memory' in file and file.endswith('.csv') and ((BLUR_FLAG and 'blur-sqrt-1-16' in file and 'gpu' in file) or not BLUR_FLAG):
        with open(os.path.join('data/csvs', file), 'r') as infile:
            reader = csv.reader(infile)
            if '-blur' in file:
                df_data['Resolution'].append(int(file.split('div2k')[1].split('-blur')[0]))
            elif '-pixelate' in file:
                df_data['Resolution'].append(int(file.split('div2k')[1].split('-pixelate')[0]))

            if 'pixelate' in file:
                df_data['Mode'].append('Pixelate')
            elif 'blur' in file:
                df_data['Mode'].append('Blur')

            if 'gpu' in file:
                df_data['Hardware'].append('GPU')
            else:
                df_data['Hardware'].append('CPU')

            for i, row in enumerate(reader):
                if i == 0:
                    df_data['Memory (MB)'].append(float(row[0].replace('MB', '')) / 50)
                elif i == 1:
                    df_data['CPU Utilization (%)'].append(float(row[0].replace('%', '')) * 2)
                elif i == 2:
                    df_data['CPU Cores'].append(row[0].replace('cpus', 'CPUs'))
                elif i == 3:
                    df_data['GPU Memory'].append(float(row[0].replace('MB', '')))
            if i < 3:
                df_data['GPU Memory'].append(0)

            with open(os.path.join('data/csvs', file.replace('-memory', '')), 'r') as infileperf:
                readerperf = csv.reader(infileperf)
                for i, row in enumerate(readerperf):
                    if row[0] == 'algo':
                        algo = float(row[1])
                    if row[0] == 'pixelateblur':
                        pixelateblur = float(row[1])
                    if row[0] == 'grid':
                        grid = float(row[1])
                df_data['Latency (s)'].append(algo + pixelateblur + grid)
                df_data['FPS'].append(1 / (algo + pixelateblur + grid))
            df_data['Device'].append('PC')
            df_data['GPU Utilization'].append(0)
            df_data['GPU Frame Speed'].append(0)
            df_data['GPU Clock Speed'].append(0)
            df_data['GPU Read/Write'].append(0)
            df_data['GPU FPS'].append(0)
            df_data['Energy Impact'].append('N/A')


df_data = parse_file('cpu_mem_usages.txt', df_data)
df_data = parse_mobile_performance('out-penguin', df_data)
df_data = parse_file('mac_cpu_mem_usages.txt', df_data)
df_data = parse_mobile_performance('mac-perf-penguins', df_data)
df_data = parse_file('ios_cpu_mem_usages.txt', df_data)
df_data = parse_mobile_performance('ios-perf-penguins', df_data)
for key, val in df_data.items():
    print(key, len(val))
orig_df = pd.DataFrame(df_data)

if not BLUR_FLAG:
    df = orig_df.where(orig_df['Hardware'] == 'CPU').dropna()
    print(df['Memory (MB)'].where(df['Mode'] == 'Blur').dropna())
    print(df['Memory (MB)'].where(df['Mode'] == 'Pixelate').dropna())
    print(df['Resolution'][df['FPS'].idxmin()])
    print(df['FPS'].min())
    print(df['Resolution'][df['Latency (s)'].idxmax()])
    print(df['Latency (s)'].max())
    print(df['Resolution'][df['Memory (MB)'].idxmax()])
    print(df['Memory (MB)'].max())
    print(df['FPS'].where(df['Resolution'] == 854).dropna())
    print(df['FPS'].where(df['Resolution'] == 1920).dropna())
    print(df['Latency (s)'].where(df['Resolution'] == 512).dropna())
    print(df['Latency (s)'].where(df['Resolution'] == 3088).dropna())
    print()

    sns.set(font_scale=1.93, style='ticks', palette=PAL)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='Latency (s)', hue='Mode', kind='line', legend='auto', palette=PAL, ci=None)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'cpu_latency.pdf'))

    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='FPS', hue='Mode', kind='line', legend='auto', palette=PAL, ci=None)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'cpu_fps.pdf'))

    sns.set(font_scale=1.5, style='ticks', palette=PAL3)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='CPU Utilization (%)', hue='Mode', kind='line', legend='auto', palette=PAL)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'cpu_utilization.pdf'))
    
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='Memory (MB)', hue='Mode', kind='line', legend='auto', palette=PAL)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'cpu_memory.pdf'))
else:
    df = orig_df
    print(df['Memory (MB)'].max())
    temp_df = orig_df.where(orig_df['Device'] == 'PC').dropna()
    print()
    print('PC')
    print(temp_df['Resolution'][temp_df['FPS'].idxmin()])
    print(temp_df['FPS'].min())
    print(temp_df['Resolution'][temp_df['GPU Memory'].idxmax()])
    print(temp_df['GPU Memory'].max())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 512).dropna())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 3088).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 854).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 1920).dropna())
    print(temp_df[['Resolution', 'FPS']].where(temp_df['Resolution'] >= 1920).dropna())
    print()
    temp_df = orig_df.where(orig_df['Device'] == 'Android').dropna()
    print('Android')
    print(temp_df['Resolution'][temp_df['FPS'].idxmin()])
    print(temp_df['FPS'].min())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 512).dropna())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 3088).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 854).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 1920).dropna())
    print(temp_df[['Resolution', 'FPS']].where(temp_df['Resolution'] >= 1920).dropna())
    print()
    temp_df = orig_df.where(orig_df['Device'] == 'Mac').dropna()
    print('Mac')
    print(temp_df['Resolution'][temp_df['FPS'].idxmin()])
    print(temp_df['FPS'].min())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 512).dropna())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 3088).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 854).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 1920).dropna())
    print(temp_df[['Resolution', 'FPS']].where(temp_df['Resolution'] >= 1920).dropna())
    print()
    temp_df = orig_df.where(orig_df['Device'] == 'iOS').dropna()
    print('iOS')
    print(temp_df['Resolution'][temp_df['FPS'].idxmin()])
    print(temp_df['FPS'].min())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 512).dropna())
    print(temp_df['Latency (s)'].where(temp_df['Resolution'] == 3088).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 854).dropna())
    print(temp_df['FPS'].where(temp_df['Resolution'] == 1920).dropna())
    print(temp_df[['Resolution', 'FPS']])
    sns.set(font_scale=1.93, style='ticks', palette=PAL3)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='Latency (s)', hue='Device', kind='line', legend='auto', palette=PAL3)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'gpu_latency.pdf'))

    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='FPS', hue='Device', kind='line', legend='auto', palette=PAL3)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'gpu_fps.pdf'))

    sns.set(font_scale=1.5, style='ticks', palette=PAL3)
    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='CPU Utilization (%)', hue='Device', kind='line', legend='auto', palette=PAL3)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    g.set(ylim=(0, 35))
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'gpu_cpu_utilization.pdf'))

    plt.figure(figsize=(16, 12))
    g = sns.relplot(data=df, x='Resolution', y='Memory (MB)', hue='Device', kind='line', legend='auto', palette=PAL3)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'gpu_memory.pdf'))
    
    df = orig_df.where((orig_df['Device'] == 'Android') | (orig_df['Device'] == 'iOS')).dropna()
    
    plt.figure(figsize=(16, 12))
    g = sns.catplot(data=df, x='Resolution', y='Energy Impact', hue='Device', kind='box', order = ['High','Medium','Low'], col_order = ['High','Medium','Low'], palette=PAL_MOBILE)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, "upper center", ncol=2, title=None, frameon=False)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join('data/plots', 'gpu_energy_impact.pdf'))