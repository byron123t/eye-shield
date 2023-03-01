import os
import csv


cur_dict = {}
cur_resolutions = {}
with open('data/cpu_mem_usages.txt', 'r') as infile:
    for line in infile:
        if len(line.strip()) > 0:
            if ' - ' in line:
                split = line.strip().split(' - ')
                cur_dict[cur_key].append(float(split[1]))
                cur_resolutions[cur_key].append(int(split[0]))
            else:
                cur_key = line.strip()
                cur_dict[cur_key] = []
                cur_resolutions[cur_key] = []
print(cur_dict)
print(cur_resolutions)
mac_cur_dict = {}
mac_cur_resolutions = {}
with open('data/cpu_mem_usages_mac.txt', 'r') as infile:
    for line in infile:
        if len(line.strip()) > 0:
            if ' - ' in line:
                split = line.strip().split(' - ')
                mac_cur_dict[cur_key].append(float(split[1]))
                mac_cur_resolutions[cur_key].append(int(split[0]))
            else:
                cur_key = line.strip()
                mac_cur_dict[cur_key] = []
                mac_cur_resolutions[cur_key] = []
print(mac_cur_dict)
print(mac_cur_resolutions)

