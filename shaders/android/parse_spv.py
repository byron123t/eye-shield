big_list = []
with open('spv.txt', 'r') as infile:
	for line in infile:
		split = line.strip().split(' ')
		for item in split:
			big_list.append('0x{}'.format(item[:2]))
			big_list.append('0x{}'.format(item[2:]))
for i in range(0, len(big_list), 15):
	print(big_list[i:i + 15])
