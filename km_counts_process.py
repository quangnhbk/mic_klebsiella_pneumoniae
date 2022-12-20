import os
import pickle

kmer_list = set(list())
N_max = 0
N_min = 999999
for file_index, filename in enumerate(os.listdir('./counts_data_8')):
    print(file_index)
    count = 0
    with open('./counts_data_8/' + filename , 'r') as data_file:
        for line in data_file.read().splitlines():
            kmer_counts = int(line.split("\t")[-1])
            kmer = line.split("\t")[0]
            kmer_list.add(kmer)
            if kmer_counts > N_max:
                N_max = kmer_counts
            if kmer_counts < N_min:
                N_min = kmer_counts
            count += 1
print(N_max, N_min)
print(len(kmer_list))

with open('all_8mers', 'wb') as fp:
    pickle.dump(kmer_list, fp)

