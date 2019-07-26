import os


# preprocess train data based on baseline.py
# step 1 train data in
def process_train(data_path, train_file, save_file):
	with open(data_path+train_file, 'r', encoding='utf-8') as f:
	    lines = f.readlines()
	    results = []
	    for line in lines:
	        features = []
	        tags = []
	        samples = line.strip().split('  ')
	        for sample in samples:
	            sample_list = sample[:-2].split('_')
	            tag = sample[-1]
	            features.extend(sample_list)
	            tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(['B-' + tag] + ['I-' + tag] * (len(sample_list)-1))
	        results.append(dict({'features': features, 'tags': tags}))
	    train_write_list = []
	    with open(data_path+save_file, 'w', encoding='utf-8') as f_out:
	        for result in results:
	            for i in range(len(result['tags'])):
	                train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
	            train_write_list.append('\n')
	        f_out.writelines(train_write_list)

# step 2 test data in
def process_valid(data_path, test_file, save_file):
	with open(data_path+test_file, 'r', encoding='utf-8') as f:
	    lines = f.readlines()
	    results = []
	    for line in lines:
	        features = []
	        sample_list = line.split('_')
	        features.extend(sample_list)
	        results.append(dict({'features': features}))
	    test_write_list = []
	    with open(data_path+save_file, 'w', encoding='utf-8') as f_out:
	        for result in results:
	            for i in range(len(result['features'])):
	                test_write_list.append(result['features'][i] + '\n')
	            test_write_list.append('\n')
	        f_out.writelines(test_write_list)


# define data path 
data_path = 'data/datagrand/'

# input train file, valid file, test file 
process_train(data_path, 'train.txt', 'dg_train.txt')
# process_train(data_path, 'valid.txt', 'valid.txt')
process_valid(data_path, 'test.txt', 'dg_test.txt')
