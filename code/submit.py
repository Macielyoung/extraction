# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:40:11 2019

@author: bwhe
"""


# submit data
# define input_file path and submit_file here
input_file = ''
submit_file = ''
f_write = open(submit_file, 'w', encoding='utf-8') 
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n\n')
    for line in lines:
        if line == '':
            continue
        tokens = line.split('\n')
        features = []
        tags = []
        for token in tokens:
            feature_tag = token.split('    ')
            features.append(feature_tag[0])
            tags.append(feature_tag[-1])
        samples = []
        i = 0
        while i < len(features):
            sample = []
            if tags[i] == 'O':
                sample.append(features[i])
                j = i + 1
                while j < len(features) and tags[j] == 'O':
                    sample.append(features[j])
                    j += 1
                samples.append('_'.join(sample) + '/o')
            else:
                """ 先不管标签准确性 
                if tags[i][0] != 'B':
                    print(tags[i][0] + ' error start')
                    j = i + 1
                else:
                """
                sample.append(features[i])
                j = i + 1
                while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                    sample.append(features[j])
                    j += 1
                samples.append('_'.join(sample) + '/' + tags[i][-1])
            i = j
        f_write.write('  '.join(samples) + '\n')