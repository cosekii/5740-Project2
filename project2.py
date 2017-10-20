from collections import defaultdict
import operator
import random
import numpy as np


def get_training_corpus(file):
    corpus = []
    f = open(file, 'r')
    i = 1
    for line in f:
        if i%3 == 1:
            new_line = ['<s>']
            for word in line.split():
                new_line.append(word)
            new_line.append('</s>')
            corpus.append(new_line)
        elif i%3 == 2:
            new_line = ['<p>']
            for word in line.split():
                new_line.append(word)
            new_line.append('</p>')
            corpus.append(new_line)
        else:
            new_line = ['<e>']
            for word in line.split():
                new_line.append(word)
            new_line.append('</e>')
            corpus.append(new_line)
        i=i+1
    return corpus

training_data = 'train.txt'
train_corpus = get_training_corpus(training_data)

def get_test_corpus(file):
    corpus = []
    f = open(file, 'r')
    i = 1
    for line in f:
        if i%3 == 1:
            new_line = ['<s>']
            for word in line.split():
                new_line.append(word)
            new_line.append('</s>')
            corpus.append(new_line)
        elif i%3 == 2:
            new_line = ['<p>']
            for word in line.split():
                new_line.append(word)
            new_line.append('</p>')
            corpus.append(new_line)
        elif i%3 == 0:
            new_line = ['<i>']
            for word in line.split():
                new_line.append(word)
            new_line.append('</i>')
            corpus.append(new_line)
        i=i+1
    return corpus
    
test_data = 'test.txt'
test_corpus = get_test_corpus(test_data)

def partition_traning_data(data, indices):
    traning = []
    valid = []
    indices = set(indices)

    for i in range(int(len(data) / 3)):
        if i in indices:
            traning.append(data[i * 3])
            traning.append(data[i * 3 + 1])
            traning.append(data[i * 3 + 2])
        else:
            valid.append(data[i * 3])
            valid.append(data[i * 3 + 1])
            valid.append(data[i * 3 + 2])
    return [traning, valid]

# 80% traning data size, 20% validating data size
original_data_size = len(train_corpus)
train_size = int(original_data_size * 0.8 / 3.0)
indices = random.sample(range(int(original_data_size / 3)), train_size)
[dev_train_corpus, corpus_valid] = partition_traning_data(train_corpus, indices)

def write_dev_valid_into_file(corpus):
    f = open('dev_valid.csv', 'w')
    w_count = 0
    for j in range(len(corpus)):
        if j % 3 != 2:
            for i in range(1,len(corpus[j])-2):
                f.writelines(str(corpus[j][i]) + ' ')
            f.writelines(str(corpus[j][len(corpus[j])-2]) + '\n')
        else:
            for i in range(1,len(corpus[j])-2):
                f.writelines(str(w_count) + ' ')
                w_count += 1
            f.writelines(str(w_count) + '\n')
            w_count += 1
    f.close()
    return 0

write_dev_valid_into_file(corpus_valid)
test_data_valid = 'dev_valid.csv'
test_corpus_using_validpart = get_test_corpus(test_data_valid)



#calculate #(ti)
def get_labels_unigram(corpus):
    labels = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for label in sentence:
                labels[label] += 1
    return labels

labels_unigram = get_labels_unigram(dev_train_corpus)

#calculate #(ti-1 ti)
def get_labels_bigram(corpus):
    labels = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 1):
                labels[sentence[i], sentence[i+1]] += 1
    return labels
labels_bigram = get_labels_bigram(dev_train_corpus)

#calculate #(ti-2 ti-1 ti)
def get_labels_trigram(corpus):
    labels = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 2):
                labels[sentence[i], sentence[i+1], sentence[i+2]] += 1
    return labels
labels_trigram = get_labels_trigram(dev_train_corpus)


#calculate P(t_i | t_i-1)
def get_transition_bigram_freqs(corpus, unigram):
    transition = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 1):
                transition[sentence[i], sentence[i+1]] += 1
    for key in transition:
        transition[key] = transition[key] * 1.0 / unigram[key[0]]
    return transition

transition_bigram_freqs = get_transition_bigram_freqs(dev_train_corpus, labels_unigram)

#calculate P(t_i| ti-2 ti-1) = #ti-2ti-1ti / #ti-2ti-1
def get_transition_trigram_freqs(corpus, bigram):
    transition = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 2):
                transition[sentence[i], sentence[i+1], sentence[i+2]] += 1
    for key in transition:
        transition[key] = transition[key] * 1.0 / bigram[key[0], key[1]]
    return transition
transition_trigram_freqs = get_transition_trigram_freqs(dev_train_corpus, labels_bigram)

#calculate P(w_i | t_i)
#<label, token>
def get_observation_freqs(corpus, unigram):
    observation = defaultdict(float)
    for j in range(len(corpus)):
        if corpus[j][0] == '<s>':
            for i in range(len(corpus[j])):
                observation[corpus[j+2][i], corpus[j][i]] += 1
    for key in observation:
        observation[key] = (observation[key] * 1.0) / unigram[key[0]]
    return observation
observation_freqs = get_observation_freqs(dev_train_corpus, labels_unigram)

#simple states B-ORG,I-ORG,B-LOC,I-LOC,B-PER,I-PER,B-MISC,I-MISC,O
def get_all_states(unigram):
    t = []
    for key in unigram:
        if key != '<e>':
            t.append(key)
    return t
T = get_all_states(labels_unigram)

#merge indices with label B-xxx with indices with label I-xxx into xxx,indices lists
def merge(b, inner):
    result = []
    c_starter = len(b)
    b = map(eval, b)
    inner = map(eval, inner)
    i = 0
    for j in range(c_starter):
        result.append(str(b[j]))
        if b[j]+1 in inner:
            k = 1
            for current in range(i,len(inner)):
                if inner[current] == b[j] + k:
                    result[j] += '-' + str(b[j] + k)
                    k += 1
                else:
                    i = current
                    break
        else:
            result[j] += '-' + str(b[j])
    return result

#basic viterbi_algorithm use 
#P(label_i | label_i-1) as transition probabilities
#P(w_i | label_i) as lexical generation probabilities
def viterbi_algorithm_bigram(corpus, transition, observation, T):
    lexical_categories = defaultdict(list)
    c = len(T)
    for lines in range(len(corpus)):
        if corpus[lines][0] == '<s>':
            n = len(corpus[lines])-2
            tags = np.zeros((n+1), dtype=np.int16)
            single_line_category = defaultdict(list)
            score = np.zeros((c,n+1), dtype=np.float64)
            bptr = np.zeros((c,n+1), dtype=np.int16)
            #initialize t=1
            for i in range(c):
                a1 = transition['<e>', T[i]]
                a2 = observation[T[i], corpus[lines][1]]
                if a1 == 0.0:
                    a1 = np.random.uniform(low=1e-11,high=1e-10)
                if a2 == 0.0:
                    a2 = np.random.uniform(low=1e-11,high=1e-10)
                score[i][1] = np.log(a1) + np.log(a2)
                bptr[i][1] = 0

            #iteration
            for t in range(2, n+1):
                for i in range(c):
                    a = np.zeros((c), dtype=np.float64)
                    for k in range(c):
                        a1 = transition[T[k], T[i]]
                        if a1 == 0.0:
                            a1 = np.random.uniform(low=1e-11,high=1e-10)
                        a[k] = score[k][t-1] + np.log(a1)
                    k = np.argmax(a)
                    a2 = observation[T[i], corpus[lines][t]]
                    if a2 == 0.0:
                        a2 = np.random.uniform(low=1e-11,high=1e-10)
                    score[i][t] = a[k] + np.log(a2)
                    bptr[i][t] = k
            #finally token w_i has label T[tags[i]] 
            b = np.zeros((c), dtype=np.float64)
            for k in range(c):
                a1 = transition[T[k], '</e>']
                if a1 == 0.0:
                    a1 = np.random.uniform(low=1e-11,high=1e-10)
                b[k] = score[k][n] + np.log(a1)
            tags[n] = np.argmax(b)
            for i in range(n-1,0,-1):
                tags[i] = bptr[tags[i+1]][i+1]            
            #finish to get <label> = index lists
            for i in range(1,n):
                single_line_category[T[tags[i]]].append(corpus[lines+2][i])
            r1 = merge(single_line_category['B-PER'], single_line_category['I-PER'])
            for m in range(len(r1)):
                lexical_categories['PER'].append(r1[m])
            r2 = merge(single_line_category['B-LOC'], single_line_category['I-LOC'])
            for m in range(len(r2)):
                lexical_categories['LOC'].append(r2[m])
            r3 = merge(single_line_category['B-ORG'], single_line_category['I-ORG'])
            for m in range(len(r3)):
                lexical_categories['ORG'].append(r3[m])
            r4 = merge(single_line_category['B-MISC'], single_line_category['I-MISC'])
            for m in range(len(r4)):
                lexical_categories['MISC'].append(r4[m])
    return lexical_categories

test_categories_bi = viterbi_algorithm_bigram(test_corpus, transition_bigram_freqs, observation_freqs, T)
validation_categories_bi = viterbi_algorithm_bigram(test_corpus_using_validpart, transition_bigram_freqs, observation_freqs, T)


def viterbi_algorithm_trigram(corpus, btransition, transition, observation, T):
    lexical_categories = defaultdict(list)
    c = len(T)
    for lines in range(len(corpus)):
        if corpus[lines][0] == '<s>':
            n = len(corpus[lines])-2
            tags = np.zeros((n+1), dtype=np.int16)
            single_line_category = defaultdict(list)
            score = np.zeros((c,c,n+2), dtype=np.float64)
            bptr = np.zeros((c,c,n+2), dtype=np.int16)
            
            #initialize t=1
            for i in range(c):
                for j in range(c):
                    a1 = transition['<e>', T[i]]
                    a2 = observation[T[i], corpus[lines][1]]
                    if a1 == 0.0:
                        a1 = np.random.uniform(low=1e-11,high=1e-10)
                    if a2 == 0.0:
                        a2 = np.random.uniform(low=1e-11,high=1e-10)
                    score[j][i][1] = np.log(a1) + np.log(a2)
                    bptr[j][i][1] = 0

            #iteration
            for t in range(2, n+1):
                for i in range(c):
                    for j in range(c):
                        temp = np.zeros((c), dtype=np.float64)
                        for w in range(c):
                            a1 = transition[T[w], T[j], T[i]]
                            if a1 == 0.0:
                                a1 = np.random.uniform(low=1e-11,high=1e-10)
                            temp[w] = score[w][j][t-1] + np.log(a1)
                        w = np.argmax(temp)
                        a2 = observation[T[i], corpus[lines][t]]
                        if a2 == 0.0:
                            a2 = np.random.uniform(low=1e-11,high=1e-10)
                        score[j][i][t] = temp[w] + np.log(a2)
                        bptr[j][i][t] = w

            a = np.zeros((c), dtype=np.float64)
            index = np.zeros((c), dtype=np.int16)
            for i in range(c):
                b = np.zeros((c), dtype=np.float64)
                for j in range(c):
                    a1 = transition[T[j], T[i], '</e>']
                    if a1 == 0.0:
                        a1 = np.random.uniform(low=1e-11,high=1e-10)
                    b[j] = score[j][i][n] + np.log(a1)
                j = np.argmax(b)
                index[i] = j
                a[i] = b[j]
            i = np.argmax(a)
            j = index[i]
            tags[n] = i
            tags[n-1] = j

            for k in range(n-2,0,-1):
                tags[k] = bptr[tags[k+1]][tags[k+2]][k+2]
          
            #finish to get <label> = index lists
            for i in range(1,n+1):
                single_line_category[T[tags[i]]].append(corpus[lines+2][i])
            r1 = merge(single_line_category['B-PER'], single_line_category['I-PER'])
            for m in range(len(r1)):
                lexical_categories['PER'].append(r1[m])
            r2 = merge(single_line_category['B-LOC'], single_line_category['I-LOC'])
            for m in range(len(r2)):
                lexical_categories['LOC'].append(r2[m])
            r3 = merge(single_line_category['B-ORG'], single_line_category['I-ORG'])
            for m in range(len(r3)):
                lexical_categories['ORG'].append(r3[m])
            r4 = merge(single_line_category['B-MISC'], single_line_category['I-MISC'])
            for m in range(len(r4)):
                lexical_categories['MISC'].append(r4[m])
    return lexical_categories

#test_categories_tri = viterbi_algorithm_trigram(test_corpus, transition_bigram_freqs, transition_trigram_freqs, observation_freqs, T)
#validation_categories_tri = viterbi_algorithm_trigram(test_corpus_using_validpart, transition_bigram_freqs, transition_trigram_freqs, observation_freqs, T)
#normalize output 
#Type,Prediction
#PER,......
#LOC,......
#ORG,......
#MISC,.....
def output_test_result(result, name):
    filename = name + '.csv'
    f = open(filename, 'w')
    f.writelines('Type,Prediction\n')
    str1 = 'PER,'
    for index in range(len(result['PER'])):
        if result['PER'][index] != '':
            str1 += str(result['PER'][index]) + ' '
    str1 += '\n'
    f.writelines(str1)
    str2 = 'LOC,'
    for index in range(len(result['LOC'])):
        if result['LOC'][index] != '':
            str2 += str(result['LOC'][index]) + ' '
    str2 += '\n'
    f.writelines(str2)
    str3 = 'ORG,'
    for index in range(len(result['ORG'])):
        if result['ORG'][index] != '':
            str3 += str(result['ORG'][index]) + ' '
    str3 += '\n'
    f.writelines(str3)
    str4 = 'MISC,'
    for index in range(len(result['MISC'])):
        if result['MISC'][index] != '':
            str4 += str(result['MISC'][index]) + ' '
    str4 += '\n'
    f.writelines(str4)
    f.close()

output_test_result(test_categories_bi, 'bigram')
#output_test_result(test_categories_tri, 'trigram')

def get_standard_corpus(corpus):
    dev_corpus = []
    for sentence in corpus:
        new_line = []
        for i in range(1,len(sentence)-1):
            new_line.append(str(sentence[i]))
        dev_corpus.append(new_line)
    return dev_corpus
standard_corpus = get_standard_corpus(corpus_valid)

def normalize_standard(corpus):
    lexical_categories = defaultdict(list)
    count = 0
    for j in range(len(corpus)):
        if j%3 == 2:
            single_line_category = defaultdict(list)
            for i in range(len(corpus[j])):
                single_line_category[corpus[j][i]].append(str(count))
                count += 1
            r1 = merge(single_line_category['B-PER'], single_line_category['I-PER'])
            for m in range(len(r1)):
                lexical_categories['PER'].append(r1[m])
            r2 = merge(single_line_category['B-LOC'], single_line_category['I-LOC'])
            for m in range(len(r2)):
                lexical_categories['LOC'].append(r2[m])
            r3 = merge(single_line_category['B-ORG'], single_line_category['I-ORG'])
            for m in range(len(r3)):
                lexical_categories['ORG'].append(r3[m])
            r4 = merge(single_line_category['B-MISC'], single_line_category['I-MISC'])
            for m in range(len(r4)):
                lexical_categories['MISC'].append(r4[m])
    return lexical_categories
dev_standard_corpus = normalize_standard(standard_corpus)

def compare(c1, c2):
    bound1 = len(c1['PER'])
    bound2 = len(c2['PER'])
    p1 = 0
    p2 = 0
    intersection = 0
    P = 0
    C = 0
    while p1<bound1 and p2<bound2:
        str1 = c1['PER'][p1]
        str2 = c2['PER'][p2]
        if str1 == str2:
            intersection += 1
            p1 += 1
            p2 += 1
            P += 1
            C += 1
        else:
            num1 = map(eval, str1.split('-'))
            num2 = map(eval, str2.split('-'))
            if num1 > num2:
                p2 += 1
                P += 1
            elif num1 < num2:
                p1 += 1
                C += 1
            else:
                p1 += 1
                p2 += 1
                P += 1
                C += 1
    if p1 == bound1 and p2 <bound2:
        while(p2 < bound2):
            P += 1
            p2 += 1
    elif p1 < bound1 and p2 == bound2:
        while (p1 < bound1):
            C += 1
            p1 += 1
    bound1 = len(c1['LOC'])
    bound2 = len(c2['LOC'])
    p1 = 0
    p2 = 0
    while p1<bound1 and p2<bound2:
        str1 = c1['LOC'][p1]
        str2 = c2['LOC'][p2]
        if str1 == str2:
            intersection += 1
            p1 += 1
            p2 += 1
            P += 1
            C += 1
        else:
            num1 = map(eval, str1.split('-'))
            num2 = map(eval, str2.split('-'))
            if num1 > num2:
                p2 += 1
                P += 1
            elif num1 < num2:
                p1 += 1
                C += 1
            else:
                p1 += 1
                p2 += 1
                P += 1
                C += 1
    if p1 == bound1 and p2 <bound2:
        while(p2 < bound2):
            P += 1
            p2 += 1
    elif p1 < bound1 and p2 == bound2:
        while (p1 < bound1):
            C += 1
            p1 += 1
    bound1 = len(c1['ORG'])
    bound2 = len(c2['ORG'])
    p1 = 0
    p2 = 0
    while p1<bound1 and p2<bound2:
        str1 = c1['ORG'][p1]
        str2 = c2['ORG'][p2]
        if str1 == str2:
            intersection += 1
            p1 += 1
            p2 += 1
            P += 1
            C += 1
        else:
            num1 = map(eval, str1.split('-'))
            num2 = map(eval, str2.split('-'))
            if num1 > num2:
                p2 += 1
                P += 1
            elif num1 < num2:
                p1 += 1
                C += 1
            else:
                p1 += 1
                p2 += 1
                P += 1
                C += 1
    if p1 == bound1 and p2 <bound2:
        while(p2 < bound2):
            P += 1
            p2 += 1
    elif p1 < bound1 and p2 == bound2:
        while (p1 < bound1):
            C += 1
            p1 += 1
    bound1 = len(c1['MISC'])
    bound2 = len(c2['MISC'])
    p1 = 0
    p2 = 0
    while p1<bound1 and p2<bound2:
        str1 = c1['MISC'][p1]
        str2 = c2['MISC'][p2]
        if str1 == str2:
            intersection += 1
            p1 += 1
            p2 += 1
            P += 1
            C += 1
        else:
            num1 = map(eval, str1.split('-'))
            num2 = map(eval, str2.split('-'))
            if num1 > num2:
                p2 += 1
                P += 1
            elif num1 < num2:
                p1 += 1
                C += 1
            else:
                p1 += 1
                p2 += 1
                P += 1
                C += 1
    if p1 == bound1 and p2 <bound2:
        while(p2 < bound2):
            P += 1
            p2 += 1
    elif p1 < bound1 and p2 == bound2:
        while (p1 < bound1):
            C += 1
            p1 += 1
    precise = intersection * 1.0 / P
    recall = intersection * 1.0 / C
    F = 2.0 * precise * recall / (precise + recall)
    return [precise, recall, F]

[p_bi, r_bi, F_bi] = compare(dev_standard_corpus, validation_categories_bi)
#[p_tri, r_tri, F_tri] = compare(dev_standard_corpus, validation_categories_tri)
print [p_bi, r_bi, F_bi]
#print [p_tri, r_tri, F_tri]


