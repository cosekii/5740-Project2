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
t_corpus = get_training_corpus(training_data)

def partition_training_data(data, indices):
    training = []
    valid = []
    indices = set(indices)
    
    for i in range(int(len(data) / 3)):
        if i in indices:
            training.append(data[i * 3])
            training.append(data[i * 3 + 1])
            training.append(data[i * 3 + 2])
        
        else:
            valid.append(data[i * 3])
            valid.append(data[i * 3 + 1])
            valid.append(data[i * 3 + 2])

return [training, valid]

# 80% training data size, 20% validating data size
original_data_size = len(t_corpus)
train_size = int(original_data_size * 0.8 / 3.0)
indices = random.sample(range(int(original_data_size / 3)), train_size)
[corpus_train, corpus_valid] = partition_training_data(t_corpus, indices)



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

def get_corpus_with_unknown(corpus):
    unk = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<s>':
            for word in sentence:
                unk[word] += 1
    removedlist = set([key for key in unk if unk[key] == 1])

    for j in range(len(corpus)):
        if corpus[j][0] == '<s>':
            for i in range(len(corpus[j])):
                if corpus[j][i] in removedlist:
                    corpus[j][i] = '<unk>'
    return corpus

dev_train_corpus = get_corpus_with_unknown(t_corpus)
dev_test_corpus = get_corpus_with_unknown(test_corpus)



#lexicon <token, label> = #occurrence
#the most basic baseline system
def generate_baseline_NE_lexicon(corpus):
    lexicon = defaultdict(float)
    c_sentences = len(corpus)/3;
    for j in range(c_sentences):
        c_words = len(corpus[3*j+2])
        for i in range(c_words):
            if corpus[3*j+2][i] != 'O':
                lexicon[corpus[3*j][i], corpus[3*j+2][i]] += 1
    return lexicon

#baseline_NE_lexicon = generate_baseline_NE_lexicon(t_corpus)

#calculate #(t_i)
def get_labels_unigram(corpus):
    labels = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for label in sentence:
                labels[label] += 1
    return labels

labels_unigram = get_labels_unigram(dev_train_corpus)

def get_labels_bigram(corpus):
    labels = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 1):
                labels[sentence[i], sentence[i+1]] += 1
    return labels
labels_bigram = get_labels_bigram(dev_train_corpus)

def get_labels_trigram(corpus):
    labels = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 2):
                labels[sentence[i], sentence[i+1], sentence[i+2]] += 1
    return labels
labels_trigram = get_labels_trigram(dev_train_corpus)


#calculate P(t_i+1 | t_i)
def get_transition_bigram_freqs_addk(corpus, unigram, k):
    transition = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 1):
                transition[sentence[i], sentence[i+1]] += 1
    V = len(unigram)
    for key in transition:
        transition[key] = (transition[key] * 1.0 + k) / (unigram[key[0]] + k * V)
    return transition

k = 0.01
transition_bigram_freqs_addk = get_transition_bigram_freqs_addk(dev_train_corpus, labels_unigram, k)

#p(v|wu)=#wuv/#wu
def get_transition_trigram_freqs_addk(corpus, bigram, k):
    transition = defaultdict(float)
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 2):
                transition[sentence[i], sentence[i+1], sentence[i+2]] += 1
    V = len(bigram)
    for key in transition:
        transition[key] = (transition[key] * 1.0 + k) / (bigram[key[0], key[1]] + V * k)
    return transition
transition_trigram_freqs_addk = get_transition_trigram_freqs_addk(dev_train_corpus, labels_bigram, k)

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
#P(label_i+1 | label_i) as transition probabilities
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
                score[i][1] = transition['<e>', T[i]] * observation[T[i], corpus[lines][1]]
                bptr[i][1] = 0
            #iteration
            for t in range(2, n+1):
                for i in range(c):
                    a = np.zeros((c), dtype=np.float64)
                    for k in range(c):
                        a[k] = score[k][t-1]*transition[T[k], T[i]]
                    k = np.argmax(a)
                    score[i][t] = score[k][t-1] * transition[T[k], T[i]] * observation[T[i], corpus[lines][t]]
                    bptr[i][t] = k
            #finally token w_i has label T[tags[i]] 
            b = np.zeros((c), dtype=np.float64)
            for k in range(c):
                b[k] = score[k][n] * transition[T[k], '</e>']
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

lexical_categories_bi = viterbi_algorithm_bigram(dev_test_corpus, transition_bigram_freqs_addk, observation_freqs, T)


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
                    score[j][i][1] = btransition['<e>', T[i]] * observation[T[i], corpus[lines][1]]
                    bptr[j][i][1] = 0

            #iteration
            for t in range(2, n+1):
                for i in range(c):
                    for j in range(c):
                        temp = np.zeros((c), dtype=np.float64)
                        for w in range(c):
                            temp[w] = score[w][j][t-1] * transition[T[w], T[j], T[i]]
                        w = np.argmax(temp)
                        score[j][i][t] = score[w][j][t-1] * transition[T[w], T[j], T[i]] * observation[T[i], corpus[lines][t]]
                        bptr[j][i][t] = w

            a = np.zeros((c), dtype=np.float64)
            index = np.zeros((c), dtype=np.int16)
            for i in range(c):
                b = np.zeros((c), dtype=np.float64)
                for j in range(c):
                    b[j] = score[j][i][n] * transition[T[j], T[i], '</e>']
                j = np.argmax(b)
                index[i] = j
                a[i] = score[j][i][n] * transition[T[j], T[i], '</e>']
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

lexical_categories_tri = viterbi_algorithm_trigram(dev_test_corpus, transition_bigram_freqs_addk, transition_trigram_freqs_addk, observation_freqs, T)
#normalize output 
#Type,Prediction
#PER,......
#LOC,......
#ORG,......
#MISC,.....
def output_test_result(result):
    f = open('smoothaddkbigram.csv', 'w')
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

output_test_result(lexical_categories_bi)

def output_test_result_b(result):
    f = open('smoothaddktrigram.csv', 'w')
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
output_test_result_b(lexical_categories_tri)
