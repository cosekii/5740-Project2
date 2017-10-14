Readme for NLP Project 2

how to run: get_training_corpus, get_test_corpus, get_labels_unigram, get_labels_bigram, get_transition_bigram_freqs, get_observation_freqs, get_all_states, viterbi_algorithm_bigram, merge, output_test_result.
(e.g. use bigram HMM without validation part)

functions:
1.get_training_corpus(file): preprocess training data "train.txt" adding start and end tokens to each line separately. 
e.g. <s>Sentence</s>, <p>POS</p>, <e>Labels</e>

2.get_test_corpus(file): preprocess testing data "test.txt" adding start and end tokens to each line separately.
e.g. <s>Sentence</s>, <p>POS</p>, <i>Indices</i>

3.partition_traning_data(data, indices): part 80% of "train.txt" as training data and the remaining as validation part.

4.write_valid_into_file(corpus): write validation part with labels into files.

5.write_dev_valid_into_file(corpus): write validation part without labels into files.

6.accuracy(c1, c2): compute the accuracy of our model when using validation part without labels as test data and validation part with labels as standard.

7.get_labels_unigram(corpus): calculate #(ti).

8.get_labels_bigram(corpus): calculate #(ti-1ti).

9.get_labels_trigram(corpus): calculate #(ti-2ti-1ti).

10.get_transition_bigram_freqs(corpus, unigram, k): calculate bigram transition probabilities P(t_i | t_i-1).

11.get_transition_trigram_freqs(corpus, bigram, k): calculate trigram transition probabilities P(t_i | t_i-2t_i-1).

12.get_observation_freqs(corpus, unigram): calculate observation probabilities P(w_i | t_i).

13.get_all_states(unigram): get all states of our HMM model.

14.merge(b, inner): normalize our output.

15.viterbi_algorithm_bigram(corpus, transition, observation, T):  viterbi algorithm using bigram.

16.viterbi_algorithm_trigram(corpus, btransition, transition, observation, T): viterbi algorithm using trigram.

17&18.output_test_result(result): write result into file

19.viterbi_algorithm_trigram_test(corpus, btransition, transition, observation, T): label data of validation part using viterbi algorithm with trigram


