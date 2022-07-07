import argparse
import zipfile
import re
import collections
import numpy as np
from six.moves import xrange
import random
import torch
import timeit
from torch.autograd import Variable
from models import SkipGramModel
from models import CBOWModel
from inference import save_embeddings

model_list = ['CBOW', 'skipgram']

skipgram_data = {}

cmd_parser = argparse.ArgumentParser(description=None)
# Data arguments
#cmd_parser.add_argument('-d', '--data', default='data/text8.zip',
#                        help='Data file for word2vec training.')
cmd_parser.add_argument('-d', '--data', default='sample.txt',
                        help='Data file for word2vec training.')
cmd_parser.add_argument('-o', '--output', default='embeddings.bin',
                        help='Output embeddings filename.')
cmd_parser.add_argument('-p', '--plot', default='tsne.png',
                        help='Plotting output filename.')
cmd_parser.add_argument('-pn', '--plot_num', default=100, type=int,
                        help='Plotting data number.')
cmd_parser.add_argument('-s', '--size', default=50000, type=int,
                        help='Vocabulary size.')
# Model training arguments
cmd_parser.add_argument('-m', '--mode', default='CBOW', choices=model_list,
                        help='Training model.')

# Default 128
cmd_parser.add_argument('-bs', '--batch_size', default=4   , type=int,
                        help='Training batch size.')

# Default 2
cmd_parser.add_argument('-ns', '--num_skips', default=2, type=int,
                        help='How many times to reuse an input to generate a label.')
cmd_parser.add_argument('-sw', '--skip_window', default=1, type=int,
                        help='How many words to consider left and right.')
# Default = 128
cmd_parser.add_argument('-ed', '--embedding_dim', default=3, type=int,
                        help='Dimension of the embedding vector.')
cmd_parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate')

# Default = 10000
cmd_parser.add_argument('-i', '--num_steps', default=3, type=int,
                        help='Number of steps to run.')
cmd_parser.add_argument('-ne', '--negative_example', default=5, type=int,
                        help='Number of negative examples.')
cmd_parser.add_argument('-c', '--clip', default=1.0, type=float,
                        help='Clip gradient norm value.')

# Device
cmd_parser.add_argument('-dc', '--disable_cuda', default=False, action='store_true',
                        help='Explicitly disable cuda and GPU.')

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename) as f:
            text = f.read(f.namelist()[0]).decode('ascii')
    else:
        with open(filename, "r") as f:
            text = f.read()
    return [word.lower() for word in re.compile('\w+').findall(text)]

def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
        Returns:
            data        list of codes (integers from 0 to vocabulary_size-1).
                        This is the original text but words are replaced by their codes
            count       list of words(strings) to count of occurrences
            dictionary  map of words(strings) to their codes(integers)
            reverse_dictionary  maps codes(integers) to words(strings)
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def generate_batch(device, data, data_index, batch_size, num_skips, skip_window):
    """Generates a batch of training data
        returns:
            centers:      a list of center word indexes for this batch.
            contexts:     a list of contexts indexes for this batch.
            data_index: current data index for next batch.
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    centers = np.ndarray(shape=(batch_size), dtype=np.int32)
    print("1. Centers: ", centers)
    contexts = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    print("1. Contexts: ", contexts)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        print("Context_words: ", context_words)
        words_to_use = random.sample(context_words, num_skips)
        print("words_to_use", words_to_use)
        for j, context_word in enumerate(words_to_use):
            centers[i * num_skips + j] = buffer[skip_window]
            contexts[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    centers = torch.LongTensor(centers).to(device)
    contexts = torch.LongTensor(contexts).to(device)
    print("2. Centers: ", centers)
    print("2. Contexts: ", contexts)
    return centers, contexts, data_index

def get_deivice(disable_cuda):
    """Get CPU/GPU device
    """
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def train(device, data, word_count, mode, vocabulary_size, embedding_dim, batch_size,
          num_skips, skip_window, num_steps, learning_rate, neg_num, clip):
    """Training and backpropagation process, returns final embedding as result"""
    if mode == 'CBOW':
        model = CBOWModel(device, vocabulary_size, embedding_dim)
    elif mode == 'skipgram':
        model = SkipGramModel(device, vocabulary_size, embedding_dim, neg_num, word_count)
    else:
        raise ValueError("Model \"%s\" not supported")
    model.to(device)
    print("** Model **: ", model)
    print("Start training on device:", device)
    print("** Model Parameters **: ", model.get_embeddings())


    p = 0
    
    for param in model.parameters(): 
        p = p + 1
        print("Param: " + str(p), param.data)



    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    loss_function = torch.nn.NLLLoss() # Negative Log Likelihood Loss #^^
    data_index = 0
    loss_val = 0
    for i in xrange(num_steps):
        # prepare feed data and forward pass
        print()
        print("Num_Steps: ", i)
        epoch = {}
        skipgram_data["epoch" + str(i + 1)] = epoch
        currepoch = skipgram_data["epoch" + str(i + 1)]
        centers, contexts, data_index = generate_batch(device, data, data_index,
                                                       batch_size, num_skips, skip_window)
        print("Main Contexts: ", contexts)
        print("Main Centers: ", centers)
        if mode == 'CBOW':
            res = model(contexts)
            epoch = res[0]
            y_pred = res[1]
            


            currepoch = epoch

            currepoch["y_pred"] = str(y_pred)
            print("Main y_pred: ", y_pred)
            print("Centers ", centers)
            loss = loss_function(y_pred, centers)
            print("Main loss: ", loss)
            currepoch["loss"] = str(loss)
            skipgram_data["epoch" + str(i + 1)] = currepoch

        elif mode == 'skipgram':
            loss = model(centers, contexts)
        else:
            raise ValueError("Model \"%s\" not supported" % model)
        # Zero gradients, perform a backward pass, and update the weights.
        # zero_grad() clears old gradients from the last step
        print("optimizer.zero_grad(): ", optimizer.zero_grad())
        loss.backward() #computes the derivatives of the loss w.r.t the parameters or anything using back propogaiton
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        print("optimizer.step(): ", optimizer.step())
        # Print loss value at certain step
        print("loss.item(): ", loss.item()) #Epoch loss of batch during training of a batch
        loss_val += loss.item()
        print("loss_val: ", loss_val)
        print("i: ", i)
        print("num_steps: ", num_steps)
        print("i % (num_steps/100): ", i % (num_steps/100))
        if i > 0 and i % (num_steps/100) == 0:
            print('  Average loss at step', i, ':', loss_val/(num_steps/100))
            loss_val = 0

    return model.get_embeddings()


def tsne_plot(embeddings, num, reverse_dictionary, filename):
    """Plot tSNE result of embeddings for a subset of words"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to plot embeddings.')
        print(ex)
        return
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(final_embeddings[:num, :])
    low_dim_labels = [reverse_dictionary[i] for i in xrange(num)]
    assert low_dim_embs.shape[0] >= len(low_dim_labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(low_dim_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    print("Saving plot to:", filename)
    plt.savefig(filename)

def getData():
    print(skipgram_data)
    return skipgram_data

#if __name__ == "__main__"

def start():
    print("main ran")
    args = cmd_parser.parse_args()
    dev = get_deivice(args.disable_cuda)
    # Data preprocessing
    vocabulary = read_data(args.data)
    print("Vocabulary: ", vocabulary)
    print('Data size', len(vocabulary))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, args.size)
    print("Dictionary: ", dictionary)
    print("Reverse_Dictionary: ", reverse_dictionary)
    print("Data: ", data)
    print("args.size: ", args.size)
    print("count: ", count)
    vocabulary_size = min(args.size, len(count))
    print(vocabulary_size)
    print('Vocabulary size', vocabulary_size)
    word_count = [ c[1] for c in count]
    # Model training
    start_time = timeit.default_timer()
    final_embeddings = train(device=dev,
                             data=data,
                             word_count=word_count,
                             mode=args.mode,
                             vocabulary_size=vocabulary_size,
                             embedding_dim=args.embedding_dim,
                             batch_size=args.batch_size,
                             num_skips=args.num_skips,
                             skip_window=args.skip_window,
                             num_steps=args.num_steps,
                             learning_rate=args.learning_rate,
                             clip=args.clip,
                             neg_num=args.negative_example)
    print('Training time:', timeit.default_timer() - start_time, 'Seconds')
    getData()

    #norm = torch.sqrt(torch.cumsum(torch.mul(final_embeddings, final_embeddings), 1))
    #nomalized_embeddings = (final_embeddings/norm).cpu().numpy()
    # Save result and plotting
    #save_embeddings(args.output, final_embeddings, dictionary)
    #tsne_plot(embeddings=nomalized_embeddings,
    #          num=min(vocabulary_size, args.plot_num),
              #reverse_dictionary=reverse_dictionary,
              #filename=args.plot)
