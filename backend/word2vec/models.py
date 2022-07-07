import torch
import torch.nn.functional as F
import numpy as np

TABLE_SIZE = 1e8

dataCBOW = {}

def create_sample_table(word_count):
    """ Create negative sample table for vocabulary, words with
        higher frequency will have higher occurrences in table.
    """
    table = []
    frequency = np.power(np.array(word_count), 0.75) # would this output one number always?
    sum_frequency = sum(frequency)
    ratio = frequency / sum_frequency
    count = np.round(ratio * TABLE_SIZE) #ratio is an array of ratios, count is a list
    for word_idx, c in enumerate(count): #count is a list
        table += [word_idx] * int(c)
    return np.array(table)

class SkipGramModel(torch.nn.Module):
    """ Center word as input, context words as target.
        Objective is to maximize the score of map from input to target.
    """
    def __init__(self, device, vocabulary_size, embedding_dim, neg_num=0, word_count=[]):
        super(SkipGramModel, self).__init__()
        self.device = device
        self.neg_num = neg_num
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim) #embedding is vocabulary size not dictionary size do doesn't include UNK
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange) #0.5 / 3
        if self.neg_num > 0:
            self.table = create_sample_table(word_count)

    def forward(self, centers, contexts):
        batch_size = len(centers)
        u_embeds = self.embeddings(centers).view(batch_size,1,-1) #Makes embeddings for input words?
        v_embeds = self.embeddings(contexts).view(batch_size,1,-1) #Makes embeddings for target words?
        score  = torch.bmm(u_embeds, v_embeds.transpose(1,2)).squeeze()
        loss = F.logsigmoid(score).squeeze()
        if self.neg_num > 0:
            neg_contexts = torch.LongTensor(np.random.choice(self.table, size=(batch_size, self.neg_num))).to(self.device)
            neg_v_embeds = self.embeddings(neg_contexts)
            neg_score = torch.bmm(u_embeds, neg_v_embeds.transpose(1,2)).squeeze()
            neg_score = torch.sum(neg_score, dim=1)
            neg_score = F.logsigmoid(-1*neg_score).squeeze()
            loss += neg_score
        return -1 * loss.sum()

    def get_embeddings(self):
        return self.embeddings.weight.data

class CBOWModel(torch.nn.Module):
    """ Context words as input, returns possiblity distribution
        prediction of center word (target).
    """
    def __init__(self, device, vocabulary_size, embedding_dim):
        #print("device:" + device)
        print("vocabulary:" + str(vocabulary_size))
        print("embedding_dim:" + str(embedding_dim))

        super(CBOWModel, self).__init__()
        self.device = device
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        print("self.embeddings: " + str(self.embeddings))
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        print("self.embeddings.weight.data.uniform_(-initrange, initrange): " + str(self.embeddings.weight.data.uniform_(-initrange, initrange)))
        self.linear1 = torch.nn.Linear(embedding_dim, vocabulary_size) #Linear equation, Skipgram does not have, linear transformation y = xW^T + b x -> embedding_dim in_feature, y -> vocab size out_feature
        print("self_linear1:" + str(self.linear1))
       


    def forward(self, contexts):
        dataCBOW = {}
        print("forward contexts: " + str(contexts))
        print("CBOW forward called")
        # input
        embeds = self.embeddings(contexts) #First it takes the context vectors as context and creates embeddings
        print("embeds context : " + str(embeds))

        # projection
        add_embeds = torch.sum(embeds, dim=1) #sums each row in the embeds with dimension 1
        print("add_embeds:" + str(add_embeds)) #adds right weights
        print("add_embeds shape: " + str(add_embeds.shape)) #adds right weigh
        dataCBOW["rightWeight"] = str(add_embeds)
        # output
        print("Linear Weigth: ",self.linear1.weight)
        print("Linear Bias: ", self.linear1.bias)

        dataCBOW["leftWeight"] = str(self.linear1.weight)
        dataCBOW["leftBias"] = str(self.linear1.bias)
        out = self.linear1(add_embeds) #adds left weight
        print("Model out: "+ str(out))
        dataCBOW["modelOut"] = str(out)
        print("output shape: " + str(out.shape))
        log_probs = F.log_softmax(out, dim=1)
        print("log_probs:" +str(log_probs))
        dataCBOW["log_probs"] = str(log_probs)
        print("log prob shape: " + str(log_probs.shape))
        print("EPOCH ",  dataCBOW)
        res = []
        res.append(dataCBOW)
        res.append(log_probs)



        return res



    def get_embeddings(self):
        return self.embeddings.weight.data


