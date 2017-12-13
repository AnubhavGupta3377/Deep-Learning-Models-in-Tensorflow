import numpy as np

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.unknown = '<unk>'
        self.add_word(self.unknown)
        
    def add_word(self, word):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            
    def construct(self, words):
        for word in words:
            self.add_word(word)
        print ('Constructed vocabulary with size: {}'.format(len(self.index_to_word)))
        
    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]
    
    def decode(self, index):
        return self.index_to_word[index]
    
    def __len__(self):
        return len(self.word_to_index)
    
def get_dataset(typeofdata):
    datafile='data/ptb.{}.txt'.format(typeofdata)
    with open(datafile) as data:
        for line in data:
            for word in line.split():
                yield word
            yield '<eos>'
            
def data_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)
        
def sample(a):
    a = np.log(a)
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))