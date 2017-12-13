from utils import Vocab, get_dataset, data_iterator, sample
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import sequence_loss
import time
import os
from copy import deepcopy

class Config(object):
    batch_size = 64
    embed_size = 50
    hidden_size = 100
    num_steps = 10
    max_epochs = 16
    early_stopping = 2
    dropout = 0.9
    lr = 0.001
    
class GRULM(object):
    def __init__(self, config=Config()):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.gru_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projection(self.gru_outputs)
        
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        output = tf.reshape(tf.concat(self.outputs, 1), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_operation(output)
        self.train_step = self.add_training_operation(self.calculate_loss)
        
    def load_data(self):
        self.vocab = Vocab()
        self.vocab.construct(get_dataset('train'))
        self.encoded_train = np.array([self.vocab.encode(word) for word in get_dataset('train')],
                                       dtype=np.int32)
        self.encoded_valid = np.array([self.vocab.encode(word) for word in get_dataset('valid')],
                                       dtype=np.int32)
    
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32,
                                shape = [None, self.config.num_steps], name='input')
        self.label_placeholder = tf.placeholder(tf.int32,
                                shape = [None, self.config.num_steps], name='target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='drouput')
        
    def add_embedding(self):
        embedding = tf.get_variable('embedding', [len(self.vocab),
                                    self.config.embed_size], trainable=True)
        embedded_input = tf.nn.embedding_lookup(embedding, self.input_placeholder)
        embedded_input = [tf.squeeze(x, [1]) for x in tf.split(
                embedded_input,self.config.num_steps,1)]
        return embedded_input
    
    def add_model(self, inputs):
        with tf.variable_scope('input_dropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]
            
        with tf.variable_scope('gru_layers') as scope:
            self.initial_state = tf.zeros([self.config.batch_size,
                                           self.config.hidden_size])
            state = self.initial_state
            gru_outputs = []
            for tstep,current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                # Update Gate
                gru_W_z = tf.get_variable('gru_W_z', [self.config.embed_size,
                                        self.config.hidden_size])
                gru_U_z = tf.get_variable('gru_U_z', [self.config.hidden_size,
                                        self.config.hidden_size])
                gru_b_z = tf.get_variable('gru_b_z', [self.config.hidden_size])
                gru_z_output = tf.nn.sigmoid(
                        tf.matmul(current_input,gru_W_z) + tf.matmul(state,gru_U_z) + gru_b_z)
                
                # Reset Gate
                gru_W_r = tf.get_variable('gru_W_r', [self.config.embed_size,
                                        self.config.hidden_size])
                gru_U_r = tf.get_variable('gru_U_r', [self.config.hidden_size,
                                        self.config.hidden_size])
                gru_b_r = tf.get_variable('gru_b_r', [self.config.hidden_size])
                gru_r_output = tf.nn.sigmoid(
                        tf.matmul(current_input,gru_W_r) + tf.matmul(state,gru_U_r) + gru_b_r)
                
                # New Memory Gate
                gru_W_nm = tf.get_variable('gru_W_nm', [self.config.embed_size,
                                        self.config.hidden_size])
                gru_U_nm = tf.get_variable('gru_U_nm', [self.config.hidden_size,
                                        self.config.hidden_size])
                gru_b_nm = tf.get_variable('gru_b_nm', [self.config.hidden_size])
                gru_nm_output = tf.nn.tanh(
                        tf.matmul(current_input,gru_W_nm) + tf.multiply(gru_r_output, tf.matmul(state,gru_U_nm) + gru_b_nm))
                
                # Final Memory Gate
                state = tf.multiply(gru_z_output,state) + tf.multiply((1-gru_z_output),gru_nm_output)
                
                gru_outputs.append(state)
            self.final_state = gru_outputs[-1]
        
        with tf.variable_scope('gru_dropout'):
            gru_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in gru_outputs]
        
        return gru_outputs
        
    def add_projection(self, gru_outputs):
        with tf.variable_scope('projection'):
            proj_U = tf.get_variable('U_matrix', [self.config.hidden_size,
                                                  len(self.vocab)])
            proj_b = tf.get_variable('b_vector', [len(self.vocab)])
            outputs = [tf.matmul(h,proj_U) + proj_b for h in gru_outputs]
        return outputs
        
    def add_loss_operation(self, output):
        all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
        cross_entropy = sequence_loss([output], [tf.reshape(
                self.label_placeholder, [-1])], all_ones, len(self.vocab))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss
    
    def add_training_operation(self, loss):
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            train_op = optimizer.minimize(loss)
        return train_op
    
    def run_epoch(self, sess, data, train_op=None, verbose=10):
        dropout = self.config.dropout
        if train_op == None:
            train_op = tf.no_op()
            dropout = 1
        total_steps = sum(1 for x in data_iterator(data, self.config.batch_size, self.config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x,y) in enumerate(data_iterator(data, self.config.batch_size, self.config.num_steps)):
            feed = {self.input_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: dropout,
                    self.initial_state: state}
            loss, state, _ = sess.run(
                    [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step%verbose == 0:
                sys.stdout.write('\r{} / {} : Avg. Loss = {}'.format(
                        step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss)
            
def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None,):
    state = model.initial_state.eval()
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    for i in range(stop_length):
        feed = {model.input_placeholder: [tokens[-1:]],
                model.initial_state: state,
                model.dropout_placeholder: 1}
        state, y_pred = session.run([model.final_state,
                                     model.predictions[-1]], feed_dict=feed)
        next_word_idx = sample(y_pred[0])
        tokens.append(next_word_idx)
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
            break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output

def generate_sentence(session, model, config, starting_text=None):
    if starting_text==None:
        starting_text = input()
    while starting_text:
        print(' '.join(generate_text(session, model, config, starting_text, stop_tokens=['<eos>'])))
        starting_text = input()
        
if __name__ == '__main__':
    config = Config()
    gen_config = deepcopy(config)
    gen_config.batch_size = gen_config.num_steps = 1
    with tf.variable_scope('GRU') as scope:
        model = GRULM(config)
        scope.reuse_variables()
        gen_model = GRULM(gen_config)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
        
    with tf.Session() as session:
        best_val_loss = float('inf')
        best_val_epoch = 0
        session.run(init)
        
        if 'checkpoint' in os.listdir('weights'):
            saver.restore(session, 'weights/gru.weights')
            
        else:
            for epoch in range(config.max_epochs):
                start = time.time()
                print('Epoch {}'.format(epoch))
                train_loss = model.run_epoch(session, model.encoded_train, model.train_step)
                val_loss = model.run_epoch(session, model.encoded_valid)
                print('Training loss: {}'.format(train_loss))
                print('Validation loss: {}'.format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    saver.save(session, 'weights/gru.weights')
                if epoch - best_val_epoch > config.early_stopping:
                    break
                print('Total time: {}'.format(time.time() - start))
            
        starting_text = 'in palo alto'
        generate_sentence(session, gen_model, gen_config, starting_text)