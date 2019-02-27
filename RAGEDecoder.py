# coding=UTF-8
__author__ = 'shiqian.csq'
import tensorflow as tf
from rage.Unit import *
from rage.contrib.seq2seq.decoder import *
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from collections import namedtuple
from rage.RAGEEncoder import ConvEncoder, ConvEncoderOutput


class ConvDecoderOutput(
    namedtuple("ConvDecoderOutput", ["logits", "predicted_ids"])):
    pass


''' Functions of RAGE convolutional decoder'''
class ConvDecoder(Decoder):
    def __init__(self, is_train, batchsize, start_token, eos_token, word_emd, POS_emd, wordid2posid, decoder_pos_emd,
                 decoder_layer, fm_list, kwidth_list, layer_keep_prob, emd_keep_prob, max_de_len, emd_size,
                 out_keep_prob, vocab_size, scope_reuse):
        self.batchsize = batchsize
        self.is_train = is_train
        self.start_token = start_token
        self.eos_token = eos_token
        # position embedding
        self.decoder_pos_emd = decoder_pos_emd
        self.decoder_layer = decoder_layer
        self.fm_list = fm_list
        self.kwidth_list = kwidth_list
        self.layer_keep_prob = layer_keep_prob
        self.emd_keep_prob = emd_keep_prob
        self.out_keep_prob = out_keep_prob
        self.max_de_len = max_de_len
        self.emd_size = emd_size
        self.vocab_size = vocab_size
        self.word_emd = word_emd
        self.initial_state = None
        # scope reuse param
        self.scope_reuse = scope_reuse
        # POS tag embedding
        self.POS_emd = POS_emd
        self.wordid2posid = wordid2posid
        # unknown POS tag sign
        self.POS_UNK = 0

    ''' Create position embedding, position embedding size is equal with word embedding size'''
    def create_position_embedding(self, lengths):
        # [max_de_len,emd_size]
        pos_emd = self.decoder_pos_emd
        batch_size = tf.shape(lengths)[0]
        # [batchsize,max_de_len,emd_size]
        pe_batch = tf.tile([pos_emd], [batch_size, 1, 1])
        # [batchsize,max_de_len]
        pos_mask = tf.sequence_mask(lengths=lengths, maxlen=self.max_de_len, dtype=tf.float32)
        # [batchsize,max_de_len,emd_size]*[batchsize,max_de_len,1]-->[batchsize,max_de_len,emd_size]
        pos_emd = pe_batch * tf.expand_dims(pos_mask, 2)
        return pos_emd

    ''' Create POS tag embedding, POS tag embedding size is equal with word embedding size'''
    def create_POS_embedding(self, dec_input_pos):
        # [batchsize,max_de_len,emd_size]
        dec_pos_emd = tf.nn.embedding_lookup(self.POS_emd, dec_input_pos)
        # [batchsize,max_de_len]
        dec_pos_mask = tf.negative(tf.cast(tf.equal(dec_input_pos, self.POS_UNK), dtype=tf.float32)) + 1.0
        dec_pos_emd = dec_pos_emd * tf.expand_dims(dec_pos_mask, axis=2)
        return dec_pos_emd

    '''The convolutional operation block of decoder'''
    def conv_block(self, enc_output, input_embed, wcom_emd):
        with tf.variable_scope("decoder_cnn", reuse=self.scope_reuse):
            # [batchsize,max_de_len,emd_size]
            next_layer = input_embed
            fm_list = self.fm_list
            kwidth_list = self.kwidth_list
            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, fm_list[0], dropout=self.emd_keep_prob,
                                                   var_scope_name="linear_mapping_before_cnn",
                                                   scope_reuse=self.scope_reuse)
            # decoder hidden state[batchsize,max_de_len,fm]
            next_layer, final_com_att = conv_decoder_stack(input_embed, enc_output.outputs, enc_output.attention_values,
                                                           next_layer, wcom_emd, fm_list, kwidth_list,
                                                           self.layer_keep_prob, self.is_train,
                                                           scope_reuse=self.scope_reuse)
            self.com_att_out = final_com_att
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("softmax", reuse=self.scope_reuse):
            if self.is_train:
                next_layer = linear_mapping_weightnorm(next_layer, self.emd_size,
                                                       var_scope_name="linear_mapping_after_cnn",
                                                       scope_reuse=self.scope_reuse)
            else:
                # [batchsize,1,emd_size]
                next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :], self.emd_size,
                                                       var_scope_name="linear_mapping_after_cnn",
                                                       scope_reuse=self.scope_reuse)
            next_layer = tf.contrib.layers.dropout(inputs=next_layer, keep_prob=self.out_keep_prob,
                                                   is_training=self.is_train)
            # train [batchsize,max_de_len,vocab_size]
            # inference [batchsize,1,vocab_size]
            logits = linear_mapping_weightnorm(next_layer, self.vocab_size, dropout=self.out_keep_prob,
                                               var_scope_name="logits_before_softmax", scope_reuse=self.scope_reuse)
            tf.get_variable_scope().reuse_variables()
        return logits

    ''' Convolutional operation for decoder at training time'''
    def conv_decoder_train(self, enc_output, dec_input, dec_input_pos, dec_groundtruth, seq_len, wcom_emd):
        emd_size = self.emd_size
        pos_emd = self.create_position_embedding(lengths=seq_len)
        POS_emd = self.create_POS_embedding(dec_input_pos)
        # [batchsize,max_de_len,emd_size]
        dec_input = tf.add(tf.add(dec_input, pos_emd), POS_emd)
        # Apply dropout to embeddings
        dec_input = tf.contrib.layers.dropout(inputs=dec_input, keep_prob=self.emd_keep_prob, is_training=self.is_train)

        logits = self.conv_block(enc_output, dec_input, wcom_emd)
        predict = tf.arg_max(logits, dimension=2)
        loss_weight = tf.cast(tf.sign(dec_groundtruth), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=dec_groundtruth, weights=loss_weight)
        return logits, predict, loss

    def batch_size(self):
        return self.batchsize

    def output_size(self):
        return ConvDecoderOutput(logits=self.vocab_size, predicted_ids=tf.TensorShape([]))

    def output_dtype(self):
        return ConvDecoderOutput(logits=tf.float32, predicted_ids=tf.int32)

    def _setup(self, initial_state, helper=None):
        self.initial_state = initial_state

    def initialize(self, name=None):
        # [False,False,....]
        finished = tf.tile([False], [self.batchsize])
        # [Start,Start,....]
        start_tokens_batch = tf.fill([self.batchsize], self.start_token)
        # [batchsize,emd_size]
        first_inputs = tf.nn.embedding_lookup(self.word_emd, start_tokens_batch)
        # [batchsize,1,emd_size]
        first_inputs = tf.expand_dims(first_inputs, 1)
        # a slot for START
        zeros_padding = tf.zeros([self.batchsize, self.max_de_len - 1, self.emd_size])
        # [batchsize,max_de_len,emd_size]
        first_inputs = tf.concat([first_inputs, zeros_padding], axis=1)
        enc_output = ConvEncoderOutput(outputs=self.initial_state.outputs,
                                       attention_values=self.initial_state.attention_values)
        return finished, first_inputs, enc_output

    def finalize(self, outputs, final_state):

        return outputs, final_state

    def next_inputs(self, sample_ids, name=None):
        # [batchsize*1]
        finished = math_ops.equal(sample_ids, self.eos_token)
        # 1
        all_finished = math_ops.reduce_all(finished)
        # [batchsize]
        sample_pos = tf.nn.embedding_lookup(self.wordid2posid, sample_ids)
        # [batchsize,emd_size]
        sample_pos_emd = tf.nn.embedding_lookup(self.POS_emd, sample_pos)
        next_inputs = control_flow_ops.cond(
            all_finished,
            lambda: tf.nn.embedding_lookup(self.word_emd, tf.tile([self.eos_token], [self.batchsize])),
            # [batchsize*1,emd_size]
            lambda: tf.add(tf.nn.embedding_lookup(self.word_emd, sample_ids), sample_pos_emd))
        return all_finished, next_inputs

    def add_position_embedding(self, inputs, time):
        seq_pos_embed = self.decoder_pos_emd[0:time + 1, :]
        seq_pos_embed = tf.expand_dims(seq_pos_embed, axis=0)
        # [batchsize,time,emd_size]
        seq_pos_embed_batch = tf.tile(seq_pos_embed, [self.batchsize, 1, 1])
        return tf.add(inputs, seq_pos_embed_batch)

    def step(self, time, inputs, state, context, name=None):
        cur_inputs = inputs[:, 0:time + 1, :]
        zeros_padding = inputs[:, time + 2:, :]
        cur_inputs_pos = self.add_position_embedding(cur_inputs, time)
        enc_output = state
        # [batchsize*1,vocab_size]
        logits = self.infer_conv_block(enc_output, cur_inputs_pos, context)
        # [batchsize*1]
        sample_ids = tf.cast(tf.argmax(logits, axis=-1), dtypes.int32)
        # next_input[batchsize*1,emd_size]
        finished, next_inputs = self.next_inputs(sample_ids=sample_ids)
        # [batchsize,1,emd_size]
        next_inputs = tf.reshape(next_inputs, [self.batchsize, 1, inputs.get_shape().as_list()[-1]])
        next_inputs = tf.concat([cur_inputs, next_inputs], axis=1)
        next_inputs = tf.concat([next_inputs, zeros_padding], axis=1)
        next_inputs.set_shape([self.batchsize, self.max_de_len, inputs.get_shape().as_list()[-1]])
        outputs = ConvDecoderOutput(logits=logits, predicted_ids=sample_ids)
        return outputs, enc_output, next_inputs, finished

    ''' The convolutional operation block for testing time'''
    def infer_conv_block(self, enc_output, input_embed, wcom_emd):
        # Apply dropout to embeddings
        # [batchsize,1,emd_size]
        input_embed = tf.contrib.layers.dropout(inputs=input_embed, keep_prob=self.emd_keep_prob,
                                                is_training=self.is_train)
        # [batchsize,1,vocab_size]
        next_layer = self.conv_block(enc_output, input_embed, wcom_emd)
        shape = next_layer.get_shape().as_list()
        # [batchsize*1,vocab_size]
        logits = tf.reshape(next_layer, [-1, shape[-1]])
        return logits

    def init_params_in_loop(self, wcom_emd):
        with tf.variable_scope("decoder", reuse=self.scope_reuse):
            initial_finished, initial_inputs, initial_state = self.initialize()
            enc_output = initial_state
            logits = self.infer_conv_block(enc_output, initial_inputs, wcom_emd)
            tf.get_variable_scope().reuse_variables()

    '''The convolutional operation of decoder'''
    def conv_decoder_infer(self, enc_output, wcom_emd):
        if not self.initial_state:
            self._setup(initial_state=enc_output)
        maximum_iterations = self.max_de_len
        self.init_params_in_loop(wcom_emd)
        tf.get_variable_scope().reuse_variables()
        outputs, final_state = dynamic_decode(decoder=self, wcom_emd=wcom_emd, output_time_major=False,
                                              impute_finished=False, maximum_iterations=maximum_iterations,
                                              scope_reuse=self.scope_reuse)
        return outputs, final_state
