# coding=UTF-8
__author__ = 'shiqian.csq'
import tensorflow as tf
from rage.Unit import *
from collections import namedtuple


class ConvEncoderOutput(
    namedtuple("ConvEncoderOutput", ["outputs", "attention_values"])):
    pass

''' Functions of RAGE convolutional decoder'''
class ConvEncoder:
    def __init__(self, is_train, POS_emd, max_en_len, encoder_pos_emd, encoder_layer, fm_list, kwidth_list,
                 emd_keep_prob, layer_keep_prob, scope_reuse):
        # position[max_en_len,emd_size]
        self.encoder_pos_emd = encoder_pos_emd
        # the layer number of encoder
        self.encoder_layer = encoder_layer
        # the feature map of each layer
        self.fm_list = fm_list
        # the window size of each layer
        self.kwidth_list = kwidth_list
        # the keep prob of embedding dropout
        self.emd_keep_prob = emd_keep_prob
        # the keep prob of dropout between layer and layer
        self.layer_keep_prob = layer_keep_prob
        self.is_train = is_train
        # [batchsize,max_en_len,emd_size]
        self.max_en_len = max_en_len
        self.scope_reuse = scope_reuse
        # POS tag embedding
        self.POS_emd = POS_emd
        self.POS_UNK = 0

    ''' Create position embedding, position embedding size is equal with word embedding size'''
    def create_position_embedding(self, lengths):
        pos_emd = self.encoder_pos_emd
        batch_size = tf.shape(lengths)[0]
        # [batchsize,max_en_len,emd_size]
        pe_batch = tf.tile([pos_emd], [batch_size, 1, 1])
        # [batchsize,batch_max_len]
        positions_mask = tf.sequence_mask(lengths=lengths, maxlen=self.max_en_len, dtype=tf.float32)
        positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)
        return positions_embed

    ''' Create POS tag embedding, POS tag embedding size is equal with word embedding size'''
    def create_POS_embedding(self, enc_input_pos):
        # [batchsize,max_en_len,emd_size]
        enc_pos_emd = tf.nn.embedding_lookup(self.POS_emd, enc_input_pos)
        # [batchsize,max_en_len]
        enc_pos_mask = tf.negative(tf.cast(tf.equal(enc_input_pos, self.POS_UNK), dtype=tf.float32)) + 1.0
        self.enc_pos_mask = enc_pos_mask
        enc_pos_emd = enc_pos_emd * tf.expand_dims(enc_pos_mask, axis=2)
        return enc_pos_emd

    ''' Encoder operation'''
    def encode(self, inputs, input_pos, sequence_length):
        # input[batchsize,max_en_len,emd_size]
        embed_size = inputs.get_shape().as_list()[-1]
        pos_emd = self.create_position_embedding(lengths=sequence_length)
        POS_emd = self.create_POS_embedding(input_pos)
        # input+pos_emd+POS
        inputs = tf.add(tf.add(inputs, pos_emd), POS_emd)
        # Apply dropout to embeddings
        inputs = tf.contrib.layers.dropout(inputs=inputs, keep_prob=self.emd_keep_prob, is_training=self.is_train)
        with tf.variable_scope("encoder_cnn", reuse=self.scope_reuse):
            next_layer = inputs
            fm_list = self.fm_list
            kwidth_list = self.kwidth_list
            # mapping emd dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, fm_list[0], dropout=self.emd_keep_prob,
                                                   var_scope_name="linear_mapping_before_cnn",
                                                   scope_reuse=self.scope_reuse)
            next_layer = conv_encoder_stack(next_layer, self.encoder_layer, fm_list, kwidth_list, self.layer_keep_prob,
                                            self.is_train, scope_reuse=self.scope_reuse)
            next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name="linear_mapping_after_cnn",
                                                   scope_reuse=self.scope_reuse)
            # residual connection
            res_connection = (next_layer + inputs) * tf.sqrt(0.5)
        return ConvEncoderOutput(next_layer, res_connection)
