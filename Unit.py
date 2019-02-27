# coding=UTF-8
__author__ = 'shiqian.csq'
import tensorflow as tf

'''Stack convolutional operation for encoder'''
def conv_encoder_stack(encoder_input_emd, layer_num, fm_list, kwidth_list, layer_keep_prob, is_train, scope_reuse):
    # [batchsize,max_en_len,emd_size]
    next_layer = encoder_input_emd
    for layer_id in range(layer_num):
        # residual Connection
        res_input = next_layer
        # featureMap num*2 for gate linear,feature num=embedding size
        fm_num = fm_list[layer_id] * 2
        kwidth = kwidth_list[layer_id]

        # drop out before input to conv
        next_layer = tf.contrib.layers.dropout(inputs=next_layer, keep_prob=layer_keep_prob, is_training=is_train)

        # [batchsize,max_en_len,fm_num*2]
        next_layer = conv1d_weightnorm(next_layer, layer_id, fm_num, kwidth, dropout=layer_keep_prob,
                                       scope_reuse=scope_reuse)
        # [batchsize,max_en_len,fm_num]
        next_layer = gate_linear_unit(next_layer)
        next_layer = (next_layer + res_input) * tf.sqrt(0.5)
    return next_layer

'''GLU'''
def gate_linear_unit(input):
    # [batchsize,max_en_len,fm_num*2]
    input_shape = input.get_shape().as_list()
    input_mes = input[:, :, 0:int(input_shape[2] / 2)]
    input_gate = tf.nn.sigmoid(input[:, :, int(input_shape[2] / 2):])
    input_pass = tf.multiply(input_mes, input_gate)
    return input_pass

'''Convolutional operation'''
def conv1d_weightnorm(inputs, layer_id, out_dim, kernel_size, scope_reuse, dropout=1.0, padding="SAME"):
    with tf.variable_scope("conv_layer_" + str(layer_id), reuse=scope_reuse):
        # input:[batch,max_en_len,emd_size]
        in_dim = int(inputs.get_shape().as_list()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        # V shape is M*N*k,  V_norm shape is k
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        # [kwight,emd_size,out_dim]
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        # [batch,max_en_len,out_dim]
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        tf.get_variable_scope().reuse_variables()
    return inputs

'''Stack Convolutional operation for decoder'''
def conv_decoder_stack(previous_word, enc_output, enc_att_values, inputs, wcom_emd, fm_list, kwidth_list,
                       layer_keep_prob, is_train, scope_reuse):
    # [batchsize,max_de_len,emd_size]
    next_layer = inputs
    final_com_att = inputs
    for layer_id in range(len(fm_list)):
        fm = fm_list[layer_id]
        kwidth = kwidth_list[layer_id]
        res_inputs = next_layer
        next_layer = tf.contrib.layers.dropout(inputs=next_layer, keep_prob=layer_keep_prob, is_training=is_train)
        # [batchsize,max_de_len+(kwidth-1)*2,emd_size]
        next_layer = tf.pad(next_layer, [[0, 0], [kwidth_list[layer_id] - 1, kwidth_list[layer_id] - 1], [0, 0]],
                            "CONSTANT")
        next_layer = conv1d_weightnorm(inputs=next_layer, layer_id=layer_id, out_dim=fm * 2, kernel_size=kwidth,
                                       dropout=layer_keep_prob, padding="VALID", scope_reuse=scope_reuse)

        # truncate
        # [batchsize,max_de_len,fm*2]
        next_layer = next_layer[:, 0:-kwidth + 1, :]
        # [batchsize,max_de_len,fm]
        next_layer = gate_linear_unit(next_layer)

        # make attention with question context
        # [batchsize,max_de_len,fm]
        enc_emd_att_out, enc_hid_att_out = make_attention(previous_word, enc_output, enc_att_values, next_layer,
                                                          layer_id, scope_reuse=scope_reuse)

        # make attention with words in comments
        com_att_out = make_com_attention_with_enc(next_layer, wcom_emd, enc_emd_att_out, layer_id,
                                                  scope_reuse=scope_reuse)
        final_com_att = com_att_out
        # [batchsize,max_de_len,fm*3]
        gate_input = tf.concat([next_layer, enc_hid_att_out, com_att_out], axis=-1)
        gate_one = tf.nn.tanh(
            tf.contrib.layers.fully_connected(gate_input, next_layer.get_shape().as_list()[-1], trainable=True,
                                              reuse=scope_reuse, scope="gate_one_" + str(layer_id)))
        # [batchsize,max_de_len,fm]
        gate_two = tf.nn.tanh(
            tf.contrib.layers.fully_connected(gate_one, next_layer.get_shape().as_list()[-1], trainable=True,
                                              reuse=scope_reuse, scope="gate_two_" + str(layer_id)))

        next_layer = next_layer + tf.multiply(gate_two, enc_hid_att_out) + tf.multiply(1 - gate_two, com_att_out)

        # add residual connections
        # [batchsize,max_de_len,fm]
        next_layer += (next_layer + res_inputs) * tf.sqrt(0.5)
    return next_layer, final_com_att

'''Attention with question context'''
def make_attention(previous_word, enc_output, enc_att_values, decoder_hidden, layer_idx, scope_reuse):
    with tf.variable_scope("attention_layer_" + str(layer_idx), reuse=scope_reuse):
        embed_size = previous_word.get_shape().as_list()[-1]
        dec_hidden_proj = linear_mapping_weightnorm(decoder_hidden, embed_size,
                                                    var_scope_name="linear_mapping_att_query",
                                                    scope_reuse=scope_reuse)  # M*N1*k1 --> M*N1*k
        # [batchsize,max_de_len,fm]
        dec_rep = (dec_hidden_proj + previous_word) * tf.sqrt(0.5)

        # without residual connection[batchsize,max_en_len,emd_size]
        encoder_output_a = enc_output
        # with residual connection[batchsize,max_en_len,emd_size]
        encoder_output_c = enc_att_values

        # [batchsize,max_de_len,max_en_len]
        att_score = tf.matmul(dec_rep, encoder_output_a, transpose_b=True)
        att_score = tf.nn.softmax(att_score, dim=-1)

        length = tf.cast(tf.shape(encoder_output_c), tf.float32)
        emd_att_out = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(1.0 / length[1])
        hid_att_out = linear_mapping_weightnorm(emd_att_out, decoder_hidden.get_shape().as_list()[-1],
                                                var_scope_name="linear_mapping_att_out", scope_reuse=scope_reuse)
        tf.get_variable_scope().reuse_variables()

        return emd_att_out, hid_att_out

'''Attention with words in comments based on question context'''
def make_com_attention_with_enc(decoder_hidden, wcom_emd, enc_emd_att_out, layer_idx, scope_reuse=False):
    with tf.variable_scope("attention_layer_with_enc_" + str(layer_idx), reuse=scope_reuse):
        # [batchsize,max_de_len,emd_size]*[batchsize,max_com_vocab,emd_size]-->[batchsize,max_de_len,max_com_vocab]
        att_score = tf.matmul(enc_emd_att_out, wcom_emd, transpose_b=True)

        att_score = tf.nn.softmax(att_score, dim=-1)
        length = tf.cast(tf.shape(wcom_emd), tf.float32)
        # [batchsize,max_de_len,fm]
        att_out = tf.matmul(att_score, wcom_emd) * length[1] * tf.sqrt(1.0 / length[1])
        # emd-->hid
        att_out = linear_mapping_weightnorm(att_out, decoder_hidden.get_shape().as_list()[-1],
                                            var_scope_name="linear_mapping_att_out", scope_reuse=scope_reuse)
        tf.get_variable_scope().reuse_variables()

        return att_out

''' Linear mapping'''
def linear_mapping_weightnorm(inputs, out_dim, scope_reuse, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name, scope_reuse):
        input_shape = inputs.get_shape().as_list()
        input_shape_tensor = tf.shape(inputs)

        # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                dropout * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)  # weightnorm bias is init zero
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
        scaler = tf.div(g, tf.norm(V, axis=0))  # g/2-norm(v)
        # [batchsize,max_de_len,out_dim]
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])  # x*v g/2-norm(v) + b
        tf.get_variable_scope().reuse_variables()
    return inputs
