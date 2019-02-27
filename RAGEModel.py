#coding=UTF-8
__author__ = 'shiqian.csq'
import tensorflow as tf
from rage.RAGEEncoder import ConvEncoder,ConvEncoderOutput
from rage.RAGEDecoder import ConvDecoder,ConvDecoderOutput
import numpy as np
import os

''' Functions of RAGE convolutional Seq2seq'''
class ConvSeq2seq:
    def __init__(self,batchsize,is_train,learning_rate,word_emd,emd_size,vocab_size,POS_size,wordid2posid,start_token,eos_token,emd_keep_prob,layer_keep_prob,out_keep_prob,max_com_vocab,enc_layer,dec_layer,enc_fm_list,dec_fm_list,enc_kwidth_list,dec_kwidth_list,max_en_len,max_de_len,scope_reuse=False,decay_steps=100000,decay_rate=0.9):
        with tf.variable_scope("RAGE",reuse=scope_reuse) as scope_model:
        #share param
            #True:training  False:inference
            self.is_train=is_train
            self.batchsize=batchsize
            self.learning_rate=learning_rate
            self.word_emd=word_emd
            #word embedding size, position embedding size, POS tag embedding size are equal, and here the size of hidden state is also equal with word embedding size
            self.emd_size=emd_size
            self.vocab_size=vocab_size
            self.start_token=start_token
            self.eos_token=eos_token
            self.emd_keep_prob=emd_keep_prob
            self.layer_keep_prob=layer_keep_prob
            self.out_keep_prob=out_keep_prob
            # the maximum size of weighted vocabulary of comments
            self.max_com_vocab=max_com_vocab
            self.POS_size=POS_size
            self.POS_emd = tf.get_variable("POS_emd", [self.POS_size, self.emd_size], tf.float32,
                                           initializer=tf.random_normal_initializer(0.0, 0.01), trainable=True)
            self.wordid2posid = tf.constant(wordid2posid)

        #encoder param
            self.max_en_len=max_en_len
            self.enc_kwidth_list=enc_kwidth_list
            self.enc_fm_list=enc_fm_list
            self.enc_layer=enc_layer
            # encoder and decoder position embedding are not shared
            self.enc_pos_emd=tf.get_variable("enc_pos_emd",[self.max_en_len,self.emd_size],tf.float32,initializer=tf.random_normal_initializer(0.0,0.01),trainable=True)

        #decoder param
            self.max_de_len=max_de_len
            self.dec_layer=dec_layer
            self.dec_fm_list=dec_fm_list
            self.dec_kwidth_list=dec_kwidth_list
            self.dec_pos_emd=tf.get_variable("dec_pos_emd",[self.max_de_len,self.emd_size],tf.float32,initializer=tf.random_normal_initializer(0.0,0.01),trainable=True)

        #create encoder
            self.encoder=ConvEncoder(self.is_train,self.POS_emd,self.max_en_len,self.enc_pos_emd,self.enc_layer,self.enc_fm_list,self.enc_kwidth_list,self.emd_keep_prob,self.layer_keep_prob,scope_reuse=scope_reuse)
        #create decoder
            self.decoder=ConvDecoder(self.is_train,self.batchsize,self.start_token,self.eos_token,self.word_emd,self.POS_emd,self.wordid2posid,self.dec_pos_emd,self.dec_layer,self.dec_fm_list,self.dec_kwidth_list,self.layer_keep_prob,self.emd_keep_prob,self.max_de_len,self.emd_size,self.out_keep_prob,self.vocab_size,scope_reuse=scope_reuse)

        #placeholder
            self.enc_input=tf.placeholder(tf.int32,[None,self.max_en_len],name="enc_input")
            self.dec_gt=tf.placeholder(tf.int32,[None,self.max_de_len],name="dec_gt")

            self.enc_input_pos = tf.placeholder(tf.int32, [None, self.max_en_len], name="enc_input_pos")
            #[batchsize,max_com_vocab]
            self.com=tf.placeholder(tf.int32,[None,self.max_com_vocab],name="com")
            self.com_emd=tf.nn.embedding_lookup(self.word_emd,self.com)
            self.raw_com_weight=tf.placeholder(tf.float32,[None,self.max_com_vocab],name="com_weight")
            #[batchsize,max_com_vocab,1]
            self.com_weight=tf.expand_dims(self.raw_com_weight,axis=-1)
            #[batchsize,max_com_vocab,emd_size]-->weight embedding
            self.wcom_emd=tf.multiply(self.com_emd,self.com_weight)

        #train
            if is_train:
                self.dec_input=tf.placeholder(tf.int32,[None,self.max_de_len],name="dec_input")
                self.dec_input_pos = tf.placeholder(tf.int32, [None, self.max_de_len], name="dec_input_pos")
                #[batchsize,max_en_len,emd_size]
                self.enc_input_emd=tf.nn.embedding_lookup(self.word_emd,self.enc_input)
            #enode
                self.enc_output=self.encoder.encode(self.enc_input_emd,self.enc_input_pos,self.length(self.enc_input_emd))

            #decode
                self.dec_input_emd=tf.nn.embedding_lookup(self.word_emd,self.dec_input)
                with tf.variable_scope("decoder",reuse=scope_reuse) as scope:
                    self.train_logit,self.train_predict,self.train_loss,=self.decoder.conv_decoder_train(self.enc_output,self.dec_input_emd,self.dec_input_pos,self.dec_gt,self.length(self.dec_input_emd),self.wcom_emd)
                    self.l2=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])*0.01
                    self.train_loss=self.train_loss+self.l2
                    tf.get_variable_scope().reuse_variables()

            #loss & optimize
                self.optimizer=tf.train.MomentumOptimizer(self.learning_rate,0.99,use_nesterov=True)
                self.gradients,self.variables=zip(*self.optimizer.compute_gradients(self.train_loss))
                self.gradients,_=tf.clip_by_global_norm(self.gradients,0.1)
                self.train_op=self.optimizer.apply_gradients(zip(self.gradients,self.variables))
                tf.get_variable_scope().reuse_variables()


            #inference
            else:
            #encode
                self.enc_input_emd=tf.nn.embedding_lookup(self.word_emd,self.enc_input)
                self.enc_output=self.encoder.encode(self.enc_input_emd,self.enc_input_pos,self.length(self.enc_input_emd))

            #decode
                self.infer_output,self.infer_state=self.decoder.conv_decoder_infer(self.enc_output,self.wcom_emd)
                #[batchsize,timestep,vocab_size]-->[batchsize,max_de_len,vocab_size]
                raw_infer_logit=self.infer_output.logits
                raw_infer_logit=tf.pad(raw_infer_logit, [[0, 0], [0,self.max_de_len], [0, 0]], "CONSTANT")
                self.infer_logit=tf.slice(raw_infer_logit,[0,0,0],[self.batchsize,self.max_de_len,self.vocab_size])
                self.infer_predict=self.infer_output.predicted_ids
                loss_weight=tf.cast(tf.sign(self.dec_gt),tf.float32)
                self.loss=tf.contrib.seq2seq.sequence_loss(logits=self.infer_logit,targets=self.dec_gt,weights=loss_weight)
                tf.get_variable_scope().reuse_variables()


    '''get the valid length of batch sentence'''
    def length(self,input_emd):
        #[batchsize,sen_len,emd_size]--->[batchsize,sen_len]
        input_0=tf.reduce_max(tf.abs(input_emd),axis=2)
        input_1=tf.sign(input_0)
        #[batchsize]
        seq_len=tf.reduce_sum(input_1,axis=1)
        return tf.cast(seq_len,tf.int32)



'''Test training'''
def demo_train():
    batchsize = 2
    learning_rate = 0.25
    is_train = True
    word_emd = np.random.ranf([10, 5])
    word_emd = np.array(word_emd, dtype='float32')
    emd_size = 5
    vocab_size = 10
    start_token = 1
    eos_token = 0
    emd_keep_prob = 0.9
    layer_keep_prob = 0.9
    out_keep_prob = 0.9
    enc_layer = 2
    dec_layer = 2
    enc_fm_list = [5, 5]
    dec_fm_list = [5, 5]
    enc_kwidth_list = [2, 2]
    dec_kwidth_list = [2, 2]
    max_en_len = 4
    max_de_len = 5
    POS_size = 6
    wordid2posid = [2, 3, 0, 0, 1, 5, 4, 3, 2, 5]
    max_com_vocab=7
    com=np.random.randint(0,9,[2,7])
    com_weight=np.random.ranf([2,7])

    #placeholder
    encoder_input = np.random.randint(0, 9, [2, 4])
    enc_input_pos = np.random.randint(0, 6, [2, 4])
    decoder_input = np.random.randint(0, 9, [2, 5])
    dec_input_pos = np.random.randint(0, 6, [2, 5])
    print("decoder_input", decoder_input)
    print("dec_input_pos", dec_input_pos)

    decoder_groundtruth = np.random.randint(0, 9, [2, 5])
    with tf.device("/cpu:0"):
        #config=tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.allow_growth=True
        with tf.Session() as sess:
            train = ConvSeq2seq(batchsize, is_train, learning_rate, word_emd, emd_size, vocab_size, POS_size,wordid2posid,start_token, eos_token,emd_keep_prob, layer_keep_prob, out_keep_prob, max_com_vocab,enc_layer, dec_layer, enc_fm_list,dec_fm_list, enc_kwidth_list, dec_kwidth_list, max_en_len, max_de_len)
            sess.run(tf.global_variables_initializer())
            _, batch_logit, batch_predict, batch_loss= sess.run([train.train_op, train.train_logit, train.train_predict, train.train_loss],feed_dict={train.enc_input: encoder_input, train.dec_input: decoder_input,train.dec_gt: decoder_groundtruth,train.enc_input_pos:enc_input_pos,train.dec_input_pos:dec_input_pos,train.com:com,train.raw_com_weight:com_weight})
            print("batch_logit", batch_logit)
            print("batch_predict", batch_predict)
            print("batch_loss", batch_loss)
            #save model
            saver = tf.train.Saver()
            saver.save(sess, "./model/check-final.cpkt")
            sess.close()

'''Test inference'''
def demo_inference():
    batchsize = 2
    learning_rate = 0.001
    is_train = False
    word_emd = np.random.ranf([10, 5])
    word_emd = np.array(word_emd, dtype='float32')
    emd_size = 5
    vocab_size = 10
    start_token = 1
    eos_token = 0
    emd_keep_prob = 0.9
    layer_keep_prob = 0.9
    out_keep_prob = 0.9
    enc_layer = 2
    dec_layer = 2
    enc_fm_list = [5, 5]
    dec_fm_list = [5, 5]
    enc_kwidth_list = [2, 2]
    dec_kwidth_list = [2, 2]
    max_en_len = 4
    max_de_len = 5
    POS_size = 6
    wordid2posid = [2, 3, 0, 0, 1, 5, 4, 3, 2, 5]
    max_com_vocab=7
    com=np.random.randint(0,9,[2,7])
    com_weight=np.random.ranf([2,7])

    #placeholder
    encoder_input = np.random.randint(0, 9, [2, 4])
    enc_input_pos = np.random.randint(0, 6, [2, 4])
    decoder_input = np.random.randint(0, 9, [2, 5])
    dec_input_pos = np.random.randint(0, 6, [2, 5])
    print("decoder_input", decoder_input)
    print("dec_input_pos", dec_input_pos)
    decoder_groundtruth = np.random.randint(0, 9, [2, 5])

    with tf.Session() as sess:
        inference = ConvSeq2seq(batchsize, is_train, learning_rate, word_emd, emd_size, vocab_size, POS_size,wordid2posid,start_token,eos_token, emd_keep_prob, layer_keep_prob, out_keep_prob, max_com_vocab,enc_layer, dec_layer,enc_fm_list, dec_fm_list, enc_kwidth_list, dec_kwidth_list, max_en_len, max_de_len)
        saver = tf.train.Saver()
        if os.path.exists("./model/" + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint("./model/"))
        else:
            print("Can't find the checkpoint.going to stop")
            return

        batch_logits, batch_predict, batch_loss= sess.run([inference.infer_logit, inference.infer_predict, inference.loss],feed_dict={inference.enc_input: encoder_input, inference.enc_input_pos:enc_input_pos,inference.com:com,inference.raw_com_weight:com_weight,inference.dec_gt: decoder_groundtruth})
        print("batch_output", batch_logits)
        print("batch_predict", batch_predict)
        print("batch_loss", batch_loss)
        sess.close()

if __name__=='__main__':
    demo_inference()
    #demo_train()
