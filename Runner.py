# coding=UTF-8
__author__ = 'shiqian.csq'
import tensorflow as tf
from rage.RAGEModel import ConvSeq2seq
from rage.DataProcess import *
import time
import os

MODE="train"
EPOCH = 50
BATCH_SIZE = 2
LEARNING_RATE = 0.01
EMD_SIZE = 200
# add PAD, EOS and START
VOCAB_SIZE = 8000+3
START_TOKEN = 1
EOS_TOKEN = 2
PAD = 0
EMD_KEEP_PROB = 0.9
LAYER_KEEP_PROB = 0.9
OUT_KEEP_PROB = 0.9
ENC_LAYER = 4
DEC_LAYER = 4
ENC_FM_LIST = [200, 200, 200, 200]
DEC_FM_LIST = [200, 200, 200, 200]
ENC_KWIDTH_LIST = [3, 3, 3, 3]
DEC_KWIDTH_LIST = [5, 5, 5, 5]
POS_SIZE = 60
MAX_COM_VOCAB = 100
MAX_EN_LEN = 30
MAX_DE_LEN = 30
SAVE_STEP = 2
SAVE_MODEL_PATH = "./model/"

'''Zip dataset for shuffle'''
def zipAll(train_enc_input, train_dec_gt, train_dec_input, train_com, train_com_weight, train_enc_input_pos,
           train_dec_input_pos):
    zipSet = []
    for enc, dec_gt, dec_input, com, com_weight, enc_input_pos, dec_input_pos in zip(train_enc_input, train_dec_gt,
                                                                                     train_dec_input, train_com,
                                                                                     train_com_weight,
                                                                                     train_enc_input_pos,
                                                                                     train_dec_input_pos):
        item = []
        item.append(enc)
        item.append(dec_gt)
        item.append(dec_input)
        item.append(com)
        item.append(com_weight)
        item.append(enc_input_pos)
        item.append(dec_input_pos)
        zipSet.append(item)
    return zipSet


'''Shuffle and unpack'''
def shuffle(zipSet):
    train_enc_input = []
    train_dec_gt = []
    train_dec_input = []
    train_com = []
    train_com_weight = []
    train_enc_input_pos = []
    train_dec_input_pos = []

    np.random.shuffle(zipSet)
    for item in zipSet:
        train_enc_input.append(item[0])
        train_dec_gt.append(item[1])
        train_dec_input.append(item[2])
        train_com.append(item[3])
        train_com_weight.append(item[4])
        train_enc_input_pos.append(item[5])
        train_dec_input_pos.append(item[6])
    return train_enc_input, train_dec_gt, train_dec_input, train_com, train_com_weight, train_enc_input_pos, train_dec_input_pos


def train(emd_mtx, wordid2posid, train_enc, train_dec_gt, train_dec_input, train_com, train_com_weight,
          train_enc_input_pos, train_dec_input_pos, valid_enc_input, valid_enc_input_pos, valid_dec_gt, valid_com,
          valid_com_weight):
    # init model
    with tf.device("/gpu:0"):
        zipSet = zipAll(train_enc, train_dec_gt, train_dec_input, train_com, train_com_weight, train_enc_input_pos,
                        train_dec_input_pos)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess_train:
            train = ConvSeq2seq(BATCH_SIZE, True, LEARNING_RATE, emd_mtx, EMD_SIZE, VOCAB_SIZE, POS_SIZE,
                                wordid2posid, START_TOKEN, EOS_TOKEN, EMD_KEEP_PROB, LAYER_KEEP_PROB, OUT_KEEP_PROB,
                                MAX_COM_VOCAB, ENC_LAYER, DEC_LAYER, ENC_FM_LIST, DEC_FM_LIST, ENC_KWIDTH_LIST,
                                DEC_KWIDTH_LIST, MAX_EN_LEN, MAX_DE_LEN, False)
            valid = ConvSeq2seq(BATCH_SIZE, False, LEARNING_RATE, emd_mtx, EMD_SIZE, VOCAB_SIZE, POS_SIZE,
                                wordid2posid, START_TOKEN, EOS_TOKEN, EMD_KEEP_PROB, LAYER_KEEP_PROB, OUT_KEEP_PROB,
                                MAX_COM_VOCAB, ENC_LAYER, DEC_LAYER, ENC_FM_LIST, DEC_FM_LIST, ENC_KWIDTH_LIST,
                                DEC_KWIDTH_LIST, MAX_EN_LEN, MAX_DE_LEN, True)
            sess_train.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000)

            for epoch in range(EPOCH):
                train_enc, train_dec_gt, train_dec_input, train_com, train_com_weight, train_enc_input_pos, train_dec_input_pos = shuffle(
                    zipSet)
                ISOTIMEFORMAT = "%Y-%m-%d %X"
                time.localtime()
                print(time.strftime(ISOTIMEFORMAT, time.localtime()))
                epoch_cost = 0.

                for start, end in zip(range(0, len(train_enc), BATCH_SIZE),
                                      range(BATCH_SIZE, len(train_enc), BATCH_SIZE)):
                    batch_enc_input = train_enc[start:end]
                    batch_enc_input_pos = train_enc_input_pos[start:end]
                    batch_dec_gt = train_dec_gt[start:end]
                    batch_dec_input = train_dec_input[start:end]
                    batch_dec_input_pos = train_dec_input_pos[start:end]
                    batch_com = train_com[start:end]
                    batch_com_weight = train_com_weight[start:end]
                    _, batch_predict, batch_loss = sess_train.run(
                        [train.train_op, train.train_predict, train.train_loss],
                        feed_dict={train.enc_input: batch_enc_input, train.dec_input: batch_dec_input,
                                   train.dec_gt: batch_dec_gt, train.com: batch_com,
                                   train.raw_com_weight: batch_com_weight, train.enc_input_pos: batch_enc_input_pos,
                                   train.dec_input_pos: batch_dec_input_pos})
                    epoch_cost = epoch_cost + batch_loss

                print("Loss at train time :", '%04d' % (epoch), "cost=", "{:.9f}".format(epoch_cost))
                # save model, and validation
                if 0 == epoch % SAVE_STEP:
                    print("Save Model at Epoch: %d" % (epoch))
                    save_path = SAVE_MODEL_PATH + "model.ckpt"
                    saver.save(sess_train, save_path, global_step=epoch)
                    validation(sess_train, valid, valid_enc_input, valid_enc_input_pos, valid_dec_gt, valid_com,
                               valid_com_weight)
            print("Train Optimization Finished!")


def validation(sess_infer, validation, valid_enc, valid_enc_input_pos, valid_dec_gt, valid_com, valid_com_weight):
    epoch_cost = 0.
    ISOTIMEFORMAT = "%Y-%m-%d %X"
    time.localtime()
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    print("Start Validation ..")
    index = 0
    for start, end in zip(range(0, len(valid_enc), BATCH_SIZE), range(BATCH_SIZE, len(valid_enc), BATCH_SIZE)):
        index += 1
        batch_enc_input = valid_enc[start:end]
        batch_enc_input_pos = valid_enc_input_pos[start:end]
        batch_dec_gt = valid_dec_gt[start:end]
        batch_com = valid_com[start:end]
        batch_com_weight = valid_com_weight[start:end]
        # print(batch_dec_gt)
        batch_loss, batch_predict = sess_infer.run([validation.loss, validation.infer_predict],
                                                   feed_dict={validation.enc_input: batch_enc_input,
                                                              validation.dec_gt: batch_dec_gt,
                                                              validation.com: batch_com,
                                                              validation.raw_com_weight: batch_com_weight,
                                                              validation.enc_input_pos: batch_enc_input_pos})
        epoch_cost = epoch_cost + batch_loss

    print("Loss at validation time =", "{:.9f}".format(epoch_cost))


def inference(emd_mtx, wordid2posid, infer_enc, infer_enc_pos, infer_dec_gt, infer_com, infer_com_weight):
    # init model
    with tf.device("/gpu:0"):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess_infer:
            inference = ConvSeq2seq(BATCH_SIZE, False, LEARNING_RATE, emd_mtx, EMD_SIZE, VOCAB_SIZE, POS_SIZE,
                                    wordid2posid, START_TOKEN, EOS_TOKEN, EMD_KEEP_PROB, LAYER_KEEP_PROB, OUT_KEEP_PROB,
                                    MAX_COM_VOCAB, ENC_LAYER, DEC_LAYER, ENC_FM_LIST, DEC_FM_LIST, ENC_KWIDTH_LIST,
                                    DEC_KWIDTH_LIST, MAX_EN_LEN, MAX_DE_LEN, False)
            saver = tf.train.Saver()
            if os.path.exists(SAVE_MODEL_PATH + "checkpoint"):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess_infer, tf.train.latest_checkpoint(SAVE_MODEL_PATH))
            else:
                print("Can't find the checkpoint.going to stop")
                return
            epoch_cost = 0.
            ISOTIMEFORMAT = "%Y-%m-%d %X"
            time.localtime()
            print(time.strftime(ISOTIMEFORMAT, time.localtime()))
            print("Start Inference ..")
            index = 0
            for start, end in zip(range(0, len(infer_enc), BATCH_SIZE),
                                  range(BATCH_SIZE, len(infer_dec_gt), BATCH_SIZE)):
                index += 1
                batch_enc_input = infer_enc[start:end]
                batch_enc_input_pos = infer_enc_pos[start:end]
                batch_dec_gt = infer_dec_gt[start:end]
                batch_com = infer_com[start:end]
                batch_com_weight = infer_com_weight[start:end]
                batch_loss, batch_predict = sess_infer.run([inference.loss, inference.infer_predict],
                                                           feed_dict={inference.enc_input: batch_enc_input,
                                                                      inference.dec_gt: batch_dec_gt,
                                                                      inference.com: batch_com,
                                                                      inference.raw_com_weight: batch_com_weight,
                                                                      inference.enc_input_pos: batch_enc_input_pos})
                epoch_cost = epoch_cost + batch_loss

            print("Loss at inference time =", "{:.9f}".format(epoch_cost))


if __name__ == '__main__':
    # load stopword for comment word weight calculation
    stopword = load_stopword("./data/stopword.dic")

    # load vocabulary
    word2id, id2word = load_vocab("./data/wdj_word_fre.txt", VOCAB_SIZE)

    # load POS tag for valid vocabulary
    pos2id, id2pos, word2pos, wordid2posid, wordid2posid_vec = load_word2pos(word2id, "./data/wdj_word_pos.txt")

    # load word embedding
    emd_dict, word_emd = load_word_embedding(word2id, "./data/wdj_word_emd.txt", EMD_SIZE)

    if MODE=="train":
    #train & validation
        # prepare data for train and validation
        enc_input, enc_input_pos, dec_gt, dec_gt_pos, dec_input, dec_input_pos, com, com_weight = read_QApair_withPos(
            word2id, pos2id, "./data/sample.txt", MAX_EN_LEN, MAX_DE_LEN, emd_dict, stopword, MAX_COM_VOCAB)
        print("End prepare train and validation data..")
        # train set
        train_enc_input = enc_input[0:-10]
        train_enc_input_pos = enc_input_pos[0:-10]
        train_dec_gt = dec_gt[0:-10]
        train_dec_gt_pos = dec_gt_pos[0:-10]
        train_dec_input = dec_input[0:-10]
        train_dec_input_pos = dec_input_pos[0:-10]
        train_com = com[0:-10]
        train_com_weight = com_weight[0:-10]
        # valid set
        valid_enc_input = enc_input[-10:]
        valid_enc_input_pos = enc_input_pos[-10:]
        valid_dec_gt = dec_gt[-10:]
        valid_dec_gt_pos = dec_gt_pos[-10:]
        valid_dec_input = dec_input[-10:]
        valid_dec_input_pos = dec_input_pos[-10:]
        valid_com = com[-10:]
        valid_com_weight = com_weight[-10:]

        train(word_emd, wordid2posid_vec, train_enc_input, train_dec_gt, train_dec_input, train_com, train_com_weight,
              train_enc_input_pos, train_dec_input_pos, valid_enc_input, valid_enc_input_pos, valid_dec_gt, valid_com,
              valid_com_weight)
    elif MODE=="inference":
    #inference
        #inference set
        infer_enc_input,infer_enc_input_pos,infer_dec_gt,infer_dec_gt_pos,infer_dec_input,infer_dec_input_pos,infer_com,infer_com_weight=read_QApair_withPos(word2id,pos2id,"./data/sample.txt",MAX_EN_LEN,MAX_DE_LEN,emd_dict,stopword,MAX_COM_VOCAB)
        print("End prepare inference data")
        inference(word_emd,wordid2posid_vec,infer_enc_input,infer_enc_input_pos,infer_dec_gt,infer_com,infer_com_weight)


