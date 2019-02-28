# coding=UTF-8
__author__ = 'shiqian.csq'
import numpy
import codecs
import gensim
import numpy as np
import jieba
import jieba.posseg as posseg

'''load stop word'''
def load_stopword(stopword_path):
    fr_sw = codecs.open(stopword_path, 'r', 'utf-8')
    # load stop word
    print("Load stop word ")
    stop_word = []
    sw = fr_sw.readline()
    while sw:
        sw = sw.replace("\n", "")
        stop_word.append(sw.strip())
        sw = fr_sw.readline()
    return stop_word


'''load vocabulary as vocab size'''
def load_vocab(vocab_path, vocab_size,PAD=0,START=1,EOS=2):
    print("Load vocabulary as size %d.." % vocab_size)
    word2id = {}
    word2id["PAD"] = PAD
    word2id["START"] = START
    word2id["EOS"] = EOS
    fr = codecs.open(vocab_path, 'r', 'utf-8')
    index = 3
    line = fr.readline()
    while index < vocab_size:
        line = line.replace("\n", "")
        word = line.split(":")
        if word[0] == ":" or word[0] == "":
            line = fr.readline()
            continue
        word2id[word[0]] = index
        line = fr.readline()
        index += 1
    print("Vocabulary size:%d" % len(word2id))
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


'''load word embedding as dimension'''
def load_word_embedding(word2id, emd_path, dimension):
    print("Load Word Embedding")
    emd_dict = {}
    fr = codecs.open(emd_path, 'r', 'utf-8')

    # load vocab_size and dimension
    line = fr.readline()
    line = fr.readline()
    i = 0
    while line:
        i += 1
        line = line.replace("\n", "")
        ves = line.split(" ")
        # ves[0] is word,ves[1:] is embedding
        if len(ves[1:]) < dimension:
            line = fr.readline()
            continue
        if ves[0] not in word2id.keys():
            line = fr.readline()
            continue
        emd_dict[ves[0]] = numpy.array(ves[1:], dtype="float32")
        line = fr.readline()
        if 0 == i % 10000:
            print("%d word have been loaded...." % i)
    emd_dict["PAD"] = numpy.zeros(dimension, dtype='float32')
    emd_dict["START"] = numpy.zeros(dimension, dtype='float32')
    emd_dict["EOS"] = numpy.zeros(dimension, dtype='float32')
    print("Embedding Dictionary has been created as size:%d" % len(emd_dict))

    emd_mtx = numpy.array(numpy.zeros([len(emd_dict), dimension]), dtype="float32")
    for key, value in emd_dict.items():
        index = word2id.get(key)
        vec = value
        for w in range(len(vec)):
            emd_mtx[index][w] = vec[w]
    print("Embedding Martix has been created as shape:" + str(emd_mtx.shape))
    return emd_dict, emd_mtx


'''load the maximum probability POS tag of valid vocabulary'''
def load_word2pos(word2id, word2pos_path):
    pos2id = {}
    word2pos = {}
    word2pos['PAD'] = "UNK"
    word2pos['START'] = "UNK"
    word2pos['EOS'] = "UNK"
    fr_word2pos = codecs.open(word2pos_path, 'r', 'utf-8')
    line_word2pos = fr_word2pos.readline()
    pos_index = 1
    word2pos_index = 0
    while line_word2pos:
        line_word2pos = line_word2pos.replace("\n", "")
        columns = line_word2pos.split(":")
        word = columns[0]
        pos = columns[1]
        word2pos_index += 1
        if word not in word2id.keys():
            line_word2pos = fr_word2pos.readline()
            continue
        if pos not in pos2id.keys():
            pos2id[pos] = pos_index
            pos_index += 1
        if word not in word2pos.keys():
            word2pos[word] = pos
        if word2pos_index % 10000 == 0:
            print("%d word2pos have been processed" % word2pos_index)
        line_word2pos = fr_word2pos.readline()
    print("word2pos size is", len(word2pos))
    print("pos2id size is", len(pos2id))
    id2pos = {v: k for k, v in pos2id.items()}

    wordid2posid = {}
    nuk_num = 0
    for word in word2id.keys():
        wordid = word2id.get(word)
        posid = 0
        if word not in word2pos.keys():
            nuk_num += 1
            posid = 0
        else:
            pos = word2pos.get(word)
            posid = pos2id.get(pos)
        wordid2posid[wordid] = posid
    wordid2posid[0] = 0
    wordid2posid[1] = 0
    wordid2posid[2] = 0
    wordid2posid_vec = numpy.array(numpy.zeros([len(wordid2posid)]), dtype="int32")
    for key, value in wordid2posid.items():
        index = key
        wordid2posid_vec[index] = value
    print("valid vocab unknown pos num", nuk_num)
    return pos2id, id2pos, word2pos, wordid2posid, wordid2posid_vec


'''Prepare Data for train or test model'''
def read_QApair_withPos(word2id, pos2id, qa_path, max_q_len, max_a_len, word_emd, stopword, max_com_vocab=100,PAD=0,START=1,EOS=2,UNK=0):
    enc_input = []
    enc_input_pos = []
    dec_gt = []
    dec_gt_pos = []
    dec_input = []
    dec_input_pos = []
    comments = []
    com_word_weight = []

    print("QA processing..")
    fr = codecs.open(qa_path, 'r', 'utf-8')
    line = fr.readline()
    bad_data = 0
    qa_num = 0
    while line:
        qa_num += 1
        line = line.replace("\n", "")
        qa = line.split("\t")
        if len(qa) < 4:
            bad_data += 1
            line = fr.readline()
            continue
        # question
        question = qa[1].split(" ")
        qus_word = []
        qus_pos = []
        for quspos in question:
            tmp = quspos.split("<pos>")
            if tmp[0] in word2id.keys():
                qus_word.append(word2id.get(tmp[0]))
                qus_pos.append(pos2id.get(tmp[1]))
        if len(qus_word) != len(qus_pos):
            print("qus word len is not equal with pos len")
        # truncate
        if len(qus_word) >= max_q_len:
            qus_word = qus_word[0:max_q_len]
        # padding
        else:
            qus_word.extend([PAD] * (max_q_len - len(qus_word)))
        enc_input.append(qus_word)

        if len(qus_pos) >= max_q_len:
            qus_pos = qus_pos[0:max_q_len]
        else:
            qus_pos.extend([UNK] * (max_q_len - len(qus_pos)))
        enc_input_pos.append(qus_pos)

        # answer
        answer_gt = qa[2].split(" ")
        ans_gt_word = []
        ans_gt_pos = []
        for anspos in answer_gt:
            tmp = anspos.split("<pos>")
            if tmp[0] in word2id.keys():
                ans_gt_word.append(word2id.get(tmp[0]))
                ans_gt_pos.append(word2id.get(tmp[1]))
        if len(ans_gt_pos) != len(ans_gt_word):
            print("ans_gt word len is not equal with pos len")

        # add EOS and PAD
        if len(ans_gt_word) >= max_a_len - 1:
            ans_gt_word = ans_gt_word[0:max_a_len - 1]
            ans_gt_word.extend([EOS])
        else:
            ans_gt_word.extend([EOS])
            ans_gt_word.extend([PAD] * (max_a_len - len(ans_gt_word)))
        dec_gt.append(ans_gt_word)
        if len(ans_gt_pos) >= max_a_len - 1:
            ans_gt_pos = ans_gt_pos[0:max_a_len - 1]
            ans_gt_pos.extend([UNK])
        else:
            ans_gt_pos.extend([UNK] * (max_a_len - len(ans_gt_pos)))
        dec_gt_pos.append(ans_gt_pos)

        # add START and PAD
        answer = qa[2].split(" ")
        ans_in_word = []
        ans_in_pos = []
        ans_in_word.append(START)
        ans_in_pos.append(UNK)
        for anspos in answer:
            tmp = anspos.split("<pos>")
            if tmp[0] in word2id.keys():
                ans_in_word.append(word2id.get(tmp[0]))
                ans_in_pos.append(pos2id.get(tmp[1]))
        if len(ans_in_word) != len(ans_in_pos):
            print("ans in word len is not equal with pos len")
        if len(ans_in_word) >= max_a_len:
            ans_in_word = ans_in_word[0:max_a_len]
        else:
            ans_in_word.extend([PAD] * (max_a_len - len(ans_in_word)))
        dec_input.append(ans_in_word)
        if len(ans_in_pos) >= max_a_len:
            ans_in_pos = ans_in_pos[0:max_a_len]
        else:
            ans_in_pos.extend([UNK] * (max_a_len - len(ans_in_pos)))
        dec_input_pos.append(ans_in_pos)

        # comments
        raw_com_list = qa[3:]
        qa_com_list = []
        for dis_com in raw_com_list:
            if dis_com == "":
                continue
            dis_com = dis_com.split("|")
            com = dis_com[1].split(" ")
            qa_com_list.append(com)

        word_weight = wordWeight(word2id, qa_com_list, word_emd, stopword)
        com_dictionary = gensim.corpora.Dictionary(documents=qa_com_list)
        for i, t1 in com_dictionary.items():
            for j, t2 in com_dictionary.items():
                break
            break
        oov_com = []
        for id, token in com_dictionary.id2token.items():
            if token not in word2id.keys():
                continue
            oov_com.append(token)
        com_index = [word2id.get(w) for w in oov_com]
        com_weight = [word_weight.get(w, 0.0) for w in oov_com]
        if len(com_index) >= max_com_vocab:
            com_index = com_index[0:max_com_vocab]
            print("Wrong with max comments vocab")
        else:
            com_index.extend([0] * (max_com_vocab - len(com_index)))
        if len(com_weight) >= max_com_vocab:
            com_weight = com_weight[0:max_com_vocab]
            print("Wrong with max comments vocab")
        else:
            com_weight.extend([0.0] * (max_com_vocab - len(com_weight)))
        comments.append(com_index)
        com_word_weight.append(com_weight)

        line = fr.readline()

    print("Question Set size:%d" % len(enc_input))
    print("Question pos Set size:%d" % len(enc_input_pos))
    print("Answer Set size:%d" % len(dec_gt))
    print("Answer pos Set size:%d" % len(dec_gt_pos))
    print("decoder input  Set:%d" % len(dec_input))
    print("decoder input  pos Set:%d" % len(dec_input_pos))
    print("Comment index size:%d" % len(comments))
    print("Comment weigth size:%d" % len(com_word_weight))
    return enc_input, enc_input_pos, dec_gt, dec_gt_pos, dec_input, dec_input_pos, comments, com_word_weight


''' Calculate weighted vocabulary'''
def wordWeight(word2id, raw_com_list, word_emd, stopword):
    com_list = []
    OOV_com_list = []
    # remove OOV word
    for com in raw_com_list:
        com = [token for token in com if token in word2id.keys()]
        OOV_com_list.append(com)
    # remove stop word
    for com in OOV_com_list:
        com = [token for token in com if token not in stopword]
        com_list.append(com)

    dictionary = gensim.corpora.Dictionary(documents=com_list)
    vocab_len = len(dictionary)
    nq = len(com_list)

    cos_matrix = np.zeros((vocab_len, vocab_len), dtype='float32')
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 == t2:
                continue
            cos_matrix[i, j] = 0.5 + 0.5 * np.dot(word_emd[t1], word_emd[t2]) / (np.linalg.norm(word_emd[t1]) * np.linalg.norm(word_emd[t2]))

    def nbow(document):
        d = np.zeros(vocab_len, dtype='float32')
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        for idx, freq in nbow:
            if freq != 0:
                d[idx] = 1
        return d

    # fs/nq
    fs = np.zeros(vocab_len, dtype='float32')
    one_hot_coms = []
    for d in com_list:
        one_hot_d = nbow(d)
        one_hot_coms.append(one_hot_d)
        fs += one_hot_d
    fs = fs / nq

    word2rel = []
    for id, t in dictionary.id2token.items():
        total_rel = 0.0
        for one_hot_com in one_hot_coms:
            if one_hot_com[id] != 0.0:
                rel = np.multiply(one_hot_com, cos_matrix[id])
                max_id = np.where(rel == np.max(rel))
                try:
                    total_rel += rel[max_id]
                except ValueError:
                    max_id = max_id[0][0]
                    total_rel += rel[max_id]
                    print("There are more than one max similarity")
        word2rel.append(total_rel[0])
    word2rel = np.array(word2rel)

    weight = np.multiply(fs, word2rel)
    max_weight_id = np.where(weight == np.max(weight))
    max_weight = weight[max_weight_id]
    if max_weight[0] == float("nan"):
        max_weight[0] = 1.0
    norm_weight = weight / max_weight[0]
    word_weight = {}
    for t, w in zip(dictionary.id2token.values(), norm_weight):
        if w == float("nan"):
            w = 1.0
        word_weight[t] = w
    return word_weight


def SavedataSet(dataset, output_path):
    fout = codecs.open(output_path, 'w', 'utf-8')
    index = 0
    for example in dataset:
        example_str = [str(w) for w in example]
        fout.write(" ".join(example_str) + "\n")
        index += 1
    print("Write Dataset size %d" % index)

def LoaddataSet(dataset_path):
    fr = codecs.open(dataset_path, 'r', 'utf-8')
    dataset = []
    line = fr.readline()
    while line:
        line = line.replace("\n", "")
        example = line.split(" ")
        example_int = [int(w) for w in example]
        dataset.append(example_int)
        line = fr.readline()
    print("Load Dataset size is %d" % len(dataset))
    return dataset


def LoaddataSetFloat(dataset_path):
    fr = codecs.open(dataset_path, 'r', 'utf-8')
    dataset = []
    line = fr.readline()
    while line:
        line = line.replace("\n", "")
        if 'nan' in line:
            print("Bad data")
            line = line.replace("nan", "1.0")
            print(line)
        example = line.split(" ")
        example_int = [float(w) for w in example]
        dataset.append(example_int)
        line = fr.readline()
    print("Load Dataset size is %d" % len(dataset))
    return dataset

def getposqa(data_path,out_path):
    fr = codecs.open(data_path, 'r', 'utf-8')
    fout=codecs.open(out_path,'w','utf-8')
    line = fr.readline()
    while line:
        line = line.replace("\n", "")
        columns = line.split("\t")
        sid=columns[0]
        question=columns[1].split(" ")
        question=[w+"<pos>n" for w in question]
        answer=columns[2].split(" ")
        answer = [w + "<pos>n" for w in answer]
        comments=columns[3:]
        line2=sid+"\t"+" ".join(question)+"\t"+" ".join(answer)+"\t"+"\t".join(comments)+"\n"
        fout.write(line2)
        line=fr.readline()
    print("end")




if __name__=='__main__':
    #getposqa("./data/out_avg_qa.txt","./data/sample.txt")
    stop_word=load_stopword("./data/stopword.dic")
    word2id,id2word=load_vocab("./data/wdj_word_fre.txt",8000+3)
    emd_dict,word_emd=load_word_embedding(word2id, "./data/wdj_word_emd.txt", 200)
    pos2id, id2pos, word2pos, wordid2posid, wordid2posid_vec=load_word2pos(word2id,"./data/wdj_word_pos.txt")
    read_QApair_withPos(word2id, pos2id,"./data/sample.txt", 30, 40, emd_dict, stop_word, max_com_vocab=100, PAD=0,
                        START=1, EOS=2, UNK=0)
    print()
