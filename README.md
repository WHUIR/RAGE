# RAGE
The Tensorflow implementation of "Review-driven Answer Generation for Product-related Questions in E-commerce ", WSDM 2019.


Feel free to contact me if you find any problem in the package.
sqchen@whu.edu.cn

## Requirements:
Tensorflow 1.0.0
Gensim 3.6.0
Numpy 1.13.3

## Data Preparation
To run the RAGE, you need to prepare your own data as follow:

### Training/Testing Data Format
Each training sample is a pair of question and answer with its auxiliary review snippets.
Example: 

Sessionid \t Question \t Answer \t Review_snippet1 \t Review_snippet2 … 

The question and answer are described as: word1<pos>POS tag word2<pos>POS tag … 

The review_snippet is described as: WMD|word1 word2 word3 …

Note: the data detail could be check in ./data/sample.txt. You could calculate WMD and extract the auxiliary review snippets using Gensim.


### Stop Word File
The stop word list.
Example: 

stop_word1 \n stop_word2 …

Note: the detail could be check in ./data/stopword.dic


### Vocabulary File
Each line is the word appear in training and testing dataset with its frequency. And the words are arranged in reverse order.
Example:

word1: frequency \n word2: frequency

Note: the detail could be check in ./data/wdj_word_fre.txt


 ### Pre-train Word Embedding File
You could pre-train the word embedding of dataset by Gensim, and save the model as the data format of ./data/wdj_word_emd.txt


### Word POS tag File
The maximum probability POS tag for each word in Vocabulary file, which are used for answer generation.
Example:

word1:POS_tag \n word2:POS_tag

Note: the detail could be check in ./data/wdj_word_pos.txt


## Configurations
MODE: train or inference

EPOCH: number of training epoch

BATCH_SIZE: batch size

LEARNING_RATE: learning rate

EMD_SIZE: word embedding size, POS tag embedding size and position embedding size

VOCAB_SIZE: valid vocabulary size, must add extra PAD, START, EOS

START_TOKEN: start token sign

EOS_TOKEN: end token sign

PAD: padding sign

EMD_KEEP_PROB: the dropout keep prob between embedding space to hidden space

LAYER_KEEP_PROB: the dropout keep prob between layer and layer

OUT_KEEP_PROB: the dropout keep prob for the final layer output

ENC_LAYER: number of encoder layer

DEC_LAYER: number of decoder layer

ENC_FM_LIST: number of filters of gated convolutional network for each layer in encoder, should be represented as a list [200,200,..]

DEC_FM_LIST: number of filters of gated convolutional network for each layer in decoder, should be represented as a list [200,200,..]

ENC_KWIDTH_LIST: window size of gate convolutional network for each layer in encoder, should be represented as a list[3,3,..]

DEC_KWIDTH_LIST: window size of gate convolutional network for each layer in decoder, should be represented as a list[3,3,..]

POS_SIZE: number of POS tag

MAX_COM_VOCAB: the maximum size of review weighted vocabulary

MAX_EN_LEN: the maximum length of question

MAX_DE_LEN: the maximum length of answer

SAVE_STEP: epoch to save model, when epoch%SAVE_STEP==0 the model would be saved

SAVE_MODEL_PATH: path to save trained model

## Launch the program
The main python entry is in class Runner.py. To launch the program there are several parameters must be setting as described above.
