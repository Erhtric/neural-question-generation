import numpy as np
import pandas as pd
import json
import re
import pickle
import os
from typing import NamedTuple
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import tensorflow as tf

import spacy
from spacy.attrs import ORTH, TAG
from spacy.language import Language
import en_core_web_sm

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# disable chained assignments to avoid annoying warning
pd.options.mode.chained_assignment = None

class Dataset(NamedTuple):
  """
  This class represent a a 3-way split processed dataset.
  """
  # Reference :- https://github.com/topper-123/Articles/blob/master/New-interesting-data-types-in-Python3.rst
  train: tf.data.Dataset
  val: tf.data.Dataset
  test: tf.data.Dataset

class SQuAD:
  def __init__(self):
    self.random_seed = None
    self.squad_df = None
    self.preproc_squad_df = None
    self.tokenizer = None
    self.buffer_size = 0

    # Spacy NLP utilities
    self.nlp = spacy.load("en_core_web_sm")
    self.nlp = en_core_web_sm.load()
    self.nlp.tokenizer.add_special_case('<sos>', [{ORTH: "<sos>"}])
    self.nlp.tokenizer.add_special_case('<eos>', [{ORTH: "<eos>"}])
    self.nlp.tokenizer.add_special_case('<pad>', [{ORTH: "<pad>"}])
    self.nlp.tokenizer.add_special_case('<unk>', [{ORTH: "<unk>"}])

    # @Language.component("pos_postprocessor_pipe")
    # def pos_postprocessor_pipe(doc) :
    #   for token in doc:
    #       if token.text == '<pad>':
    #           token.pos_ = {TAG: '<pad>'}
    #   return doc

    # self.nlp.add_pipe("pos_postprocessor_pipe", before='parser')

  def __call__(self, dataset_config: dict, path: dict, tokenized: bool = True, compute_pos: bool = False, tensor_type: bool = True):
    """The method loads a subset of the SQuAD dataset, preprocess it and optionally it returns
    it tokenized.

    Args:
        dataset_config: a dictionary containing the dataset configuration infos
        path: a dictionary containing the dataset path infos
        tokenized (boolean): specifies if the context and question data should be both tokenized
        tensor_type (boolean): specifies if the context and question data should be converted to tensors

    Returns (depending on the input parameters):
        pd.DataFrame: training dataset
        pd.DataFrame: validation dataset
        pd.DataFrame: testing dataset
          OR
        NamedTuple: dataset, (dict, dict, dict)
    """
    self.random_seed = dataset_config['random_seed']
    self.buffer_size = dataset_config['buffer_size']
    self.batch_size = dataset_config['batch_size']
    self.train_size = dataset_config['train_size']
    self.test_size = dataset_config['test_size']
    self.training_json_path = path['training_json_path']
    self.save_pkl_path = path['save_pkl_path']
    self.process_data_dirpath = path["process_data"]
    self.max_length_context = 0
    self.max_length_question = 0

    # if os.path.exists(os.path.join(self.process_data_dirpath, 'train_dataset')) and \
    #         os.path.exists(os.path.join(self.process_data_dirpath, 'val_dataset')) and \
    #         os.path.exists(os.path.join(self.process_data_dirpath, 'test_dataset')):
    #   print('Loading datasets from files...')

    #   train_dataset = tf.data.experimental.load(os.path.join(self.process_data_dirpath, 'train_dataset'))
    #   val_dataset = tf.data.experimental.load(os.path.join(self.process_data_dirpath, 'val_dataset'))
    #   test_dataset = tf.data.experimental.load(os.path.join(self.process_data_dirpath, 'test_dataset'))

    #   dataset = Dataset(
    #       train=train_dataset,
    #       val=val_dataset,
    #       test=test_dataset)

    #   return dataset

    # Load dataset from file
    self.load_dataset(dataset_config['num_examples'])

    # Extract answer
    self.extract_answer()

    # Preprocess context and question
    self.preprocess()
    self.compute_max_length()

    # Perform splitting
    X_train, y_train, X_val, y_val, X_test, y_test = self.split_train_val(self.preproc_squad_df)
    X_train = X_train.iloc[:100]
    y_train = y_train.iloc[:100]

    X_val = X_val.iloc[:100]
    y_val = y_val.iloc[:100]

    X_test = X_test.iloc[:100]
    y_test = y_test.iloc[:100]

    # Initialize Tokenizer for the source: in our case the context sentences
    self.tokenizer_context = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                                   oov_token='<unk>',
                                                                   num_words=dataset_config['num_words_context'])
    # initialize also for the target, namely the question sentences
    self.tokenizer_question = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                                   oov_token='<unk>',
                                                                   num_words=dataset_config['num_words_question'])

    self.tokenizer_context_pos = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                                      # oov_token='<unk>',
                                                                      lower=False,)

    self.tokenizer_question_pos = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                                        # oov_token='<unk>',
                                                                        lower=False,)

    if tokenized:
      X_train_tokenized, word_to_idx_train_context = self.__tokenize_context(X_train, test=False)
      y_train_tokenized, word_to_idx_train_question = self.__tokenize_question(y_train, test=False)

      # update the max length for the other splits
      self.max_length_context = X_train_tokenized.context.iloc[0].shape[0]
      self.max_length_question = y_train_tokenized.iloc[0].shape[0]

      if compute_pos:
        print("Computing POS tags for the train set...")
        X_train_tokenized = self.compute_pos_tags_context(X_train_tokenized)
        y_train_tokenized = self.compute_pos_tags_question(y_train_tokenized)

      X_val_tokenized, word_to_idx_val_context = self.__tokenize_context(X_val, test=False)
      y_val_tokenized, word_to_idx_val_question = self.__tokenize_question(y_val, test=False)

      if compute_pos:
        print("Computing POS tags for the val set...")
        X_val_tokenized = self.compute_pos_tags_context(X_val_tokenized)
        y_val_tokenized = self.compute_pos_tags_question(y_val_tokenized)

      # The test set should handle the oov words as unkwown words
      X_test_tokenized, word_to_idx_test_context = self.__tokenize_context(X_test, test=True)
      y_test_tokenized, word_to_idx_test_question = self.__tokenize_question(y_test, test=True)

      if compute_pos:
        print("Computing POS tags for the test set...")
        X_test_tokenized = self.compute_pos_tags_context(X_test_tokenized)
        y_test_tokenized = self.compute_pos_tags_question(y_test_tokenized)

      word_to_idx_context = (word_to_idx_train_context, word_to_idx_val_context, word_to_idx_test_context)
      word_to_idx_question = (word_to_idx_train_question, word_to_idx_val_question, word_to_idx_test_question)

      if tensor_type:
        # Returns tf.Data.Dataset objects (tokenized)
        train_dataset = self.to_tensor(X_train_tokenized, y_train_tokenized)
        val_dataset = self.to_tensor(X_val_tokenized, y_val_tokenized)
        test_dataset = self.to_tensor(X_test_tokenized, y_test_tokenized)

        # # Configure the dataset for performance
        # train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        # val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        # test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        if self.process_data_dirpath:
          os.makedirs(self.process_data_dirpath, exist_ok=True)

          tf.data.experimental.save(train_dataset, os.path.join(self.process_data_dirpath, 'train_dataset'))
          tf.data.experimental.save(val_dataset, os.path.join(self.process_data_dirpath, 'val_dataset'))
          tf.data.experimental.save(test_dataset, os.path.join(self.process_data_dirpath, 'test_dataset'))

        dataset = Dataset(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset)

        return dataset, word_to_idx_context, word_to_idx_question
      else:
        # Returns pd.DataFrame objects (tokenized)
        return X_train_tokenized, y_train_tokenized, X_val_tokenized, y_val_tokenized, X_test_tokenized, y_test_tokenized
    else:
      return X_train, y_train, X_val, y_val, X_test, y_test

  def compute_max_length(self):
    context_list = list(self.preproc_squad_df.context)
    question_list = list(self.preproc_squad_df.question)

    context_length = [len(sen.split()) for sen in context_list]
    question_length = [len(sen.split()) for sen in question_list]

    self.max_length_context = int(np.quantile(context_length, 0.995))
    self.max_length_question = int(np.quantile(question_length, 0.995))

  def load_dataset(self, num_examples):
    """
    Extract the dataset from the json file. Already grouped by title.

    :param path: [Optional] specifies the local path where the training_set.json file is located

    :return
        - the extracted dataset in a dataframe format
    """
    if os.path.exists(self.save_pkl_path):
      print('File already exists! Loading from .pkl...\n')
      print(f'Dir path {self.save_pkl_path}')
      self.squad_df = pd.read_pickle(self.save_pkl_path)
      self.squad_df = self.squad_df[:num_examples]
    else:
      print('Loading from .json...\n')
      print(f'Dir path {self.training_json_path}')
      with open(self.training_json_path) as f:
          data = json.load(f)

      df_array = []
      for current_subject in data['data']:
      # for current_subject in data:
          title = current_subject['title']

          for current_context in current_subject['paragraphs']:
              context = current_context['context']

              for current_qas in current_context['qas']:
                # Each qas is a list made of id, question, answers
                id = current_qas['id']
                question = current_qas['question']
                answers = current_qas['answers']

                for current_answer in current_qas['answers']:
                  answer_start = current_answer['answer_start']
                  text = current_answer['text']

                  record = { "id": id,
                            "title": title,
                            "context": context,
                            "question": question,
                            "answer_start": answer_start,
                            "answer": text
                            }

                  df_array.append(record)
      # Save file
      pd.to_pickle(pd.DataFrame(df_array), self.save_pkl_path)
      self.squad_df = pd.DataFrame(df_array)[:num_examples]

  def preprocess(self):
    df = self.squad_df.copy()

    # Pre-processing context
    context = list(df.context)
    preproc_context = []

    for c in context:
      c = self.__preprocess_sentence(c, question=False)
      preproc_context.append(c)

    df.context = preproc_context

    # Pre-processing questions
    question = list(df.question)
    preproc_question = []

    for q in question:
      q = self.__preprocess_sentence(q, question=True)
      preproc_question.append(q)

    df.question = preproc_question

    # Remove features that are not useful
    df = df.drop(['id'], axis=1)
    self.preproc_squad_df = df

  def __preprocess_sentence(self, sen, question):
    # Creating a space between a word and the punctuation following it
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sen = re.sub(r"([?.!,¿])", r" \1 ", sen)
    sen = re.sub(r'[" "]+', " ", sen)

    # Replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sen = re.sub(r"[^a-zA-Z0-9?!.,¿]+", " ", sen)

    sen = sen.strip()

    # Adding a start and an end token to the sentence so that the model know when to
    # start and stop predicting.
    # if not question: sen = '<SOS> ' + sen + ' <EOS>'
    sen = '<SOS> ' + sen + ' <EOS>'
    return sen

  def __answer_start_end(self, df):
    """
    Creates a list of starting indexes and ending indexes for the answers.

    :param df: the target Dataframe

    :return: a dataframe containing the start and the end indexes foreach answer (ending index is excluded).

    """
    start_idx = df.answer_start
    end_idx = [start + len(list(answer)) for start, answer in zip(list(start_idx), list(df.answer))]
    return pd.DataFrame(list(zip(start_idx, end_idx)), columns=['start', 'end'])

  def split_train_val(self, df):
    """
    This method splits the dataframe in training and test sets, or eventually, in training, validation and test sets.

    Args
        :param df: the target Dataframe

    Returns:
        - Data and labels for training, validation and test sets if val is True
        - Data and labels for training and test sets if val is False

    """
    # Maybe we have also to return the index for the starting answer
    X = df.drop(['answer_start', 'question', 'answer'], axis=1).copy()
    idx = self.__answer_start_end(df)
    X['start'] = idx['start']
    X['end'] = idx['end']
    y = df['question']

    # In the first step we will split the data in training and remaining dataset
    splitter = GroupShuffleSplit(train_size=self.train_size, n_splits=2, random_state=self.random_seed)
    split = splitter.split(X, groups=X['title'])
    train_idx, rem_idx = next(split)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_rem = X.iloc[rem_idx]
    y_rem = y.iloc[rem_idx]


    # Val and test test accounts for the remaining percentage of the total data
    splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=2, random_state=self.random_seed)
    split = splitter.split(X_rem, groups=X_rem['title'])
    val_idx, test_idx = next(split)

    X_val = X_rem.iloc[val_idx]
    y_val = y_rem.iloc[val_idx]

    X_test = X_rem.iloc[test_idx]
    y_test = y_rem.iloc[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

  def compute_pos_tags_context(self, X_tokenized):
    context = X_tokenized.context

    # Spacy requires words so we have to untokenize the context
    context = self.tokenizer_context.sequences_to_texts(context)

    pos_tags = []
    for idx, sentence in tqdm(enumerate(context), desc="Computing POS tags for the context...", total=len(context), unit="seq"):
      # Generate POS tags
      doc = self.nlp(sentence)
      pos_tags_str = " ".join([w.pos_ for w in doc])

      if len(sentence.split()) != len(pos_tags_str.split()):
        # TODO: undestand why this happens
        # print("Index: ", context[idx].split())
        # print("Sentence: ", sentence.split(), len(sentence.split()))
        # print("POS tags: ", pos_tags_str.split(), len(pos_tags_str.split()))
        pos_tags_str = " ".join([w.pos_ for w in doc[:-1]])
        # print("The number of words in the sentence and the number of POS tags are different! The last word is removed.")

      # Fit the tokenizer on the POS tags
      self.tokenizer_context_pos.fit_on_texts([pos_tags_str])

      # The sentence now is the concatenation of the sentence and the POS tags
      sentence = self.tokenizer_context.texts_to_sequences([sentence.strip()])[0]
      pos_tags_str = self.tokenizer_context_pos.texts_to_sequences([pos_tags_str.strip()])[0]

      pos_tags.append(pos_tags_str)

    # Add the pos tags to the dataframe
    X_tokenized["context_pos"] = pd.Series(pos_tags)

    return X_tokenized

  def compute_pos_tags_question(self, Y_tokenized):
    question = Y_tokenized.copy()

    # Spacy requires words so we have to untokenize the question
    question = self.tokenizer_question.sequences_to_texts(question)

    pos_tags = []
    for sentence in tqdm(question, desc="Computing POS tags for the question...", total=len(question), unit="seq"):
      # Generate POS tags
      doc = self.nlp(sentence)
      pos_tags_str = " ".join([w.pos_ for w in doc])

      if len(sentence.split()) != len(pos_tags_str.split()):
        # TODO: undestand why this happens
        # print("Index: ", context[idx].split())
        # print("Sentence: ", sentence.split(), len(sentence.split()))
        # print("POS tags: ", pos_tags_str.split(), len(pos_tags_str.split()))
        pos_tags_str = " ".join([w.pos_ for w in doc[:-1]])
        # print("The number of words in the sentence and the number of POS tags are different! The last word is removed.")

      # Fit the tokenizer on the POS tags
      self.tokenizer_question_pos.fit_on_texts([pos_tags_str])

      # The sentence now is the concatenation of the sentence and the POS tags
      sentence = self.tokenizer_question.texts_to_sequences([sentence])[0]
      pos_tags_str = self.tokenizer_question_pos.texts_to_sequences([pos_tags_str])[0]

      pos_tags.append(pos_tags_str)

    # Add the pos tags to the dataframe
    Y_tokenized_pos = pd.DataFrame(Y_tokenized, columns=["question"])
    Y_tokenized_pos["question_pos"] = pd.Series(pos_tags)

    return Y_tokenized_pos

  def __tokenize_context(self, X, test):
    context = X.context
    if not test: self.tokenizer_context.fit_on_texts(context)
    context_tf = self.tokenizer_context.texts_to_sequences(context)

    if self.max_length_context != 0:
      context_tf_pad = tf.keras.preprocessing.sequence.pad_sequences(context_tf, maxlen=self.max_length_context, padding='post')
    else:
      context_tf_pad = tf.keras.preprocessing.sequence.pad_sequences(context_tf, padding='post')

    for i, _ in enumerate(context):
      X['context'].iloc[i] = context_tf_pad[i]

    # Add the padding
    self.tokenizer_context.word_index['<pad>'] = 0
    self.tokenizer_context.index_word[0] = '<pad>'

    self.tokenizer_context_pos.word_index['<PAD>'] = 0
    self.tokenizer_context_pos.index_word[0] = '<PAD>'

    return X, self.tokenizer_context.word_index

  def __tokenize_question(self, y, test):
    question = y
    if not test: self.tokenizer_question.fit_on_texts(question)
    question_tf = self.tokenizer_question.texts_to_sequences(question)

    if self.max_length_question != 0:
      question_tf_pad = tf.keras.preprocessing.sequence.pad_sequences(question_tf, maxlen=self.max_length_question, padding='post')
    else:
      question_tf_pad = tf.keras.preprocessing.sequence.pad_sequences(question_tf, padding='post')

    for i, _ in enumerate(question):
      y.iloc[i] = question_tf_pad[i]

    # Add the padding
    self.tokenizer_question.word_index['<pad>'] = 0
    self.tokenizer_question.index_word[0] = '<pad>'

    self.tokenizer_question_pos.word_index['<PAD>'] = 0
    self.tokenizer_question_pos.index_word[0] = '<PAD>'

    return y, self.tokenizer_question.word_index

  def extract_answer(self):
    df = self.squad_df.copy()
    start_end = self.__answer_start_end(df)
    context = list(df.context)

    selected_sentences = []
    for i, par in enumerate(context):
      sentences = sent_tokenize(par)
      start = start_end.iloc[i].start
      end = start_end.iloc[i].end
      right_sentence = ""
      context_characters = 0

      for j, sen in enumerate(sentences):
        sen += ' '
        context_characters += len(sen)
        # If the answer is completely in the current sentence
        if(start < context_characters and end <= context_characters):
          right_sentence = sen
          selected_sentences.append(right_sentence)
          break
        # the answer is in both the current and the next sentence
        if(start < context_characters and end > context_characters):
          right_sentence = sen + sentences[j+1]
          selected_sentences.append(right_sentence)
          break

    self.squad_df.context = selected_sentences

  def to_tensor(self, X, y, train=True):
    # X = X.context.copy()
    # y = y.copy()

    X = X.context.copy()
    X_pos = X.context_pos.copy()
    y = y.question.copy()
    y_pos = y.question_pos.copy()

    # Reference:- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(list(X), tf.int64), tf.cast(list(X_pos), tf.int64)),
        (tf.cast(list(y), tf.int64), tf.cast(list(y_pos), tf.int64))
      )

    if train:
      dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
    else:
      dataset = dataset.batch(self.batch_size, drop_remainder=True)

    return dataset