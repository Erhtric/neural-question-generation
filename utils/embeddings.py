import numpy as np
import gensim.downloader as gloader
from gensim.models import KeyedVectors
import tqdm as tqdm

class GloVe:
  def __init__(self, embedding_dimension):
    self.embedding_dimension = embedding_dimension

    try:
      self.embedding_model = KeyedVectors.load(f'./data/glove_model_{self.embedding_dimension}')
    except FileNotFoundError:
      print('[Warning] Model not found in local folder, please wait...')
      self.embedding_model = self.load_glove()
      self.embedding_model.save(f'./data/glove_model_{self.embedding_dimension}')
      print('Download finished. Model loaded!')

  def load_glove(self):
    """
    Loads a pre-trained GloVe embedding model via gensim library.

    We have a matrix that associate words to a vector of a user-defined dimension.
    """

    download_path = "glove-wiki-gigaword-{}".format(self.embedding_dimension)

    try:
      emb_model = gloader.load(download_path)
    except ValueError as e:
      print("Generic error when loading GloVe")
      print("Check embedding dimension")
      raise e

    emb_model = gloader.load(download_path)
    return emb_model

  def build_embedding_matrix(self, word_to_idx: list, vocab_size: int) -> np.ndarray:
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param vocab_size: size of the vocabulary

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the
        dataset specific vocabulary (shape |V| x d)
    """
    embedding_matrix = np.zeros((vocab_size, self.embedding_dimension), dtype=np.float32)
    oov_count = 0
    oov_words = []

    # For each word which is not present in the vocabulary we assign a random vector, otherwise we take the GloVe embedding
    for word, idx in word_to_idx.items():
      try:
        embedding_vector = self.embedding_model[word]
      except (KeyError, TypeError):
        oov_count += 1
        oov_words.append(word)
        embedding_vector = np.random.uniform(low=-0.5,
                                             high=0.5,
                                             size=self.embedding_dimension)

      embedding_matrix[idx] = embedding_vector

    # print(f'\n[Debug] {oov_count} OOV words found!\n')
    return embedding_matrix, oov_words