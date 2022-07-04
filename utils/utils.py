import argparse
from embeddings import GloVe

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def prepare_embeddings(word_to_idx_context, word_to_idx_question):
    # Initalize the handler for GloVe
    glove_handler = GloVe(embedding_dimension=300)

    # We will create the matrix by using only the words present in the training and validation set
    embedding_matrix_context, oov_words_context = glove_handler.build_embedding_matrix(word_to_idx_context[2], len(word_to_idx_context[2]))

    embedding_matrix_question, oov_words_question = glove_handler.build_embedding_matrix(word_to_idx_question[2], len(word_to_idx_question[2]))

    return embedding_matrix_context, embedding_matrix_question