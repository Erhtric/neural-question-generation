import argparse
from embeddings import GloVe
from configs.config import model_config

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
    glove_handler = GloVe(embedding_dimension=model_config['embedding_dimension'])

    # We will create the matrix by using only the words present in the training and validation set
    embedding_matrix_context, _ = glove_handler.build_embedding_matrix(
        word_to_idx_context[2],
        len(word_to_idx_context[2]))
    embedding_matrix_question, _ = glove_handler.build_embedding_matrix(
        word_to_idx_question[2],
        len(word_to_idx_question[2]))

    return embedding_matrix_context, embedding_matrix_question