import argparse
from matplotlib import pyplot as plt
from embeddings import GloVe
from configs.config import model_config
from models.trainers import Trainer
from typing import NamedTuple

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

# Utility function in order to build the compiled model
def build_trainer(model_config,
                embedding_matrix_context,
                embedding_matrix_question,
                compile_info):
    model = Trainer(model_config,
                  embedding_matrix_context=embedding_matrix_context,
                  embedding_matrix_question=embedding_matrix_question)

    model.compile(**compile_info)
    return model

def train_model(model,
                dataset: NamedTuple,
                training_info):
    """
    Training routine for the Keras model.
    At the end of the training, retrieved History data is shown.

    :param model: Keras built model
    :param dataset: the split dataset
    :param training_info: dictionary storing model fit() argument information

    :return
        model: trained Keras model
    """
    print("Start training \nParameters: {}".format(training_info))
    history = model.fit(dataset.train,
                        validation_data=dataset.val,
                        **training_info)
    print("Training completed")
    return history, model

def plot_history(hist):
    fig, axs = plt.subplots(1, 3, figsize=(20,5))

    fig.suptitle('Training trends')
    axs[0].plot(hist.history['accuracy'])
    axs[0].plot(hist.history['val_accuracy'])
    axs[0].plot(hist.history['masked_accuracy'])
    axs[0].plot(hist.history['val_masked_accuracy'])
    axs[0].set_title("model accuracy")
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train_acc', 'val_acc', 'train_mask_acc', 'val_mask_acc'], loc='best')

    axs[1].plot(hist.history['loss'])
    axs[1].plot(hist.history['val_loss'])
    axs[1].set_title("model batch loss")
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train_loss', 'val_loss'], loc='best')

    axs[2].plot(hist.history['perplexity'])
    axs[2].plot(hist.history['val_perplexity'])
    axs[2].set_title("model perplexity")
    axs[2].set_ylabel('perplexity')
    axs[2].set_xlabel('epoch')
    axs[2].legend(['train_perplexity', 'val_perplexity'], loc='best')

    plt.show()