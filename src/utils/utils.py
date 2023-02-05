from matplotlib import pyplot as plt
from .embeddings import GloVe
from configs.config import model_config
from models.trainers.trainer import Trainer
from typing import NamedTuple
import numpy as np
import re
import tensorflow as tf

def prepare_embeddings(word_to_idx_context: list, word_to_idx_question: list):
    """Creates the embedding matrix for the context and question words by using the pretrained GloVe embeddings.

    Args:
        word_to_idx_context (list): list containing the word to index mappings for the context words for the split dataset.
        word_to_idx_question (list): list containing the word to index mappings for the question words for the split dataset.

    Returns:
        the embedding matrices for the context and question words.
    """
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
def build_trainer(model_config: dict, embedding_matrix_context: np.ndarray, embedding_matrix_question:np.ndarray, compile_info: dict):
    """Builds the model for the model configuration and compiling information given.

    Args:
        model_config (dict): dictionary containing the model configuration
        embedding_matrix_context (numpy.array): embedding matrix for the context words.
        embedding_matrix_question (numpy.array): embedding matrix for the question words.
        compile_info (dict): dictionary containing the compile information for the model.

    Returns:
        model: compiled Keras model
    """
    print(model_config)
    model = Trainer(model_config, embedding_matrix_context=embedding_matrix_context, embedding_matrix_question=embedding_matrix_question)
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
    """Plot the history values resulting from the training. The history is a dictionary with the following keys:
    1. loss
    2. accuracy and masked accuracy
    3. perplexity

    both from the validation and the test sets.
    """
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

def preprocess_input(sen, data_generator):
    sen = re.sub(r"([?.!,¿])", r" \1 ", sen)
    sen = re.sub(r'[" "]+', " ", sen)

    # Replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sen = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", sen)

    sen = sen.strip()

    # Adding a start and an end token to the sentence so that the model know when to
    # start and stop predicting.
    # if not question: sen = '<SOS> ' + sen + ' <EOS>'
    sen = '<SOS> ' + sen + ' <EOS>'
    sen = data_generator.tokenizer_context.texts_to_sequences([sen])[0]
    sen = tf.keras.preprocessing.sequence.pad_sequences([sen],
                                                        maxlen=data_generator.max_length_context,
                                                        padding='post')
    return sen

def compute_pos_batch():
    pass