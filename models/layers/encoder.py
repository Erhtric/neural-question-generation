import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, LSTM, Bidirectional, Input, Concatenate, SpatialDropout1D

class Encoder(Model):
  def __init__(self, model_config, embedding_matrix, **kwargs):
    super(Encoder, self).__init__(**kwargs)
    self.batch_size = model_config['batch_size']

    self.encoder_input = Input(shape=(model_config['max_length_context'],),
                             batch_size=self.batch_size,
                             dtype=tf.int32,
                             name='Context')
    self.embedding = Embedding(input_dim=embedding_matrix.shape[0],
                               output_dim=embedding_matrix.shape[1],
                               input_length=model_config['max_length_context'],
                               embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                               trainable=False,
                               mask_zero=True,
                               name='Context_embedding')

    self.spatial_dropout = SpatialDropout1D(model_config['dropout_rate'])

    self.bi_lstm = Bidirectional(LSTM(model_config['enc_units']//2,
                                      return_sequences=True,
                                      return_state=True,
                                      dropout=model_config['dropout_rate'],
                                      kernel_regularizer=tf.keras.regularizers.L2(model_config['regularizer'])), 
                                  name='Context_encoding',
                                  merge_mode='concat')

    self.concatenate = Concatenate(axis=1, name='Merge')

  def call(self, inputs, state=None, training=False):
    # 1. The input is a tokenized and padded sentence containing the answer from the context
    # 2. The embedding layer looks up for the embedding for each token, the mask is automatically produced
    vectors = self.embedding(inputs)

    vectors = self.spatial_dropout(vectors)

    # 3. The Bi-LSTM processes the embedding sequence forward and backward:
    #     output shape: ('batch', 'max_length_context', 'units')
    #     hidden state shape: fw ('batch', 'units//2'), bw ('batch', 'units//2')
    #     cell state shape: fw ('batch', 'units//2'), bw ('batch', 'units//2')
    output, forward_h, forward_c, backward_h, backward_c = self.bi_lstm(vectors, initial_state=state, training=training)

    # 4. Concatenate the forward and the backward states
    h = self.concatenate([forward_h, backward_h])
    c = self.concatenate([forward_c, backward_c])
    encoder_state = [h, c]

    # 5. Return the new sequence processed by the encoder and its state
    return [output, encoder_state]

  # Reference :- https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
  def build_graph(self):
    return tf.keras.Model(inputs=self.encoder_input, outputs=self.call(self.encoder_input))

  def plot_model(self):
    return tf.keras.utils.plot_model(
        self.build_graph(),
        to_file="encoder.jpg",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True
    )