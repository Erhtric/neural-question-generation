import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, LSTM, Input, Dense, AdditiveAttention, Concatenate

class Decoder(Model):
  def __init__(self, model_config, embedding_matrix, **kwargs):
    super().__init__(**kwargs)
    self.batch_size = model_config['batch_size']
    self.input_dim = embedding_matrix.shape[0]
    self.output_dim = embedding_matrix.shape[1]
    self.max_length_context = model_config['max_length_context']
    self.max_length_question = model_config['max_length_question']
    self.enc_units = model_config['enc_units']
    self.dec_units = model_config['dec_units']

    # Attributes
    self.new_token_input = Input(shape=(1),
                           batch_size=self.batch_size,
                           name='Token_t')
    self.enc_output_input = Input(shape=(self.max_length_context, self.enc_units),
                            batch_size=self.batch_size,
                            name='Enc_output')

    self.embedding = Embedding(input_dim=self.input_dim,
                               output_dim=self.output_dim,
                               embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                               trainable=False,
                               mask_zero=True,
                               name='Question_embedding')

    self.lstm_layer_1 = LSTM(self.dec_units,
                           return_sequences=True,
                           return_state=True,
                           dropout=model_config['dropout_rate'],
                           kernel_regularizer=tf.keras.regularizers.L2(model_config['regularizer']),
                           name='Decoding_question_1')

    self.lstm_layer_2 = LSTM(self.dec_units,
                           return_sequences=True,
                           return_state=True,
                           dropout=model_config['dropout_rate'],
                           kernel_regularizer=tf.keras.regularizers.L2(model_config['regularizer']),
                           name='Decoding_question_2')

    # Additive attention layer, a.k.a. Bahdanau-style attention.
    self.attention = AdditiveAttention(name='Attention_head')

    self.concatenate = Concatenate(axis=-1, name='Merge')

    self.fc1 = Dense(self.dec_units * 2, activation=tf.keras.activations.tanh, use_bias=False, name='Dense_Wt')

    self.dropout = tf.keras.layers.Dropout(model_config['dropout_rate'])

    # Compute the logits
    self.decoder_logits = Dense(self.input_dim, use_bias=False, name='Logits_Ws')

  def call(self, new_token, enc_output, state=None, training=False):
    # 1. The embedding layer looks up for the embedding for each token, masks is automatically computed
    # vectors shape: (batch_size, 1, embedding_dimension)
    vectors = self.embedding(new_token)
    if tf.shape(vectors).shape == 2: vectors = tf.expand_dims(vectors, axis=1)

    # 2. Process one step with the LSTM
    # LSTM expects inputs of shape: (batch_size, timestep, feature)
    output, h, c = self.lstm_layer_1(vectors, initial_state=state, training=training)
    output, h, c = self.lstm_layer_2(output, initial_state=(h, c), training=training)

    # 4. Use the LSTM cell output as the query for the attention over the encoder output, that is the value.
    # The mask is automatically passed as argument by the keras backend.
    context_vector, attention_weights = self.attention([output, enc_output],
                                                       training=training,
                                                       return_attention_scores=True)

    # 5. Join the context_vector and cell output [ct; ht] shape: (batch t, value_units + query_units)
    output_and_context_vector = self.concatenate([context_vector, output])

    # at = tanh(Wt@[ht, ct])
    attention_vector = self.fc1(output_and_context_vector)

    attention_vector = self.dropout(attention_vector)

    # logits = softmax(Ws@at), it produces unscaled logits
    logits = self.decoder_logits(attention_vector)

    return [logits, attention_weights, (h, c)]

  # Reference :- https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
  def build_graph(self):
    # return tf.keras.Model(inputs=[self.new_token_input, self.enc_output_input], outputs=self.call([self.new_token_input, self.enc_output_input]))
    new_token_input = Input(shape=(1), batch_size=self.batch_size, name='new_token')
    enc_output_input = Input(shape=(self.max_length_context, self.enc_units), batch_size=self.batch_size, name='enc_output')
    model = tf.keras.Model(inputs=[new_token_input, enc_output_input], outputs=self.call(new_token_input, enc_output_input))
    return model

  def plot_model(self, filepath: str = "./pictures/decoder.jpg"):
    return tf.keras.utils.plot_model(
        self.build_graph(),
        to_file=filepath,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True
    )