import imp
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.metrics import Accuracy, Mean
from models.layers.encoder import Encoder
from models.layers.decoder import Decoder
from models.layers.masking import Masking
from metrics import MaskedAccuracy, Perplexity

class Trainer(Model):
  def __init__(self, model_config, embedding_matrix_context, embedding_matrix_question, **kwargs):
    """
    Creates a model to be trained. It builds the both the encoder and the decoder by
    exploiting the tf.keras.Model Sub-Classing API.
    Also it defines a wrapper to use the tf.function compilation for the tensorflow computational
    graph.
    """
    super(Trainer, self).__init__(**kwargs)
    self.context_input = Input(shape=(model_config['max_length_context']), batch_size=model_config['batch_size'])
    self.question_input = Input(shape=(model_config['max_length_question']), batch_size=model_config['batch_size'])
    self.masking = Masking(mask_value=0)
    self.encoder = Encoder(model_config=model_config, embedding_matrix=embedding_matrix_context)
    self.decoder = Decoder(model_config=model_config, embedding_matrix=embedding_matrix_question)

    # Attributes
    self.max_length_question = model_config['max_length_question']
    self.batch_size = model_config['batch_size']

    # Performance metrics
    self.train_accuracy = Accuracy()
    self.train_accuracy_sentence = MaskedAccuracy()
    self.test_accuracy = Accuracy()
    self.test_accuracy_sentence = MaskedAccuracy()
    self.train_perplexity = Perplexity()
    self.test_perplexity = Perplexity()

    # Loss tracker, resets at the beginning of each epoch
    self.train_loss_tracker = Mean(name='loss')
    self.test_loss_tracker = Mean(name='loss')

  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be called
    # automatically at the start of each epoch or at the start of `evaluate()`.
    # If you don't implement this property, you have to call # `reset_states()`
    # yourself at the time of your choosing.
    return [self.train_loss_tracker, self.train_accuracy, self.train_perplexity, self.train_accuracy_sentence,
            self.test_loss_tracker, self.test_accuracy, self.test_perplexity, self.test_accuracy_sentence]

  def call(self, inputs, training=False):
    """
    It performs a forward pass. Calls the model on new inputs and returns the outputs as tensors.
    """
    context, question = inputs

    context = self.masking(context)
    question = self.masking(question)

    # We collect the question predicted by the decoder, the first character is the starting token
    y_pred = tf.fill([self.batch_size, 1], question[0][0])

    # Keep a loss tracking value
    if training:
      self.train_loss_tracker.reset_state()
    else:
      self.test_loss_tracker.reset_state()

    # Set the loss to 0 for each batch
    loss = tf.constant(0.0)

    # Encode the input, that is the context
    encoder_outputs, encoder_state = self.encoder(context, training=training)

    # The decoder should be initialized with the encoder last state
    decoder_state = encoder_state

    t = 0
    # We have to run the decoder for all the length of the question
    while t < (self.max_length_question - 1):
      # We have to pass two tokens:
      #   1. the token at time step t, namely the token in which we need to start run the decoder
      #   2. the token at time step t+1, that is the next token in the sequence that needs to be compared with
      # Note that: at the start of the question the new_token will be start-of-seq tag
      new_token = tf.gather(question, t, axis=1)
      target_token = tf.gather(question, t+1, axis=1)

      # Here we call the decoder in order to produce the token at time step t+1, it returns,
      #   1. the partial loss for the predicted token,
      #   2. the new decoder state,
      #   3. the predicted token at time step t+1
      step_loss, decoder_state, token_pred = self.step_decoder(
          (new_token, target_token),
          encoder_outputs,
          decoder_state,
          training=training)

      # Concatenate the predicted token to the already obtained question
      y_pred = tf.concat([y_pred, token_pred], axis=1)

      # Each partial loss contributes to the total loss
      loss = loss + step_loss
      t = t + 1

    # Since we have computed a cumulated partial scores of the losses produced
    # at each decoder step we then average for the number of elements that are
    # not padding
    loss = loss / tf.math.reduce_sum(tf.cast(question != 0, dtype=loss.dtype))

    return y_pred, loss

  def step_decoder(self, tokens, encoder_outputs, decoder_state, training):
    """
    Run a single iteration of the decoder and computers the incremental loss between the
    produced token and the token in the target input.
    """
    new_token = tokens[0]

    # Run the decoder one time, this will return the logits for the token at timestep t+1 for the entire batch
    decoder_logits, _, decoder_state = self.decoder([new_token, encoder_outputs], state=decoder_state, training=training)

    y_true = tf.expand_dims(tokens[1], axis=1)
    y_pred = decoder_logits

    # Compare the produced logits with the true token
    step_loss = self.loss(y_true=y_true, y_pred=y_pred)

    # Greedily extract the word that has the maximum value in the produced logits
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.int64)

    if training:
      self.train_accuracy.update_state(y_true=y_true, y_pred=y_pred, sample_weight=tf.cast((y_true != 0), tf.int32))
    else:
      self.test_accuracy.update_state(y_true=y_true, y_pred=y_pred, sample_weight=tf.cast((y_true != 0), tf.int32))

    return step_loss, decoder_state, y_pred

  @tf.function
  def train_step(self, inputs):
    """
    Optimization step for a batch.
    """
    context, question = inputs
    with tf.GradientTape() as tape:
      # Computes the forward pass: it returns the predicted question and the loss value averaged over the batch
      y_pred, loss = self(inputs, training=True)

    self.train_loss_tracker.update_state(loss)

    # Compute gradients
    tr_variables = self.trainable_variables
    grads = tape.gradient(loss, tr_variables)

    # Apply some clipping (by norm) as done in the paper and update the weights
    ads = [tf.clip_by_norm(g, 5.0) for g in grads]
    self.optimizer.apply_gradients(zip(grads, tr_variables))

    # Compute metrics
    self.train_perplexity.update_state(loss)
    self.train_accuracy_sentence.update_state(y_true=question, y_pred=y_pred)

    return {m.name: m.result() for m in self.metrics[:4]}

  @tf.function
  def test_step(self, inputs):
    """
    The logic for one evaluation step. This function should contain the mathematical logic 
    for one step of evaluation. This typically includes the forward pass, 
    loss calculation, and metrics updates.

    Called when at each epoch's end to validate on the validation data given.
    """
    context, question = inputs

    context = self.masking(context)
    question = self.masking(question)

    # Computes a forward pass with training set to False
    y_pred, loss = self(inputs, training=False)
    self.test_loss_tracker.update_state(loss)

    # Compute metrics
    self.test_perplexity.update_state(loss)
    self.test_accuracy_sentence.update_state(y_true=question, y_pred=y_pred)

    return {m.name: m.result() for m in self.metrics[4:]}

  # # # Reference :- https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
  def build_graph(self):
    return tf.keras.Model(inputs=[self.context_input, self.question_input], outputs=self.call([self.context_input, self.question_input]))

  def plot_model(self):
    return tf.keras.utils.plot_model(
        self.build_graph(),
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True
    )
