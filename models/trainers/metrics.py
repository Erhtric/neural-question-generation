import tensorflow as tf
from keras.metrics import Metric

class Perplexity(Metric):
  def __init__(self, name='perplexity', **kwargs):
    super(Perplexity, self).__init__(name=name, **kwargs)
    self.scores = self.add_weight(name='perplexity_scores', initializer='zeros')

  def update_state(self, loss):
    """
    Reference :- https://www.surgehq.ai/blog/how-good-is-your-chatbot-an-introduction-to-perplexity-in-nlp
    """
    self.scores.assign(tf.exp(loss))

  def result(self): return self.scores
  def reset_state(self): self.scores.assign(0)

# Also the accuracy should mask the padding
class MaskedAccuracy(Metric):
  def __init__(self, name='masked_accuracy',**kwargs):
    super(MaskedAccuracy, self).__init__(name=name, **kwargs)
    self.scores = self.add_weight(name='accuracy_scores', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    # We mask since we are not interested in the final accuracy score with the padding
    mask = tf.cast(tf.math.greater(y_true, 0), dtype=tf.float32)

    correct = tf.cast(tf.math.equal(y_true, y_pred), dtype=tf.float32)
    correct = tf.math.reduce_sum(mask * correct)
    total_legit = tf.math.reduce_sum(mask)

    self.scores.assign(correct / total_legit)

  def result(self): return self.scores
  def reset_state(self): self.scores.assign(0)