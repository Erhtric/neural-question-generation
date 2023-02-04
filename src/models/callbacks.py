import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from configs.config import model_config, path

class GetEpochNumber(Callback):
  def __init__(self, **kwargs) -> None:
    super(GetEpochNumber, self).__init__(**kwargs)

  def on_epoch_end(self, epoch, logs=None):
    self.model.epoch_number.assign_add(delta=1)

class BatchLogs(tf.keras.callbacks.Callback):
  def __init__(self, key) -> None:
    self.key = key
    self.logs = []

  def on_train_batch_ends(self, n, logs):
    self.logs.append(logs[self.key])

class CustomLearningRateScheduler(Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index (integer, indexed from 0)
        and current learning rate as inputs and returns a new learning rate
        as output (float).
  """
  def __init__(self):
    super(CustomLearningRateScheduler, self).__init__()
    self.schedule = self.lr_schedule

    self.LR_SCHEDULE = model_config['lr_schedule']

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, "lr"):
        raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.schedule(epoch, lr)

    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print(f"\nEpoch {epoch+1}: Learning rate is {scheduled_lr}." )

  def lr_schedule(self, epoch, lr):
    """
    Helper function to retrieve the scheduled learning rate based on epoch.
    """
    if epoch < self.LR_SCHEDULE[0][0] or epoch > self.LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(self.LR_SCHEDULE)):
        if epoch == self.LR_SCHEDULE[i][0]:
            return self.LR_SCHEDULE[i][1]
    return lr

# Initialize the callbacks

batch_loss = BatchLogs('batch_loss')
perplexity = BatchLogs('perplexity')
accuracy = BatchLogs('accuracy')
lr_scheduler = CustomLearningRateScheduler()
epoch_counter = GetEpochNumber()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path['log_dir'], histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_perplexity',
                                                  patience=8,
                                                  mode='max',
                                                  restore_best_weights=True)