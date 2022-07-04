import tensorflow as tf
from keras.losses import Loss, SparseCategoricalCrossentropy 

class MaskedLoss(Loss):
    """
    This is a subclass of the keras.losses.Loss class. It implements
    a sparse categorical crossentropy loss with a mask, that is each padding
    elements weigth is set to 0. It is summed over the batch and then returned.
    """
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred) 

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        # Sum the loss over the batch since reduction is set to NONE
        loss = tf.reduce_sum(loss)
        return loss