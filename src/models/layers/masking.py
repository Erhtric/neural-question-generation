import tensorflow as tf
from keras.layers import Layer

class CustomMasking(Layer):
    def __init__(self, mask_value=0, **kwargs):
        super(CustomMasking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, inputs, mask=None):
        return tf.math.reduce_any(tf.math.not_equal(inputs, self.mask_value),
                                  axis=list(range(2, len(inputs.shape))))

    def call(self, inputs):
        axes = list(range(2, len(inputs.shape)))
        boolean_mask = tf.math.reduce_any(tf.math.not_equal(inputs, self.mask_value),
                             axis=axes, keepdims=True)
        outputs = inputs * tf.cast(boolean_mask, inputs.dtype)
        # Compute the mask and outputs simultaneously.
        outputs._keras_mask = tf.squeeze(boolean_mask, axis=axes)  # pylint: disable=protected-access
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(tf.keras.layers.Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))