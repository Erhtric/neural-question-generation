import random
import numpy as np
import tensorflow as tf
import tqdm as tqdm

from .eval_metrics import METEOR, ROUGE, AnswerabilityMetric

class Evaluator(tf.Module):
    def __init__(self, model, evaluation_config, tokenizer_context, tokenizer_question):
        """It creates the model for the inference and evaluation.

        Args:
            model: the trained model
            evaluation_config: the configuration for the evaluation
            tokenizer_context: the tokenizer fitted on the context
            tokenizer_question: the tokenizer fitted on the question
        """
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.tokenizer_context = tokenizer_context
        # The tokenizer will be used for the conversion from question tokens to
        # strings. It requires the tokenizer fit on the questions.
        self.tokenizer_question = tokenizer_question

        self.result_tokens = None               # Tokens predicted
        self.result_text = None                 # Text predicted
        self.token_mask = self.create_mask()  # Mask for the tokens

        self.start_idx = tokenizer_question.word_index['<sos>']
        self.end_idx = tokenizer_question.word_index['<eos>']
        self.unk_idx = tokenizer_question.word_index['<unk>']

        # Config
        self.temperature = evaluation_config['temperature']

        # Metrics
        # These ones are computed and refreshed for each batch
        self.meteor_metric_batch = METEOR()
        self.rouge_metric_batch = ROUGE()
        self.answerability_metric = AnswerabilityMetric()

        # These metrics are the mean value among all the batches
        self.meteor_metric = tf.keras.metrics.Mean(name='meteor_mean')
        self.rouge1_precision_metric = tf.keras.metrics.Mean(name='rouge1_precision_mean')
        self.rouge1_recall_metric = tf.keras.metrics.Mean(name='rouge1_recall_mean')
        self.rouge1_fmeasure_metric = tf.keras.metrics.Mean(name='rouge1_fmeasure_mean')
        self.rouge2_precision_metric = tf.keras.metrics.Mean(name='rouge2_precision_mean')
        self.rouge2_recall_metric = tf.keras.metrics.Mean(name='rouge2_recall_mean')
        self.rouge2_fmeasure_metric = tf.keras.metrics.Mean(name='rouge2_fmeasure_mean')
        self.rougeL_precision_metric = tf.keras.metrics.Mean(name='rougeL_precision_mean')
        self.rougeL_recall_metric = tf.keras.metrics.Mean(name='rougeL_recall_mean')
        self.rougeL_fmeasure_metric = tf.keras.metrics.Mean(name='rougeL_fmeasure_mean')

        # Store evaluation results
        self.results = dict()
        self.results_answerability = dict()

    def token_to_string(self, result_tokens: tf.Tensor):
        """This method converts the tokens to strings.

        Args:
            result_tokens (tf.Tensor): the tokens to convert

        Returns:
            list: the tokenized list of strings converted in text by using the tokenizer provided in the class
        """
        list_tokens = result_tokens.numpy().tolist()
        list_text = self.tokenizer_question.sequences_to_texts(list_tokens)
        result_text = [s.split() for s in list_text]

        self.result_tokens = result_tokens
        self.result_text = result_text
        return result_text

    def create_mask(self):
        """This method creates a mask for the padding, the unknwon words and the start/ending tokens.
        Note that the mask needs to be applied on a logits output of the model.

        Returns:
            a mask for the unwanted tokens
        """
        masked_words = ['<pad>', '<sos>', '<eos>']
        token_mask_ids = [self.tokenizer_question.word_index[mask] for mask in masked_words]
        token_mask = np.zeros(shape=(len(self.tokenizer_question.word_index),), dtype=bool)
        token_mask[np.array(token_mask_ids)] = True
        return token_mask

    def remove_tags(self, result_tokens: tf.Tensor):
        """This method removes the tags from the tokens."""
        list_tokens = result_tokens.numpy().tolist()
        tag_words = ['<pad>', '<sos>', '<eos>']
        token_tag_ids = [self.tokenizer_question.word_index[tag] for tag in tag_words]

        #For every element of the batch we extract the token list and we remove the unwanted tokens
        list_tokens = [[token for token in token_list if token not in token_tag_ids] for token_list in list_tokens]

        texts = self.tokenizer_question.sequences_to_texts(list_tokens)
        texts = [s.split() for s in texts]

        return texts

    @tf.autograph.experimental.do_not_convert
    def evaluate_metric_answerability(self, inputs, max_length, metric_name=None, metric_value=None):
        """Evaluate the model on the given inputs by using the answerability metric. The predicted questions will have the same length as the
        max length of the question specified in input. It is mandatory to have already computed the other metrics
        on the same inputs

        Args:
            inputs (tf.data.Dataset): the dataset to evaluate
            max_length (int): the max length of the questions predicted
            metric_name (str, optional): the name of the metric on which the answerability will give weight. Defaults to None.
            metric_value (np.ndarray, optional): the value of the metric chosen for the weigthed average. Defaults to None.

        Returns:
            a dictionary with the results of the evaluation with the answerability metric and the metric chosen
        """
        self.answerability_metric.reset_state()

        seq = []
        for (context, question_true) in tqdm(inputs):
            question_true = self.remove_tags(question_true)
            prediction = self.predict_step(inputs=context,
                                            max_length=max_length,
                                            return_attention=False,
                                            pretty_predict=False)

            question_pred = prediction['text']

            # Compute answerability weighted score on the given metric chosen
            self.answerability_metric.update_state(y_pred=question_pred, y_true=question_true, metric_name=metric_name, metric_value=tf.constant(metric_value))
            seq.append(self.answerability_metric.result())

        self.results_answerability[f'{metric_name}'] = tf.reduce_mean(seq).numpy()
        return self.results_answerability

    @tf.autograph.experimental.do_not_convert
    def evaluate(self, inputs, max_length):
        """Evaluate the model on the given inputs. The predicted questions will have the same length as the
        max length of the question specified in input.

        Args:
            inputs (tf.data.Dataset): the dataset to evaluate
            max_length (int): the max length of the questions predicted

        Returns:
            a dictionary containing the results of the evaluation
        """
        self.meteor_metric.reset_state()
        self.rouge1_precision_metric.reset_state()
        self.rouge1_recall_metric.reset_state()
        self.rouge1_fmeasure_metric.reset_state()
        self.rouge2_precision_metric.reset_state()
        self.rouge2_recall_metric.reset_state()
        self.rouge2_fmeasure_metric.reset_state()
        self.rougeL_precision_metric.reset_state()
        self.rougeL_recall_metric.reset_state()
        self.rougeL_fmeasure_metric.reset_state()

        for (context, question_true) in tqdm(inputs):
            self.meteor_metric_batch.reset_state()
            self.rouge_metric_batch.reset_state()

            question_true = self.remove_tags(question_true)
            prediction = self.predict_step(inputs=context, max_length=max_length, return_attention=False, pretty_predict=False)
            question_pred = prediction['text']

            # Compute the metric for the current batch
            self.meteor_metric_batch.update_state(y_true=question_true, y_pred=question_pred)
            # Compute the mean over the batches
            self.meteor_metric.update_state(self.meteor_metric_batch.result()['meteor'])

            # Compute for the current batch
            self.rouge_metric_batch.update_state(y_true=question_true, y_pred=question_pred)
            # Compute the mean over the batches
            self.rouge1_precision_metric.update_state(self.rouge_metric_batch.result()['precision_1'])
            self.rouge1_recall_metric.update_state(self.rouge_metric_batch.result()['recall_1'])
            self.rouge1_fmeasure_metric.update_state(self.rouge_metric_batch.result()['fmeasure_1'])
            self.rouge2_precision_metric.update_state(self.rouge_metric_batch.result()['precision_2'])
            self.rouge2_recall_metric.update_state(self.rouge_metric_batch.result()['recall_2'])
            self.rouge2_fmeasure_metric.update_state(self.rouge_metric_batch.result()['fmeasure_2'])
            self.rougeL_precision_metric.update_state(self.rouge_metric_batch.result()['precisionL'])
            self.rougeL_recall_metric.update_state(self.rouge_metric_batch.result()['recallL'])
            self.rougeL_fmeasure_metric.update_state(self.rouge_metric_batch.result()['fmeasureL'])

        self.results = {'METEOR': self.meteor_metric.result().numpy(),
                'ROUGE_1_PRECISION': self.rouge1_precision_metric.result().numpy(),
                'ROUGE_1_RECALL': self.rouge1_recall_metric.result().numpy(),
                'ROUGE_1_FMEASURE': self.rouge1_fmeasure_metric.result().numpy(),
                'ROUGE_2_PRECISION': self.rouge2_precision_metric.result().numpy(),
                'ROUGE_2_RECALL': self.rouge2_recall_metric.result().numpy(),
                'ROUGE_2_FMEASURE': self.rouge2_fmeasure_metric.result().numpy(),
                'ROUGE_L_PRECISION': self.rougeL_precision_metric.result().numpy(),
                'ROUGE_L_RECALL': self.rougeL_recall_metric.result().numpy(),
                'ROUGE_L_FMEASURE': self.rougeL_fmeasure_metric.result().numpy()}

        return self.results

    def predict(self, inputs, max_length, return_attention=False, pretty_predict=False):
        """Generates output predictions for the input samples. Computation is done in batches.

        Args:
            inputs (tf.data.Dataset): the dataset to evaluate
            max_length (int): the max length of the questions predicted
            return_attention (bool, optional): flag to return the attention weights associated. Defaults to False.
            pretty_predict (bool, optional): forces to return a list of sentences in a readable format. Defaults to False.

        Returns:
            the predicted questions (and attention weights if requested)
        """
        results = []
        for (context, _) in tqdm(inputs):
            results.append(self.predict_step(inputs=context,
                                            max_length=max_length,
                                            return_attention=return_attention,
                                            pretty_predict=pretty_predict))

        return results

    def predict_step(self, inputs, max_length, return_attention, pretty_predict):
        """The logic for one inference step."""
        # Similarly for what it has been done in the train step
        encoder_output, encoder_state = self.encoder(inputs)
        decoder_state = encoder_state

        # Generate the first token of each sentence, that is the <sos> token
        new_token = tf.fill([self.model.batch_size, 1], self.start_idx)

        result_tokens = []
        attention = []

        # Mask used to avoid performing the computation on the <eos> token
        done = tf.zeros(shape=(self.model.batch_size, 1), dtype=tf.bool)
        # Mask used to recognize the <unk> token
        unk = tf.zeros(shape=(self.model.batch_size, 1), dtype=tf.bool)

        for _ in range(max_length):
            # Decode the token at the next timestep
            decoder_logits, attention_weights, decoder_state = self.decoder([new_token, encoder_output], state=decoder_state)

            attention.append(attention_weights)

            # Sample the new token accordingly to the distribution produced by the decoder
            new_token = self.temperature_sampling(decoder_logits)

            # if a sequence has reached <eos> set it as done
            done = done | (new_token == self.end_idx)
            # Once a sequence is done it only produces 0-padding.
            new_token = tf.where(done, tf.constant(0, dtype=tf.int64), new_token)

            # if a token produce has value <unk> set it as unk
            unk = unk | (new_token == self.unk_idx)
            # Once a token has been tagged as unk we have to chenage its value with
            # the value in the context that has the highest attention
            highest_attention = tf.math.argmax(attention_weights, axis=-1)
            context_attention = tf.gather(inputs, highest_attention, axis=-1, batch_dims=1)
            new_token = tf.where(unk, context_attention, new_token)

            result_tokens.append(new_token)

            if tf.reduce_all(done):
                break

        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.token_to_string(result_tokens)
        if pretty_predict: result_text = self.prettify(result_text)

        attention_stack = tf.concat(attention, axis=-1)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}

    def prettify(self, result_text):
        """Converts the predicted text into a readable format."""
        results = []
        for sen in result_text:
            results.append(" ".join(list(sen)))
        return results

    def temperature_sampling(self, logits):
        """Samples a token from the distribution produced by the decoder. The temperature parameter is used to
        control the degree of randomness.

        For the temperature choice see here:
        Reference :- https://nlp.stanford.edu/blog/maximum-likelihood-decoding-with-rnns-the-good-the-bad-and-the-ugly/

        Usually:
            - temperature = 1: no randomness-> the freezing function now scales the logits by the temperature
            - temperature = 0: no randomness-> argmax, this means more correctess from a grammatical viewpoint
        """
        # First of all we use broadcast the generated mask to the expected logits' shape
        # token_mask shape: (batch_size, timestep, vocab_size)
        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]

        # The logits for all the tokens that have to not be used are set to -1.0
        logits = tf.where(token_mask, -1.0, logits)

        # Freezing function
        # Higher temperature -> greater variety
        # Lower temperature -> grammatically correct
        if self.temperature == 0.0:
        # the freezing function is the argmax, behaving like a greedy search
            new_token = tf.argmax(logits, axis=-1)
        else:
        # the freezing function now scales the logits.
        # for temperature == 1.0 is the identity function
            logits = tf.squeeze(logits, axis=1)
            new_token = tf.random.categorical(logits / self.temperature, num_samples=1)
        return new_token