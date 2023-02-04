import tensorflow as tf
import spacy
from keras.metrics import Metric
from datasets import load_metric

class METEOR(Metric):
    def __init__(self, name=f"meteor_metric", **kwargs):
        """
        Initialize the metric object for computing METEOR.
        """
        super(METEOR, self).__init__(name=name, **kwargs)
        # Reference :- https://github.com/huggingface/datasets/tree/master/metrics/meteor
        self.meteor = load_metric("meteor")
        self.scores = self.add_weight(name=f"meteor_scores", initializer="zeros", dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        results = self.meteor.compute(predictions=y_pred, references=y_true)
        self.scores.assign(tf.constant(results['meteor'], dtype=tf.float64))

    def result(self):
        return {'meteor': self.scores}

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.scores.assign(0.0)

class ROUGE(Metric):
    def __init__(self, name=f"rouge_metric", **kwargs):
        """
        Initialize the metric object for computing ROUGE.
        """
        super(ROUGE, self).__init__(name=name, **kwargs)
        # Reference :- https://github.com/huggingface/datasets/tree/master/metrics/rouge
        self.rouge = load_metric("rouge")

        self.precision_1 = self.add_weight(name=f"rouge1_precision_scores", initializer="zeros")
        self.recall_1 = self.add_weight(name=f"rouge1_recall_scores", initializer="zeros")
        self.fmeasure_1 = self.add_weight(name=f"rouge1_fmeasure_scores", initializer="zeros")

        self.precision_2 = self.add_weight(name=f"rouge2_precision_scores", initializer="zeros")
        self.recall_2 = self.add_weight(name=f"rouge2_recall_scores", initializer="zeros")
        self.fmeasure_2 = self.add_weight(name=f"rouge2_fmeasure_scores", initializer="zeros")

        self.precisionL = self.add_weight(name=f"rougeL_precision_scores", initializer="zeros")
        self.recallL = self.add_weight(name=f"rougeL_recall_scores", initializer="zeros")
        self.fmeasureL = self.add_weight(name=f"rougeL_fmeasure_scores", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        results = self.rouge.compute(predictions=y_pred, references=y_true, rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=True)

        self.precision_1.assign(tf.constant(results['rouge1'].mid.precision, dtype=tf.float32))
        self.recall_1.assign(tf.constant(results['rouge1'].mid.recall, dtype=tf.float32))
        self.fmeasure_1.assign(tf.constant(results['rouge1'].mid.fmeasure, dtype=tf.float32))

        self.precision_2.assign(tf.constant(results['rouge2'].mid.precision, dtype=tf.float32))
        self.recall_2.assign(tf.constant(results['rouge2'].mid.recall, dtype=tf.float32))
        self.fmeasure_2.assign(tf.constant(results['rouge2'].mid.fmeasure, dtype=tf.float32))

        self.precisionL.assign(tf.constant(results['rougeL'].mid.precision, dtype=tf.float32))
        self.recallL.assign(tf.constant(results['rougeL'].mid.recall, dtype=tf.float32))
        self.fmeasureL.assign(tf.constant(results['rougeL'].mid.fmeasure, dtype=tf.float32))

    def result(self):
        return {'precision_1': self.precision_1,
                'recall_1': self.recall_1,
                'fmeasure_1': self.fmeasure_1,
                'precision_2': self.precision_2,
                'recall_2': self.recall_2,
                'fmeasure_2': self.fmeasure_2,
                'precisionL': self.precisionL,
                'recallL': self.recallL,
                'fmeasureL': self.fmeasureL}

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.precision_1.assign(0.0)
        self.recall_1.assign(0.0)
        self.fmeasure_1.assign(0.0)

        self.precision_2.assign(0.0)
        self.recall_2.assign(0.0)
        self.fmeasure_2.assign(0.0)

        self.precisionL.assign(0.0)
        self.recallL.assign(0.0)
        self.fmeasureL.assign(0.0)

class AnswerabilityMetric(Metric):
    def __init__(self, **kwargs):
        """
        Initialize the metric object for computing Answerability.
        """
        super(AnswerabilityMetric, self).__init__(**kwargs)
        self.nlp = spacy.load('en_core_web_sm')
        self.scores = self.add_weight(name = f"answerability", initializer= "zeros")

    def update_state(self, y_pred, y_true, metric_name=None, metric_value=None, alpha=0.7, sample_weight=None):
        """It updates the metric state by computing the answerability score for a given metric
        """
        if (metric_value == None or metric_name == None):
            value_answerability = self.batch_answerability_value(y_pred, y_true)
        else:
            value_answerability = self.q_metric(y_pred, y_true, metric_value, alpha)
            self.scores = self.add_weight(name = f"answerability_{metric_name}", initializer= "zeros")
        self.scores.assign(value_answerability)

    def compute_ner(self,sentence):
        """Find the NER into the sentence"""
        doc = self.nlp(sentence)
        count = 0
        entities = []
        for ent in doc.ents:
            entities.append(str(ent))
        return entities

    # Find the question words into the sentence
    def question_words(self,sentence):
        question_words = []
        question_terms = ['what','where','when','whom','how','which','whose','why','?']
        for q in question_terms:
            if q in sentence: question_words.append(q)
        return question_words

    # Extract the keyword based on the POS
    def get_keywords(self,text):
        hotwords = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN']
        doc = self.nlp(text) # 2
        for token in doc:
            if(token.text in self.nlp.Defaults.stop_words):
                continue
            if(token.pos_ in pos_tag):
                hotwords.append(token.text)
        return hotwords

    #Extraction of context words using keywords extraction by spaCy
    def extract_context_word(self,sentence):
        numOfKeywords = len(sentence)
        context_words = []
        keywords= self.get_keywords(sentence)
        for kw in keywords:
            context_words.append(kw)
        return context_words

    #Extraction of function words using stopwords extraction by spaCy
    def function_words(self,sentence):
        function_words = []
        stopwords = self.nlp.Defaults.stop_words
        for q in list(stopwords):
            if(q in sentence):
                function_words.append(q)
        return function_words

    def answerability_value (self, pred, truth, weights = [0.25, 0.25, 0.25]):
        if(type(pred)== list):
            pred = " ".join(pred)
        if(type(truth)== list):
            truth = " ".join(truth)
        #Set the weights
        w_relevant, w_ner, w_question = weights

        #Find the relevant words of truth and prediction
        truth_relevant = self.extract_context_word(truth)
        pred_relevant = self.extract_context_word(pred)
        #Calculate the count of matching words
        C_relevant = len([w for w in truth_relevant if w in pred_relevant])
        #If no relevant words the weight is reduced to 0
        if (len(truth_relevant)==0 and len(pred_relevant)==0):
            w_relevant=0

        #Find NER of truth and prediction
        truth_ner = self.compute_ner(truth)
        pred_ner = self.compute_ner(pred)
        #Calculate the count of matching words
        C_ner = len([w for w in truth_ner if w in pred_ner])
        #If no NER words the weight is reduced to 0
        if (len(truth_ner)==0 and len(pred_ner)==0):
            w_ner=0

        #Find question words in truth and pred
        truth_question = self.question_words(truth)
        pred_question = self.question_words(pred)
        #Calculate the count of matching words
        C_question = len([w for w in truth_question if w in pred_question])
        #If no question words the weight is reduced to 0
        if (len(truth_question)==0 and len(pred_question)==0):
            w_question=0

        #Find functional words as the rest of the words excluding the already counted
        truth_functional = self.function_words(truth)
        pred_functional = self.function_words(pred)
        w_functional = 1 - (w_relevant+ w_ner+ w_question)

        C_functional = len([w for w in truth_functional if w in pred_functional])

        #Calculate precision and recall
        P = w_relevant*C_relevant/(len(pred_relevant)+1e-10) + w_ner*C_ner/(len(pred_ner)+1e-10) + w_question*C_question/(len(pred_question)+1e-10) + w_functional*C_functional/(len(pred_functional)+1e-10)
        R = w_relevant*C_relevant/(len(truth_relevant)+1e-10) + w_ner*C_ner/(len(truth_ner)+1e-10) + w_question*C_question/(len(truth_question)+1e-10) + w_functional*C_functional/(len(truth_functional)+1e-10)

        #Compute final answerability value
        answerability = 2 * P * R / (P + R + 1e-10)

        return answerability

    #Find the mean answerability value of a batch
    def batch_answerability_value(self, batch_pred, batch_truth):
        answ_values = []
        for i in range(len(batch_pred)):
            answ_values.append(self.answerability_value(list(batch_pred[i]),batch_truth[i]))
        batch_answ = tf.reduce_mean(answ_values)

        return batch_answ

    def q_metric(self,batch_pred,batch_truth,metric_value,alpha):
        answ = self.batch_answerability_value(batch_pred,batch_truth).numpy()
        m = metric_value.numpy()
        return tf.constant(alpha*(answ) + (1-alpha)*(m),dtype=tf.float32)

    def result(self): return self.scores
    def reset_state(self): self.scores.assign(0)