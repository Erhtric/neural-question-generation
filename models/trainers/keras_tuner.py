import tensorflow as tf
from keras.optimizers import Adam
import keras_tuner as kt
from loss import MaskedLoss
from configs.config import keras_tuner_config, ENABLE_KERAS_TUNER
from utils.utils import build_trainer

def run_keras_tuner(dataset, embedding_matrix_context, embedding_matrix_question):
    if ENABLE_KERAS_TUNER:
        tuner = kt.Hyperband(lambda hp: build_trainer(embedding_matrix_context=embedding_matrix_context,
                                                embedding_matrix_question=embedding_matrix_question,
                                                model_config={
                                                    'batch_size': hp.Choice('batch_size', [256]),
                                                    "dropout_rate": hp.Choice('dropout_rate', [0.3]),
                                                    "regularizer": hp.Choice("regularizer", [1e-3, 1e-4]),
                                                    "enc_units":  hp.Choice('units', [600]),
                                                    "dec_units": hp.Choice('units', [600]),
                                                    'max_length_context': dataset.train.element_spec[0].shape[1],
                                                    'max_length_question': dataset.train.element_spec[1].shape[1],
                                                },
                                                compile_info={
                                                    'loss': MaskedLoss(),
                                                    'optimizer': Adam(learning_rate=hp.Choice('learning_rate', [1e-5, 8e-6, 3e-5])),
                                                }),
                        objective=kt.Objective("val_perplexity", direction="min"),
                        max_epochs=keras_tuner_config['epochs_tuning'],
                        overwrite=True,
                        directory="tuner",
                        project_name="tuner_qg"
                        )

        tuner.search_space_summary()

        #stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode="max", restore_best_weights=True)

        tuner.search(dataset.train,
                        validation_data = dataset.val,
                        epochs=keras_tuner_config['epochs_tuning'],
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_perplexity',
                                                        patience=5,
                                                        mode='max',
                                                        restore_best_weights=True)] )
        best_hps = tuner.get_best_hyperparameters()[0]

        print(f"The hyperparameter search is complete.\n"
                f"The optimal regualizer rate is: {best_hps.get('batch_size')}.\n"
                f"The optimal regualizer rate is: {best_hps.get('regularizer')}.\n"
                f"The optimal rate for Dropout layer is: {best_hps.get('dropout_rate')}.\n"
                f"The optimal number of units is: {best_hps.get('enc_units')}.\n"
                f"The optimal learning rate for the optimizer is: {best_hps.get('learning_rate')}.")