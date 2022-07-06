BATCH_SIZE = 256
UNITS = 512
ENABLE_KERAS_TUNER = False

# Dataset configuration: in this case we are working with a reduced version
# of the SQuAD dataset.
dataset_config = {
    'num_examples': 90000,
    'train_size': 0.65,
    'test_size': 0.40,
    'num_words_context': 45000,
    'num_words_question': 28000,
    'buffer_size': 32000,
    'batch_size': BATCH_SIZE,
    'random_seed': 13,
}

# Model configuration: this is the configuration of the model that will be
# trained.
model_config = {
    'batch_size': BATCH_SIZE,
    'enc_units': UNITS,
    'dec_units': UNITS,
    'max_length_context': None,
    'max_length_question': None,
    'dropout_rate': .3,
    'regularizer': 1e-3,
    'embedding_dimension': 300,
    'lr_schedule': [
        # (epoch to start, learning rate) tuples
        (15, 1e-1),
        (23, 5e-2),
        ],
}

# Relative path to the directory containing the dataset, the checkpoints and the processed dataset
path = {
    'training_json_path': "./data/squad.json",
    # 'training_json_path': "./data/dev-v2.0.json",
    'save_pkl_path': "./data/squadv1.1.pkl",
    # 'checkpoint_dir': "./training_checkpoints",
    'log_dir': "./models/logs",
}

# Evaluation configuration: this is the configuration of the model that will be
# used to evaluate the performance of the model.
evaluation_config = {
    'temperature': 0.7,
}

keras_tuner_config = {
    'epochs_tuning': 30,
}

# training_info = {
#     'verbose': 1,
#     'epochs': 30,
#     'batch_size': dataset_config['batch_size'],
#     'callbacks': [
#                   BatchLogs('batch_loss'),
#                   BatchLogs('perplexity'),
#                   BatchLogs('accuracy'),
#                   # lr_scheduler,
#                   # tensorboard_callback,
#                   # epoch_counter,
#                   tf.keras.callbacks.EarlyStopping(monitor='val_perplexity', patience=3, mode='max', restore_best_weights=True)
#                   ],
# }

# compile_info = {
#     'loss': MaskedLoss(),
#     'optimizer': tf.keras.optimizers.Adam(learning_rate=8e-6)}