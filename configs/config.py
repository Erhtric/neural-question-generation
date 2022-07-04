BATCH_SIZE = 128
UNITS = 300
TRAIN = False
ENABLE_KERAS_TUNER = False

# Dataset configuration: in this case we are working with a reduced version
# of the SQuAD dataset.
dataset_config = {
    'num_examples': 90000,
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
    'dropout_rate': None,
    'regularizer': None,
    'embedding_dimension': 300,
    'lr_schedule': [
        # (epoch to start, learning rate) tuples
        (15, 1e-1),
        (23, 5e-2),
        # (12, 5e-5),
        # (14, 1e-5),
        ],
}

# Relative path to the directory containing the dataset, the checkpoints and the processed dataset
path = {
    'training_json_path': "./data/training_set.json",
    'save_pkl_path': "./data/squadv2.pkl",
    # 'checkpoint_dir': "./training_checkpoints",
    'log_dir': "./models/logs",
}

# Evaluation configuration: this is the configuration of the model that will be
# used to evaluate the performance of the model.
evaluation_config = {
    'batch_size': 1,
    'temperature': 0.7,
}

