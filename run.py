from pip import main
from data_loader.data_generator import SQuAD
from configs.config import *
from utils import dirs

if __name__ == '__main__':
    # main(['install', '-r', 'requirements.txt']) # Install all requirements

    # Create directories
    dirs.create_dirs(['./data', './models/training_checkpoints', './models/logs'])

    # Create data generator for the dataset
    print(f'Current dataset configuration: {dataset_config}')
    data_generator = SQuAD()
    print('Loading dataset...please wait')
    print(data_generator)
    print('Dataset loaded!')

    dataset, word_to_idx_context, word_to_idx_question = data_generator(**dataset_config, training_json_path=path['training_json_path'], save_pkl_path=path['save_pkl_path'], tokenized=True)

    max_length_context = dataset.train.element_spec[0].shape[1]
    max_length_question = dataset.train.element_spec[1].shape[1]

    print(f'Sentences max lenght: {max_length_context}')
    print(f'Questions max lenght: {max_length_question}')