from pip import main
from data_loader.data_generator import SQuAD
from utils.utils import prepare_embeddings
import configs.config as config
from utils import dirs
import pprint

if __name__ == '__main__':
    # main(['install', '-r', 'requirements.txt']) # Install all requirements

    # Create directories
    dirs.create_dirs(['./data', './models/training_checkpoints', './models/logs'])

    # Load configurations
    dataset_config = config.dataset_config
    path = config.path
    model_config = config.model_config

    print('Current dataset configuration:\n')
    pprint.pprint(dataset_config)
    print()
    print('Current paths:\n')
    pprint.pprint(path)
    print()

    data_generator = SQuAD()
    print('Loading dataset...please wait')
    dataset, word_to_idx_context, word_to_idx_question = data_generator(
        **dataset_config,
        training_json_path=path['training_json_path'],
        save_pkl_path=path['save_pkl_path'],
        tokenized=True)
    print('Dataset loaded!\n')

    print(40*'=')
    print(f'Context sentences max lenght: {data_generator.max_length_context}')
    print(f'Question sentences max lenght: {data_generator.max_length_question}')
    print(f'Vocab size --- [Context]: {len(word_to_idx_context[1])} [Question]: {len(word_to_idx_question[1])}')
    print(40*'=')

    model_config['max_length_context'] = data_generator.max_length_context
    model_config['max_length_question'] = data_generator.max_length_question

    # Build embedding matrix for the context and the question
    # We will fix the embedding dimension to be 300
    embedding_matrix_context, embedding_matrix_question = prepare_embeddings(word_to_idx_context=word_to_idx_context, word_to_idx_question=word_to_idx_question)

