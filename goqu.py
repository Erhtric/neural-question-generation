import sys


if __name__ == '__main__':
    """
    Generation Of QUestions. GoQU is a tool for generating questions from a given context.
    """
    # main(['install', '-r', 'requirements.txt']) # Install all requirements

    args = str(sys.argv)
    if len(sys.argv) == 1:
        print("Please provide the arguments: call -h or --help for help")
        exit(1)
    else:
        if "-t" in args or "--train" in args:
            print("Training the model")
            # Train the model
        elif "-p" in args or "--predict" in args:
            print("Predicting the questions")
            # Predict the questions
            context = input("Enter the sentence!\n")
            print("Context: ", context)
        elif "-e" in args or "--evaluate" in args:
            print("Evaluating the model")
            # Evaluate the model
        elif "-h" in args or "--help" in args:
            print("Help")
            # Help
            print("Call the program with the following arguments: -t or --train to train the model, -p or --predict to predict the questions, -e or --evaluate to evaluate the model")
        elif "-v" in args:
            # Libraries importing
            #from pip import main
            from data_loader.data_generator import SQuAD
            from utils.utils import prepare_embeddings
            from configs.config import dataset_config, model_config, path
            from utils import dirs
            import pprint
            import os

            # Create directories
            dirs.create_dirs(['./data', './models/training_checkpoints', './models/logs'])
            # Clean the logs directory from any previous runs if any
            os.system('rm -rf ./models/logs/')

            # Load configurations
            dataset_config = dataset_config
            path = path
            model_config = model_config

            print('Current dataset configuration:\n')
            pprint.pprint(dataset_config)
            print()
            print('Current paths:\n')
            pprint.pprint(path)
            print()

            data_generator = SQuAD()
            print('Loading dataset...please wait')
            dataset, word_to_idx_context, word_to_idx_question = data_generator(dataset_config, path, tokenized=True)
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

