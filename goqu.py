from logging import exception
import sys

from utils.utils import preprocess_input

def load_dataset(dataset_config, path):
    """Loads the dataset from the given path."""
    print('Current dataset configuration:\n')
    pprint.pprint(dataset_config)
    print()
    print('Current paths:\n')
    pprint.pprint(path)
    print()
    data_generator = SQuAD()
    dataset, word_to_idx_context, word_to_idx_question = data_generator(dataset_config, path, tokenized=True)

    model_config['max_length_context'] = data_generator.max_length_context
    model_config['max_length_question'] = data_generator.max_length_question

    # Save the word_indexes
    with open('./data/index_context', 'wb') as f:
        pickle.dump(word_to_idx_context, f)

    with open('./data/index_question', 'wb') as f:
        pickle.dump(word_to_idx_question, f)

    # # Save the tokenizers
    # with open('./data/tokenizer_context', 'wb') as f:
    #     pickle.dump(data_generator.tokenizer_context, f)

    # with open('./data/tokenizer_question', 'wb') as f:
    #     pickle.dump(data_generator.tokenizer_context, f)

    # Save the generator
    with open('./data/data_generator', 'wb') as f:
        pickle.dump(data_generator, f)

    print('Dataset loaded!\n')

    return dataset, word_to_idx_context, word_to_idx_question, data_generator


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

            # Libraries
            from data_loader.data_generator import SQuAD
            from utils.utils import prepare_embeddings, build_trainer, train_model
            from utils import dirs
            from configs.config import dataset_config, model_config, path, ENABLE_KERAS_TUNER
            from models.callbacks import BatchLogs
            from models.loss import MaskedLoss
            import pprint
            import os
            import tensorflow as tf
            import pickle

            # Create directories
            dirs.create_dirs(['./data', './models/training_checkpoints', './models/logs'])
            # Clean the logs directory from any previous runs if any
            os.system('rm -rf ./models/logs/')

            # Load configurations
            dataset_config = dataset_config
            path = path
            model_config = model_config

            dataset, word_to_idx_context, word_to_idx_question, data_generator = load_dataset(dataset_config, path)

            print(40*'=')
            print(f'Context sentences max lenght: {data_generator.max_length_context}')
            print(f'Question sentences max lenght: {data_generator.max_length_question}')
            print(f'Vocab size --- [Context]: {len(word_to_idx_context[1])} [Question]: {len(word_to_idx_question[1])}')
            print(40*'=')

            # Build embedding matrix for the context and the question
            # We will fix the embedding dimension to be 300
            embedding_matrix_context, embedding_matrix_question = prepare_embeddings(word_to_idx_context=word_to_idx_context, word_to_idx_question=word_to_idx_question)


            training_info = {
                'verbose': 1,
                'epochs': 1,
                'batch_size': dataset_config['batch_size'],
                'callbacks': [
                            BatchLogs('batch_loss'),
                            BatchLogs('perplexity'),
                            BatchLogs('accuracy'),
                            # lr_scheduler,
                            # tensorboard_callback,
                            # epoch_counter,
                            # early_stopping
                            ],
            }

            compile_info = {
                'loss': MaskedLoss(),
                'optimizer': tf.keras.optimizers.Adam(learning_rate=8e-6)}

            # Build the model for training
            qg_trainer = build_trainer(model_config,
                                    embedding_matrix_context,
                                    embedding_matrix_question,
                                    compile_info)

            raise Exception("Not Yet Implemented!")

            if not ENABLE_KERAS_TUNER:
                print('Libraries not updated. Please Fix!')
                history, qg_model = train_model(model=qg_trainer, dataset=dataset, training_info=training_info)
        elif "-p" in args or "--predict" in args:
            print("Predicting the questions")

            # Libraries
            import tensorflow as tf
            import pickle
            from configs.config import model_config
            from utils.utils import prepare_embeddings, build_trainer, preprocess_input
            from models.loss import MaskedLoss
            from models.eval.evaluator import Evaluator
            from configs.config import evaluation_config, dataset_config

            model_config = model_config

            # Load pre-fitted configurations on the SQuAD
            with open('./data/index_context', 'rb') as f:
                word_to_idx_context = pickle.load(f)

            with open('./data/index_question', 'rb') as f:
                word_to_idx_question = pickle.load(f)

            with open('./data/data_generator', 'rb') as f:
                data_generator = pickle.load(f)

            embedding_matrix_context, embedding_matrix_question = prepare_embeddings(word_to_idx_context=word_to_idx_context, word_to_idx_question=word_to_idx_question)

            model_config['max_length_context'] = data_generator.max_length_context
            model_config['max_length_question'] = data_generator.max_length_question

            # Load weights
            compile_info = {
                'loss': MaskedLoss(),
                'optimizer': tf.keras.optimizers.Adam(learning_rate=8e-6)}

            model_config['batch_size'] = 1
            qg_model = build_trainer(model_config, embedding_matrix_context, embedding_matrix_question, compile_info)
            qg_model.load_weights('./models/weights/weights_1')

            # Predict the questions
            context = input("Enter the sentence: \n")
            print("Context: ", context)

            context = preprocess_input(context, data_generator)
            print("Processed: ", context.shape)

            max_length = input("Enter output max length: ")

            evaluation_config['temperature'] = 0.0
            qg_predictor = Evaluator(model=qg_model,
                                    evaluation_config=evaluation_config,
                                    tokenizer_question=data_generator.tokenizer_question,
                                    tokenizer_context=data_generator.tokenizer_context)

            # context = tf.repeat(context, model_config['batch_size'], axis=0)
            predictions = qg_predictor.predict_step(context, int(max_length), False, True)
            print(predictions['text'])
        elif "-e" in args or "--evaluate" in args:
            # raise Exception("Not Yet Implemented!")
            print("Evaluating the model")
            # Libraries
            # Evaluate the model
        elif "-q" in args or "--predict-test" in args:
            print("Predicting questions on the test set")

            # Libraries
            from data_loader.data_generator import SQuAD
            from utils.utils import prepare_embeddings, build_trainer, train_model
            from utils import dirs
            from configs.config import dataset_config, model_config, path, evaluation_config
            from models.eval.evaluator import Evaluator
            from models.callbacks import BatchLogs
            from models.loss import MaskedLoss
            import pprint
            import os
            import tensorflow as tf
            import pickle

            model_config = model_config

            dataset, word_to_idx_context, word_to_idx_question, data_generator = load_dataset(dataset_config, path)

            embedding_matrix_context, embedding_matrix_question = prepare_embeddings(word_to_idx_context=word_to_idx_context, word_to_idx_question=word_to_idx_question)

            model_config['max_length_context'] = data_generator.max_length_context
            model_config['max_length_question'] = data_generator.max_length_question

            # Load weights
            compile_info = {
                'loss': MaskedLoss(),
                'optimizer': tf.keras.optimizers.Adam(learning_rate=8e-6)}

            qg_model = build_trainer(model_config, embedding_matrix_context, embedding_matrix_question, compile_info)
            qg_model.load_weights('./models/weights/weights_1')

            max_length = input("Enter output max length on which to compute the predictions: ")

            # evaluation_config['temperature'] = 0.0
            qg_predictor = Evaluator(model=qg_model,
                                    evaluation_config=evaluation_config,
                                    tokenizer_question=data_generator.tokenizer_question,
                                    tokenizer_context=data_generator.tokenizer_context)

            # context = tf.repeat(context, model_config['batch_size'], axis=0)
            predictions = qg_predictor.predict(dataset.test, int(max_length), False, True)
            print('Outputting the first ten predicted phrases on the test set')

            for sen in predictions[0]['text'][:10]:
                print(sen)

        elif "-h" in args or "--help" in args:
            print("Call the program with the following arguments: -t or --train to train the model, -p or --predict to predict a question, -e or --evaluate to evaluate the model, -q or --predict-test to predict questions on the test set")

