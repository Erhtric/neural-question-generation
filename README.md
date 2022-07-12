# Question Generation using Deep Learning

This repository contains a final project realized for the **Natural Language Processing** course of the [Master's degree in Artificial Intelligence](https://corsi.unibo.it/2cycle/artificial-intelligence), University of Bologna.

# Table Of Contents

-  [Data](#data)
-  [Project Details](#project-details)
    -  [Folder structure](#folder-structure)
    -  [Technologies and Frameworks](#technologies-and-frameworks)
    -  [Configurations and enviroments](#configurations-and-enviroments)
    -  [Versioning](#versioning)
 -  [Future Works](#future-works)
 -  [Bibliography](#bibliography)
 -  [License](#license)

# Data

The dataset on which we trained, developed and tested our *Question Generation* (QG) network is the ***Stanford Question Answering Dataset*** (**SQuAD**) version 1.1, which is a collection of question-answer pairs derived from Wikipedia articles. The dataset was processed in order to better accomodonate our needs for the implementation.

# Project Details

This project tries to solve the **Question Generation** task by using the ideas introduced in the paper by *Du et al.* [[1]](#1-learning-to-ask-neural-question-generation-for-reading-comprehensionhttpsaclanthologyorgp17-1123-du-et-al-acl-2017). It acknowledge that by implementing our revisited version of the model proposed in the 2017 by exploiting newer technologies and using the acclaimed Tensorflow framework provided by Google. To this end, this project only purpouse is only an educational and we do not reserve any credit for the great work done by Du et al.

## Folder structure
--------------

```
├── goqu.py
│
├── GoQUreport.pdf
│
├── requirements.txt
│
├──  configs
│   └── config.py   - this file contains the configurations for the project.
│
├──  data   - this folder contains the data for the training and some additional files useful for further operations.
│
├── models
│   └── eval
│       ├── eval_metrics.py     - this file contains the metrics used for the evaluation.
│       └── evaluator.py        - this file contains the class used for evaluation.
│   └── layers
│       ├── decoder.py          - this file contains the decoder layer.
│       ├── encoder.py          - this file contains the encoder layer.
│       └── masking.py          - this file contains the custom masking layer.
│   └── trainers
│       ├── keras_tuner.py      - this file contains the code for the automatic tuning.
│       ├── trainer.py          - this file contains the class used for training.
│       └── metrics.py          - this file contains the metrics used for evaluating training.
│   ├── weights             - this folder contains the pre-trained weights from colab.
│   ├── loss.py         - this file contains the loss used by the model
│   └── callbacks.py    - this file contains the classes used as callbacks
│
├── data_loader
│   └── data_generator.py   - this file contains the dataset methods for loading and processing it.
│
└── utils   - this folder contains utility methods useful for complementary operations
     ├── dirs.py
     ├── embeddings.py
     └── utils.py

```

## Technologies and Frameworks

Frameworks:
- [Tensorflow (v2.5.0)](https://www.tensorflow.org/)
- [Datasets (v1.18.4)](https://github.com/huggingface/datasets)

Platforms
- [Google Colaboratory]()

## Configurations and enviroments

The `config.py` file contains all the configurations needed by the project. The environment could be loaded by using `conda` by launching the command:
```shell
$ conda create --name <env> --file requirements.txt
```

## Versioning

We used Git for versioning.

# Future Works
Possible improvements to this project could be:
- encoding additional information to the embedding dimension, this means that we could concatenate to each word vector its NER and POS tags to augment the information given to the network,
- adding a more sophisticated decoding in the last part, instead of using the temperature sampling, see [beam search decoding](https://scholar.google.it/scholar?q=beam+search+decoding&hl=en&as_sdt=0&as_vis=1&oi=scholart),
- use [contextual word embeddings](https://scholar.google.it/scholar?hl=en&as_sdt=0%2C5&as_vis=1&q=contextual+word+embeddings&btnG=&oq=contextual+word),
- use a different model, maybe more sophisticated.

# Bibliography

### 1. [Learning to Ask: Neural Question Generation for Reading Comprehension](https://aclanthology.org/P17-1123) (Du et al., ACL 2017)

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.
