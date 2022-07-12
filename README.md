# Question Generation using Deep Learning

This repository contains a final project realized for the Natural Language Processing course of the Master's degree in Artificial Intelligence, University of Bologna.

# Table Of Contents

-  [Data](#data)
-  [Project Details](#project-details)
    -  [Folder structure](#folder-structure)

# Data

The dataset on which we trained, developed and tested our QG network is the ***Stanford Question Answering Dataset*** (**SQuAD**) version 1.1, which is a collection of question-answer pairs derived from Wikipedia articles. The dataset was processed in order to better accomodonate our needs for the implementation.

# Project Details


Folder structure
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
│
├── data_loader
│   └── data_generator.py   - this file contains the dataset methods for loading and processing it.
│
└── utils   - this folder contains utility methods useful for complementary operations
     ├── dirs.py
     ├── embeddings.py
     └── utils.py

```

<!--
# Table Of Contents

-  [Data](#data)
-  [In Details](#in-details)
    -  [Project architecture](#project-architecture)
    -  [Folder structure](#folder-structure)
    -  [ Main Components](#main-components)
        -  [Models](#models)
        -  [Trainer](#trainer)
        -  [Data Loader](#data-loader)
        -  [Logger](#logger)
        -  [Configuration](#configuration)
        -  [Main](#main)
    -  [Technologies and Frameworks](#technologies-and-frameworks)
 -  [Future Work](#future-work)
 -  [Bibliography](#bibliography)
 -  [Acknowledgments](#acknowledgments)





## Main Components

### Models
--------------
- #### **Base model**
    
    Base model is an abstract class that must be Inherited by any model you create, the idea behind this is that there's much shared stuff between all models.
    The base model contains:
    - ***Save*** -This function to save a checkpoint to the desk. 
    - ***Load*** -This function to load a checkpoint from the desk.
    - ***Cur_epoch, Global_step counters*** -These variables to keep track of the current epoch and global step.
    - ***Init_Saver*** An abstract function to initialize the saver used for saving and loading the checkpoint, ***Note***: override this function in the model you want to implement.
    - ***Build_model*** Here's an abstract function to define the model, ***Note***: override this function in the model you want to implement.
- #### **Your model**
    Here's where you implement your model.
    So you should :
    - Create your model class and inherit the base_model class
    - override "build_model" where you write the tensorflow model you want
    - override "init_save" where you create a tensorflow saver to use it to save and load checkpoint
    - call the "build_model" and "init_saver" in the initializer.

### Trainer
--------------
- #### **Base trainer**
    Base trainer is an abstract class that just wrap the training process.
    
- #### **Your trainer**
     Here's what you should implement in your trainer.
    1. Create your trainer class and inherit the base_trainer class.
    2. override these two functions "train_step", "train_epoch" where you implement the training process of each step and each epoch.
### Data Loader
This class is responsible for all data handling and processing and provide an easy interface that can be used by the trainer.
### Logger
This class is responsible for the tensorboard summary, in your trainer create a dictionary of all tensorflow variables you want to summarize then pass this dictionary to logger.summarize().


This class also supports reporting to **Comet.ml** which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.
Add your API key [in the configuration file](configs/example.json#L9):

For example: "comet_api_key": "your key here"


### Comet.ml Integration
This template also supports reporting to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric. 

Add your API key [in the configuration file](configs/example.json#L9):


For example:  `"comet_api_key": "your key here"` 

Here's how it looks after you start training:
<div align="center">

<img align="center" width="800" src="https://comet-ml.nyc3.digitaloceanspaces.com/CometDemo.gif">

</div>

You can also link your Github repository to your comet.ml project for full version control. 
[Here's a live page showing the example from this repo](https://www.comet.ml/gidim/tensorflow-project-template/caba580d8d1547ccaed982693a645507/chart)



### Configuration
I use Json as configuration method and then parse it, so write all configs you want then parse it using "utils/config/process_config" and pass this configuration object to all other objects.
### Main
Here's where you combine all previous part.
1. Parse the config file.
2. Create a tensorflow session.
2. Create an instance of "Model", "Data_Generator" and "Logger" and parse the config to all of them.
3. Create an instance of "Trainer" and pass all previous objects to it.
4. Now you can train your model by calling "Trainer.train()"

# Technologies and Frameworks

Frameworks:
- [Tensorflow (v2.9.0)](https://www.tensorflow.org/)

# Future Work
- MISSING

# Bibliography

1. [Learning to Ask: Neural Question Generation for Reading Comprehension](https://aclanthology.org/P17-1123) (Du et al., ACL 2017)

# Acknowledgments

Currently using the sources from `Python-templates/Tensorflow-Project-Template`, I want to thank them for the excellent work.
--->