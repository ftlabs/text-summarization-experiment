# FTLabs Tensorflow Project

*Another document containing all the information relevant to this project, can be found* [here](https://docs.google.com/document/d/1NBYFPBjO3Fqwv7RFy4imiMDRkBLxr1GF39RrHuslQFA/edit#)

**Outline**
- S-API, search for articles
- C-API, retrieve the articles
- Transform the articles in the appropriate form, so that we can train, evaluate and test the DNN on them
- Create an EC2 instance on AWS to develop the model. We started developing the model of this project using a t2.xlarge instance and then
	moved to a machine with GPU, which increased the performance by around 5 times.
- Setup Tensorflow, clone textsum model, check that everything has been installed properly
- Training-Evaluating-Testing the model
	* Define the parameters' values (data_path, vocab_path, article_key, abstract_key, learning rate, layers etc) or go with their defaults and start the training
	* Run evaluation as the training goes on in order to see how the performance of the model improves towards unseen data
	* Test the model (mode= decode) to receive predictions, either on unseen or training data
	* Use Tensorboard to visualize parameters and the values of useful metrics
- Share the trained model both through the EC2 instance and/or as a copy distributed to other local machines


This project is our first attempt to make use of Tensorflow and specifically
the textsum model. This model enables us to create article summaries in an automated way, which is one
of the areas that we are currently researching. In the field of automated text summarization, Deep learning
is currently the most promising approach. The whole project has been developed on Tensorflow 1.0.1.

In the following sections you will find further details and instructions on how to complete the steps that were outlined above.

**Content Data**
To get started you first need to get hold of the data (articles and their 
metadata), to do that you will need keys to access the search API and the 
content API. Then set SAPI_key and CAPI_key environment variables with their 
corresponding values and run the ft-content-tensorflow.ipynb, this is an 
IPython script so you will need to install IPython notebooks in order to run 
it, more info on that can be found [here](https://ipython.org/install.html). 
After running this script you should have a separate json file for each 
article published from 2008 and on. The json files will be organised in 
folders based on the year and the month they were published. Then you will also
have aggregated files with content text data for each month and each year.
A corresponding vocabulary is also constructed from each content text data 
file, containing all the tokens encountered along with their absolute
frequencies.
As of March 2017, a bit more than half a millions articles were available and 
were retrieved.
In order to train the model a large amount of data-articles is needed, anything 
around 500,000 articles should be sufficient. 


After gathering all the data needed, you will have to install Tensorflow. 
This project was developed under Tensorflow 1.0.1, so more recent 
versions might not be compatible. More information on how to install 
Tensorflow, can be found on [this page](https://www.tensorflow.org/install/). 

We are making use of the textsum model provided [here](https://github.com/tensorflow/models/tree/master/textsum).
Our initial effort was to train a model that would generate titles of articles given their body/text.
Based on the results of that project we will then move on to attempt training a model for text summarization.

**Training**

The training set needs to have the same structure as the text-data-sample file provided 
and it needs to be converted to binary before the model can train on it. To convert the 
text file into binary you can use the data_convert_example.py provided with textsum.

To start the training you need to run the following, as described in models/textsum:

bazel-bin/textsum/seq2seq_attention \
--mode=train \
--data_path={path to the training data binary file} \
--vocab_path={path to the vocabulary file} \
--article_key=article \
--abstract_key=abstract \
--log_root=textsum/log_root \
--train_dir=textsum/log_root/train

We can also define the number of maximum epochs that we want the training to run, by
adding: "--max_run_steps={number of maximum epochs}" .


**Evaluating**

bazel-bin/textsum/seq2seq_attention \
--mode=train \
--data_path={path to the evaluation data binary file} \
--vocab_path={path to the vocabulary file} \
--article_key=article \
--abstract_key=abstract \
--log_root=textsum/log_root \
--eval_dir=textsum/log_root/eval

We can also define the number of maximum epochs that we want the evaluation to run, by
adding: "--max_run_steps={number of maximum epochs}" .


**Decoding**

bazel-bin/textsum/seq2seq_attention \
--mode=decode \
--data_path={path to the test data binary file} \
--vocab_path={path to the vocabulary file} \
--article_key=article \
--abstract_key=abstract \
--beam_size={beam size for beam search decoding}
--log_root=textsum/log_root \
--eval_dir=textsum/log_root/eval

**Moving the already trained model**

While training the model we can then move it to another machine to continue the training or use the model to see how it
performs. In order to do that we just need to copy the log_root folder, which contains the most recent checkpoints that 
were created during the training. Pasting the log_root folder inside the textsum directory of a project already built with bazel,
will carry over everything that is needed to have an identical model.
