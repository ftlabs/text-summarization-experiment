# FTLabs Tensorflow Project

*Another document containing more information relevant to this project, can be found* [here](https://docs.google.com/document/d/1NBYFPBjO3Fqwv7RFy4imiMDRkBLxr1GF39RrHuslQFA/edit#)

## Outline
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
	* Use Tensorboard to visualize parameters and the values of several metrics
- Share the trained model both through the EC2 instance and/or as a copy distributed to other local machines


This project is our first attempt to make use of Tensorflow and specifically
the textsum model. This model enables us to create article summaries in an automated way, which is one
of the areas that we are currently researching. In the field of automated text summarization, Deep learning
is currently the most promising approach. The whole project has been developed on Tensorflow 1.0.1.

In the following sections you will find further details and instructions on how to complete the steps that were outlined above.

## Installing Tensorflow

Before gathering all the data needed, you will have to install Tensorflow. 
This project was developed under Tensorflow 1.0.1, so more recent 
versions might not be compatible. More information on how to install 
Tensorflow, can be found on [this page](https://www.tensorflow.org/install/). 

We are making use of the textsum model provided [here](https://github.com/tensorflow/models/tree/master/textsum). 
So on top of Tensorflow, textsum must be configured and build with bazel, as described in the github repository.

Since we did not have articles with their summaries to use as a training set, our initial effort was to train a model that would generate 
titles of articles given their body/text.
Based on the results of that effort we will then move on to attempt training a model for text summarization.


## Content Data
To get started you first need to get hold of the data (articles and their 
metadata), to do that you will need keys to [access the search API and the 
content API](https://developer.ft.com/docs). Then set SAPI_key and CAPI_key environment variables with their 
corresponding values and run the ft-content-tensorflow.ipynb, this is an 
IPython script so you will need to install IPython notebooks in order to run 
it, more info on that can be found [here](https://ipython.org/install.html). 
After running the relevant parts of this script you should have a separate json file for each 
article published from 2008 and on. The json files will be organised in 
folders based on the year and the month they were published. You will also
have aggregated files with content data in text format for each month and each year.
A corresponding vocabulary is also constructed from each content data text 
file, containing all the tokens encountered along with their absolute frequencies and finally merged into an aggregated vocabulary containing the 300,000 most common tokens. In order to reduce the number of unique tokens and make training more efficient we convert all tokens to lowercase, this is optional.
As of March 2017, a bit more than half a millions articles were available and were retrieved.
In order to train the model a large amount of data-articles is needed, this experiment was done using around 500,000 articles
for training the model. 

## Training

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


## Evaluating

Evaluation should be performed as we train the model, to make sure our model is generalising correctly and is not overfitting on the training set.
Running evaluation on unseen data is crucial, to achieve the above and identify when the training process should be stopped.

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


## Decoding

In order to get predictions (titles, summaries or whatever our model is supposed to generate), we need to run our model with "mode=decode",
we can also define the beam_size which is the number of possible predictions our model will generate before it decides which one is the best.


bazel-bin/textsum/seq2seq_attention \
--mode=decode \
--data_path={path to the test data binary file} \
--vocab_path={path to the vocabulary file} \
--article_key=article \
--abstract_key=abstract \
--beam_size={beam size for beam search decoding}
--log_root=textsum/log_root \
--eval_dir=textsum/log_root/eval

## Moving the already trained model

Moving an already existing/trained model to another machine either to continue training or use it to get some results, is a very easy task.
Just copy the log_root folder, as defined during the training, evaluation and decoding process. This folder contains the most recent checkpoints that 
were created during the training, the results produced when decoding and the evaluation scores. Pasting the log_root folder inside the textsum directory of a project already built with bazel,
will create an identical model ready to use.

## Relevant projects & Useful links

1. [Sumy](https://pypi.python.org/pypi/sumy)
2. [Gensim](https://github.com/RaRe-Technologies/gensim)
3. [Smmry](http://smmry.com/about)
4. [TextSum on GPU](https://eilianyu.wordpress.com/2016/10/17/text-summarization-using-sequence-to-sequence-model-in-tensorflow-and-gpu-computing/)
5. [Improved text summarization model](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html)
6. [Keyword Generator](https://github.com/jlonij/keyword-generator)
7. [Abstractive Summarization via Phrase Selection and Merging](http://www.cs.cmu.edu/~lbing/pub/acl2015-bing.pdf)
