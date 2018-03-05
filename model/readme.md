# Model

This application runs our concatenated p-means model. It automatically downloads all required resources from the web
and starts a webserver that allows researchers to generate sentence embeddings over a simple HTTP API.


## Requirements

   * Python 2.7


## Setup

  * Run ```pip install -r requirements.txt``` to install all required python packages


## Running the Application

  * You can run the application using ```python main.py```. It will standardly load the `en-de` model and
  start a webserver on port 5000 with a HTTP API that allows you to generate sentence embeddings.
  * Run ```python main.py --help``` to see all possible options.

When the application finished loading all word embeddings, 
just visit http://localhost:5000 in your webbrowser for further instructions and example python code.

   It is important to mention that we standardly load a smaller version of fasttext with only the 300k most frequent 
   tokens for reduced file size and faster downloads. To fully reproduce our cross-lingual experiments, use the full
   fasttext files (see comments in main.py).


## Extending the Application

The application can be extended with further word embeddings, p-means, and other moments. 


#### Adding new models and additional word embeddings

The file `main.py` defines a variable named `embeddings` that holds a dictionary defining all models. For each model 
there is a list of word embedding definitions. You can freely extend this list, or add new models by adding new entries
to the dictionary.


#### Defining p-means or moments

In the file `sentence_embeddings.py` we define a variable named `operations` that holds a dictionary that specifies all 
available operations (compression strategies, p-means). You can add new entries here and define arbitrary
operations such as additional p-means, moments, or further compression and summarization strategies.

All of those newly defined operations (and word embeddings) will appear in the web interface.
