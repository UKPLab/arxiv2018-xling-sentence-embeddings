# Cross-Lingual Mapping of Word Embeddings

We publish our scripts to map word embeddings of two separate vector spaces into a common shared vector space. 
Details of our method can be found in the appendix of our paper.  


## Requirements

   * Python 2.7
   * tensorflow (CPU is fine)
   
You also need a corpus of aligned sentences, one file per language (1st line of file a is translation of 1st line of 
file b). In our work we used 100K sentences from the TED corpus. UN or Europarl are further options. Make sure that all 
tokens are postfixed with their respective language flag (e.g. "table_en" and "tisch_de"). 

Finally you need word embeddings. The embeddings file should contain tokens from both languages. e.g. if you want to map
English and German word embeddings into shared space, concatenate both embedding files and make sure that every token 
from the English file has an "_en" postfix and every token from the German file has a "_de" postfix. 
**This embedding file has to be integrated into the embeddings webserver subproject.**


## Setup

  * Run ```pip install -r requirements.txt``` to install all required python packages


## Running the Application

  * First, start up the embeddings webserver subproject with the word embeddings that should be (re-)mapped.
  * Create a configuration file based on `example.yaml`. (This file contains further documentation on the configuration
  options.)
  * Run the application with ```python run_single.py <my-config.yaml>```.
  
The training process will start. As soon as the model converges a webserver will be started. This webserver provides 
a simple HTTP API to map individual word embeddings (or even sentence embeddings based on p-means).

The `map.py` gives an example on how to use the API. You can use it to map the whole embeddings file you trained:
`python map.py embeddings-path=/path/to/embs out-path=/path/to/embs.out mapping-server-url=http://127.0.0.1:8099 lang-a=en`.