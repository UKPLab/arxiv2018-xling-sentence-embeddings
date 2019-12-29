# Concatenated Power Mean Embeddings as Universal Cross-Lingual Sentence Representations

This repository contains the data and code to reproduce the results of our paper: https://arxiv.org/abs/1803.01400 
It also contains the cross-lingual word embeddings that we used in our experiments, translated variants of SNLI, and our code to map embeddings of two languages into common space.

Please use the following citation:

```
@article{rueckle:2018,
  title = {Concatenated Power Mean Embeddings as Universal Cross-Lingual Sentence Representations},
  author = {R{\"u}ckl{\'e}, Andreas and Eger, Steffen and Peyrard, Maxime and Gurevych, Iryna},
  journal = {arXiv},
  year = {2018},
  url = "https://arxiv.org/abs/1803.01400"
}
```

> **Abstract:** Average word embeddings are a common baseline for more sophisticated sentence embedding techniques. However, they typically fall short of the performances of more complex models such as InferSent. Here, we generalize the concept of average word embeddings to power mean word embeddings. We show that the concatenation of different types of power mean word embeddings considerably closes the gap to state-of-the-art methods monolingually and substantially outperforms these more complex techniques cross-lingually. In addition, our proposed method outperforms different recently proposed baselines such as SIF and Sent2Vec by a solid margin, thus constituting a much harder-to-beat monolingual baseline.


Contact persons: Andreas Rücklé, Steffen Eger

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 


## Usage

We offer several [TF-Hub](https://www.tensorflow.org/hub/) modules for convenience:

```
url_de = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/en-de/1'
url_fr = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/en-fr/1'
url_monolingual = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/monolingual/1'

embed = hub.Module(url)
representations = embed(["A_en long_en sentence_en ._en", "another_en sentence_en"])
```

The input strings have to be tokenized (tokens split by spaces), postfixed with _en/_de/_fr (except for the monolingual model) **and lowercased**. (We usually don't lowercase everything but at this time we don't see a simple method of doing this in a saved TF graph.) If you want to work with non-lowercased sequences, download and run the model as described below.


For full reproducibility please use our python code:

```
cd model
pip install -r requirements.txt
python main.py
```


The figure below shows the average monolingual performance of the different sentence embeddings models that we tested in relation to their dimensionality (this is figure 1 from our paper). The TF-Hub modules contain our full model (all power means and concatenations). The python code in ```/model``` can be used to obtain sentence embeddings for other concatenations and power mean combinations. To achieve the best results with our models, we recommend normalizing the sentence embeddings with the z-norm.

<img src="https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings/raw/master/figure-1.png?raw=true" width="500px">


## Sub-Projects

This repository contains different sub-projects:

```
<ROOT>
├── README.md
├── model/
├── evaluation/
├── data/
└── map-word-embeddings/
```

**[Model](model/)**
This is our concatenated p-means model. On execution we will automatically fetch all required resources and provide an embeddings webserver that can generate sentence embeddings using our models (en-de, en-fr, monolingual).

**[Evaluation](evaluation/)**
Contains our evaluation framework that we use to evaluate the three additional tasks we provide (mainly from argumentation mining).

**[Data](data/)**
We provide our datasets and other resources in this folder. This includes our cross-lingual tasks. 

**[Map-Word-Embeddings](map-word-embeddings/)**
We provide the software that we used to induce our cross-lingual word embeddings and to re-map existing ones. See the appendix of our paper for more details.



## Additional Downloads

  * Cross-lingual SNLI: [en-de](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/translated-snli/en-de-translated-snli-4x.zip), [en-fr](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/translated-snli/en-fr-translated-snli-4x.zip)
  * en-de cross-lingual word embeddings: [BIVCD](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_bivcd_en_de.txt.gz), [AttractRepel](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_attract_repel_en_de.txt.gz), [Fasttext (300K)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_300k_en_de.txt.gz), [Fasttext (Full)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_en_de.txt.gz)
  * en-fr cross-lingual word embeddings: [BIVCD](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_bivcd_en_fr.txt.gz), [AttractRepel](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_attract_repel_en_fr.txt.gz), [Fasttext (300K)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_300k_en_fr.txt.gz), [Fasttext (Full)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_en_fr.txt.gz)


More details can be found in the [data folder](data/).
