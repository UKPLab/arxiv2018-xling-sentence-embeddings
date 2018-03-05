# Data

We provide preprocessing scripts and datasets that we use in our paper.


## Tasks 

> At this time we are preparing the (cross-lingual) task data for public release.


## Cross-Lingual Word Embeddings

As part of our work we trained word embeddings (BIVCD) and (re-)mapped others with the method described in the appendix of our paper. 

  * en-de word embeddings: [BIVCD](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_bivcd_en_de.txt.gz), [AttractRepel](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_attract_repel_en_de.txt.gz), [Fasttext (300K)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_300k_en_de.txt.gz), [Fasttext (Full)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_en_de.txt.gz)
  * en-fr word embeddings: [BIVCD](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_bivcd_en_fr.txt.gz), [AttractRepel](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_attract_repel_en_fr.txt.gz), [Fasttext (300K)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_300k_en_fr.txt.gz), [Fasttext (Full)](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/xling-wordembeddings/mapped_fasttext_en_fr.txt.gz)

Fasttext 300K only contain the 300K most frequent tokens (of both languages). The full versions are mapped variants of the full pre-trained fasttext. Use the full versions to reproduce our results.


## Translated SNLI

We trained our cross-lingual adaptations of InferSent on (machine-) translated cross-lingual variants of SNLI:

   * [en-de SNLI](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/translated-snli/en-de-translated-snli-4x.zip)
   * [en-fr SNLI](https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/translated-snli/en-fr-translated-snli-4x.zip)

The above contain SNLI with all possible language combinations of the sentence pairs (en-en, en-de, de-en, de-de). Thus, the datasets are four times as large as the original.

> We plan to release translated SNLI corpora in different languages soon (de,fr,es,ar).


## Licenses

Please read LICENSE.txt and NOTICE.txt in the project root. We distribute derivational data under the same license as the original.