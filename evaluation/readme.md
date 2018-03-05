# Evaluation Framework

We provide a framework to evaluate cross-lingual sentence embeddings. 
Important features are:
  * Clean software architecture: components are (mostly) decoupled
  * Random subsample validation to mitigate effects from randomness (e.g. random initialization)
  * Easily extensible (new models, new datasets, new tasks, ...)
  * Automated execution of different configurations with the generation of excel reports

> We will provide additional sample configurations soon when we released the data for the cross-lingual tasks. 

## Requirements

   * Python 2.7
   * tensorflow (CPU is fine, we used TF 1.2)


## Setup

  * Run ```pip install -r requirements.txt``` to install all required python packages    


## Run Experiments

You can either run one individual experiment or many subsample runs over multiple configurations and different tasks. 
The latter also includes optimization of the learning rate, which is an important parameter that can have a noticeable 
influence on the results.


### Individual Experiments

For each experiment you need a configuration file that defines important parameters and specifies which modules should
be loaded. An example is given in `configs/individual/AC_fr.yaml` for the execution of a single cross-lingual (en->fr)
experiment on the AC dataset. See this configuration file for further documentation. 

You can run this experiment with: `python run_single.py configs/infividual/AC_fr.yaml`.


### Automation: Multiple Experiments

In our paper we used random subsample validation over many repeated runs and optimize the learning rate. We performed 
this for a number of different sentence embeddings on a number of different datasets. This makes a lot of different 
experiments. Of course, we automated most of it.

The script `run_multiple_optim_lr.py` will automatically run many different configurations on different tasks, and at 
the same time optimize the learning rate for each individual configuration. After finishing each task, an excel report 
will be written, which summarizes the model performances.

For this mode, we need a separate configuration file that specifies all tasks and runs. See `configs/multiple.yaml` for
an example. This file specifies two tasks with different runs and many repetitions for random subsample validation. 
Importantly, this configuration relies on templates that specify general task configurations. An example is the file
`configs/subsample_templates/AC.yaml`. This file specifies general parameters but uses template strings for parameters
that are specific to a run, e.g. the type of word embeddings that should be concatenated (the model).

You can run this experiment with: `python run_multiple_optim_lr.py configs/multiple.yaml`. The excel report will be 
written to the current folder.


## Extending the Software

Our software can be extended with new models, new datasets, new training procedures, or new evaluation 
scripts: Just implement the interfaces specified in `experiment/__init__.py` in a separate file and create a variable
named "component" in the same file that points to the component class. You can now use this component in your 
configuration files (e.g. as model-module). 

To create a new module you can also just copy an existing module and replace its implementation. E.g. copy 
`experiment.sentence_classification.model.linear` and implement an MLP instead.