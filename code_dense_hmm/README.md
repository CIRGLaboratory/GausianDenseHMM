# GaussianDenseHMM

This document contains extra information for our modification of the DenseHMM for Gaussian emission.

All theoretical details and experimetns summary can be found in the document: 
[https://www.overleaf.com/read/nwsjtkqfpmhr](https://www.overleaf.com/read/nwsjtkqfpmhr)
Contact Klaudia or Piotr if you need editing rights.

##  Models

[models_gaussian.py](models_gaussian.py) - our implementation of GaussianDenseHMM (based on DenseHMM and hmmlearn library)

Contains custom implementation of standard GaussianHMM and implementation of DenseHMM with EM and co-occurrence based learning.

[models_gaussian_A.py](models_gaussian_A.py) - scratch implementation for learning GaussianHMM basing on co-occurrences (only needed changes on above script). 

## Tests

1. Results of GaussianDenseHMM with co-occurrence based learning for different data and model sizes.

   - [test_gaussian.py](test_gaussian.py) - run experiments
   - [test_gaussian.ipynb](test_gaussian.ipynb) - present results

2. Check what's the cost of different changes in models (continuous emission, dense representation).
    
   - [eval_cooc.py](eval_cooc.py) - run experiments
   - [eval_cooc.ipynb](eval_cooc.ipynb) - present results

3. Benchmark for GaussianDenseHMM with EM learning:

   - [simple_test.py](simple_test.py)

4. Simple demonstration of the models' operation:

   - [check_gaussian_cooc.ipynb](check_gaussian_cooc.ipynb) - standard model with co-occurence based learning (uses scratch implementation)
   - [co-occurrence_expectations.ipynb](co-occurrence_expectations.ipynb) - a simulation how do co-occurence matrix differ for real and sample distribution.
   - [discretisation.ipynb](discretisation.ipynb) - illustration of selected discretization technique

