# Varying Coefficient Neural Network with Functional Targeted Regularization for Estimating Continuous Treatment Effects
\[ICLR 2021\] Code for: [Varying Coefficient Neural Network with Functional Targeted Regularization for Estimating Continuous Treatment Effects](https://openreview.net/forum?id=RmB-88r9dL)

We investigate the problem of estimating the average dose-response curve (ADRF) with neural network model. We develop a new network architecture called varying coefficient network that is powful in representing the treatment effect while preserving the continuous structure of ADRF. To improve finite sample performance, we generalize targeted regularization to obtain a doubly robust estimator of the whole ADRF curve.

## How to run

-- generate simulated data
simu1_generate_data.py will generate simulated

-- train and evaluating the methods

to run a singe run of models/methods with one dataset, please use main.py
You can also use main.py to generate estimated ADRF curve (Fig 1 in the paper).

to run all models/methods in numbers of datasets, please use main_batch.py

Please see run.sh for some sample command to produce our experiment
