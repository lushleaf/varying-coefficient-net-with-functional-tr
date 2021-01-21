# Varying Coefficient Neural Network with Functional Targeted Regularization for Estimating Continuous Treatment Effects
\[ICLR 2021\] Code for: [Varying Coefficient Neural Network with Functional Targeted Regularization for Estimating Continuous Treatment Effects](https://openreview.net/forum?id=RmB-88r9dL)

We investigate the problem of estimating the average dose-response curve (ADRF) with neural network model. We develop a new network architecture called varying coefficient network that is powful in representing the treatment effect while preserving the continuous structure of ADRF. To improve finite sample performance, we generalize targeted regularization to obtain a doubly robust estimator of the whole ADRF curve.

A typical comparison of estimated ADRF with a previous model is as follows.

<img src="fig/Vc_Dr.png" width=300></img>

## How to run

-- generate simulated data

    simu1_generate_data.py

-- train and evaluating the methods

To run a singe run of models/methods with one dataset:
    
    main.py

You can also use it to generate estimated ADRF curve (Fig 1 in the paper).

To run all models/methods in numbers of datasets, please use

    main_batch.py

Some sample command to produce our experiment.
    
    run.sh
