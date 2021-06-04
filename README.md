# cml-mnist

This is a simple classification project based on the MNIST dataset

The focus is not really testing  various models but the implementation of CI/CD on a machine learning project

For that reason https://cml.dev/ is used.

Continuous Machine Learning (CML) is an open-source library for implementing continuous integration & delivery (CI/CD) in machine learning projects. 
Use it to automate parts of your development workflow, including model training and evaluation, comparing ML experiments across your project history, and monitoring changing datasets.


The normal workflow is:

- GitHub will deploy a runner machine with a specified CML Docker environment

- The runner will execute a workflow to train a ML model (python train.py)

- A visual CML report about the model performance will be returned as a comment in the pull request

The key file enabling these actions is .github/workflows/cml.yaml

This is demontrated in the [pull request](https://github.com/Inquiring-Nomad/cml-mnist/pull/1#commitcomment-51722989) submitted for the experiment branch , which generates a report markdown of the accuracy as well as the confusion matrix of the model.

