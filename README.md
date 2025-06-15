## exploring overfitting, underfitting, and regularization with pytorch cnn on cifar-10

### overview

1. this project explores the impact of underfitting, overfitting, and various regularization techniques (l1, l2, elastic net) and early stopping on a convolutional neural network (cnn) (cifar-10) while training from scratch. 
2. the goal of the project is to visually and quantitatively compare how these techniques help mitigate overfitting and improve model generalization.
3. the experiment uses the cifar-10 dataset, a standard benchmark for image classification.

### key concepts explored

1.  **underfitting:** training a model that is too simple or not trained long enough to capture the underlying patterns in the data.
2.  **overfitting:** training a model that learns the training data too well, including noise and specific patterns, leading to poor performance on unseen data (validation/test sets). identified by diverging training and validation performance metrics (e.g., loss increasing on validation set while decreasing on training set).
3.  **l2 regularization (weight decay):** adds a penalty proportional to the square of the magnitude of model weights to the loss function. encourages smaller weights, leading to simpler models less sensitive to input variations. implemented via `optim.adamw`'s `weight_decay`.
4.  **l1 regularization (lasso):** adds a penalty proportional to the absolute value of the magnitude of model weights. encourages sparsity (some weights become exactly zero), potentially performing feature selection. implemented manually by adding the l1 norm to the loss.
5.  **elastic net regularization:** a combination of l1 and l2 regularization, aiming to leverage the benefits of both. implemented using both `weight_decay` and manual l1 loss addition.
6.  **early stopping:** a technique where training is halted when the model's performance on a validation dataset stops improving (or starts degrading) for a predefined number of epochs (patience). this prevents the model from training too far into the overfitting regime.


### results

### training history plots
![plot results 1](https://github.com/user-attachments/assets/e00ba250-383e-4bcb-819a-46bb35d2bca9)

### final evaluation on test set

the models were evaluated on the unseen cifar-10 test set after training completion (or early stopping).

| model             |   test loss |   test accuracy |   stopped epoch |
| :---------------- | ----------: | --------------: | --------------: |
| underfitting      |    0.711778 |          0.7546 |              60 |
| baseline          |    0.444848 |          **0.8875** |              60 |
| l2 regularization |    0.468008 |          0.8736 |              60 |
| l1 regularization |    **0.390260** |          0.8786 |              60 |
| elastic net       |    0.421788 |          0.8814 |              60 |
| early stopping    |    0.427962 |          0.8643 |              **32** |


### llm usage

this is an exporation/educational project whose codes were generated using ai tools. used generative ai tools to write cifar model on pytorch and implement regularization methods to study its affects on result accuracy. 