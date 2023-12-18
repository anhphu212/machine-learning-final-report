# machine-learning-final-report
## Table of contents
**[CHAPTER 1: OPTIMIZATION ALGORITHMS IN MACHINE LEARNING]**
- 1.1 Introduction
- 1.2 How do Optimizers work?
- Optimization algorithms
  - 1.3.1 Gradient Descent
    - 1.3.1.1 What is Gradient descent?
    - 1.3.1.2 Importancce of Learning rate
    - 1.3.1.3 Advantages and disadvantages
  - 1.3.2 Stochastic Gradient Descent (SGD)
  - 1.3.3 Mini Batch Stochastic Gradient Descent (MB-SGD)
  - 1.3.4 SGD with Momentum
  - 1.3.5 Nesterov Accelerated Gradient (NAG)
  - 1.3.6 Adaptive Gradient Descent(AdaGrad)
  - 1.3.7 AdaDelta
  - 1.3.8 RMSprop
  - 1.3.9 Adaptive Moment Estimation (Adam)
- 1.4 When to choose this algorithm?
  
**[CHAPTER 2: CONTINUAL LEARNING AND TEST IN PRODUCTION]**
- 2.1 Continual Learning
  - 2.1.1 Why Continual Learning?
  - 2.1.2 Concept: Stateless retraining VS Stateful training
    - 2.1.2.1 Stateless retraining
    - 2.1.2.2 Stateful training (aka fine-tuning, incremental learning)
  - 2.1.3 Concept: feature reuse through log and wait
  - 2.1.4 Continual Learning Challenges
    - 2.1.4.1 Fresh data access challenge
    - 2.1.4.2 Evaluation Challenge
    - 2.1.4.3 Data scaling challenge
    - 2.1.4.4 Algorithm challenge
  - 2.1.5 The Four Stages of Continual Learning
    - 2.1.5.1 Stage 1: Manual, stateless retraining
    - 2.1.5.2 Stage 2: Fixed schedule automated stateless retraining
    - 2.1.5.3 Stage 3: Fixed schedule automated stateful training
    - 2.1.5.4 Stage 4: Continual learning
  - 2.1.6 How often to Update your models
    - 2.1.6.1 Measuring the value of data freshness
    - 2.1.6.2 When should I do model iteration?
- 2.2 Testing models in Production
  - 2.2.1 Pre-deployment offline evaluations
  - 2.2.2 Testing in Production Strategies
    - 2.2.2.1 Shadow Deployment
    - 2.2.2.2 A/B Testing
    - 2.2.2.3 Canary Release
    - 2.2.2.4 Interleaving Experiments
    - 2.2.2.5 Bandits
