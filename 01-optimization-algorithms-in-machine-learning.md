# CHAPTER 1: OPTIMIZATION ALGORITHMS IN MACHINE LEARNING
Overview of some of the most used optimizers while training a neural network.

## 1.1 Introduction
In the field of machine learning, the concept of loss quantifies the model's current performance, indicating how poorly it is performing. The objective is to leverage this loss information to enhance the model's capabilities. The primary goal is to minimize the loss, as a lower loss corresponds to improved model performance. The systematic procedure of reducing (or increasing) any mathematical expression is referred to as optimization.

Optimizers represent algorithms or techniques employed to adjust neural network attributes, such as weights and learning rate, with the aim of diminishing losses. These optimization methods are crucial for addressing optimization problems by minimizing the associated function.

## 1.2 How do Optimizers work?
For a helpful analogy, envision a hiker navigating down a mountain while blindfolded. Determining the exact direction is impossible, but she can discern whether she's progressing downward (making headway) or upward (losing ground). By consistently choosing paths leading downward, she eventually reaches the mountain base.

Similarly, establishing the optimal weights for your model right from the start is challenging. Yet, through trial and error guided by the loss function (analogous to the hiker's descent), you can progressively converge on the right configuration.

The adjustments to neural network weights or learning rates necessary for reducing losses are dictated by the optimizers employed. Optimization algorithms play a crucial role in minimizing losses and delivering the most precise outcomes.

Over the past few years, numerous optimizers have been researched, each carrying its own set of pros and cons. To gain a comprehensive understanding of these algorithms, explore the entire article, which delves into their workings, advantages, and disadvantages.

The article will delve into various types of optimizers, elucidating their mechanisms for precisely minimizing the loss function.
  1. Gradient Descent
  2. Stochastic Gradient Descent (SGD)
  3. Mini Batch Stochastic Gradient Descent (MB-SGD)
  4. SGD with momentum
  5. Nesterov Accelerated Gradient (NAG)
  6. Adaptive Gradient (AdaGrad)
  7. AdaDelta
  8. RMSprop
  9. Adam
  
## 1.3 Optimization algorithms
### 1.3.1 Gradient Descent
Gradient descent is an optimization algorithm utilized during the training of machine learning models. Operating on a convex function, it systematically adjusts its parameters through iterative steps to reduce a specified function towards its local minimum.

#### 1.3.1.1 What is Gradient descent?
Gradient Descent stands as an optimization algorithm designed to locate a local minimum within a differentiable function. Its primary purpose is to identify the parameter values (coefficients) of a function that minimize a specified cost function.

The process commences by setting the initial values for the parameters, and then, through iterative steps guided by calculus, gradient descent systematically refines these values to minimize the given cost function.

Initialization strategies are employed to set the initial weight, and with each epoch, the weight undergoes updates in accordance with the prescribed update equation.


The given equation calculates the gradient of the cost function J(θ) w.r.t to the parameters or weights θ across the entire training dataset:

Our objective is to reach the lowest point on our graph, representing the relationship between the cost and weights (Cost and weights) or a point where further descent is no longer possible—a local minimum.

Now, let's explore the concept of "Gradient."
"A gradient measures how much the output of a function changes if you change the inputs a little bit." — Lex Fridman (MIT)

#### 1.3.1.2 Importance of Learning rate
The size of the steps that gradient descent takes toward the local minimum is determined by the learning rate, which dictates the speed of our movement towards the optimal weights.

To ensure that gradient descent effectively reaches the local minimum, it is crucial to set the learning rate to an appropriate value neither too low nor too high. This consideration is significant because excessively large steps might result in the algorithm bouncing back and forth within the convex function of gradient descent (as illustrated in the left image below). Conversely, if the learning rate is set too small, gradient descent will eventually reach the local minimum, but the process may be prolonged (as depicted in the right image below).

Therefore, it is essential to avoid setting the learning rate either too high or too low. To assess the effectiveness of the learning rate, it can be visualized on a graph.

In programming code, the implementation of gradient descent typically resembles the following structure:

for i in range(nb_epochs):   
    params_grad = evaluate_gradient(loss_function, data, params)          
    params = params - learning_rate * params_grad
    
Over a predetermined number of epochs, the initial step involves computing the gradient vector, denoted as params_grad, of the loss function concerning our parameter vector, params, across the entire dataset.

#### 1.3.1.3 Advantages and disadvantages
Advantages:
  - Straightforward computation.
  - Simple to implement.
  - Easily understandable.
    
Disadvantages:
  - Prone to getting stuck in local minima.
  - Weight adjustments occur after calculating the gradient on the entire dataset. Consequently, if the dataset is extensive, convergence to the minima may take an extended period.
  - Demands significant memory resources to calculate the gradient across the entire dataset.

### 1.3.2 Stochastic Gradient Descent (SGD)
The Stochastic Gradient Descent (SGD) algorithm is a development of the Gradient Descent, designed to address certain drawbacks of the GD algorithm. Gradient Descent is hindered by its need for substantial memory to process the entire dataset of n points simultaneously in order to compute the derivative of the loss function. In contrast, the SGD algorithm calculates the derivative by considering one point at a time, mitigating the memory requirements.

SGD conducts a parameter update for every training example, denoted as x(i), along with its corresponding label y(i):

θ = θ − α⋅∂(J(θ;x(i),y(i)))/∂θ

where {x(i) ,y(i)} are the training examples.
To expedite the training process, we perform a Gradient Descent step for each training example. The potential implications of this approach are illustrated in the image below.

  - On the left side, we observe Stochastic Gradient Descent (where m=1 per step), where a Gradient Descent step is taken for each individual example. On the right side is Gradient Descent (1 step per entire training set).
  - SGD exhibits noticeable noise, yet it is considerably faster, albeit with a risk of not converging to a minimum.
  - To strike a balance between the two approaches, Mini-batch Gradient Descent (MGD) is often employed. MGD involves examining a smaller subset of training set examples at a time, typically in batches of a certain size (commonly a power of 2, such as 2^6, etc.).
  - Mini-batch Gradient Descent offers relative stability compared to Stochastic Gradient Descent (SGD), although it does introduce oscillations since gradient steps are taken based on a sample of the training set rather than the entire set, as in Batch Gradient Descent (BGD).

In SGD, it is noted that updates require more iterations compared to gradient descent to reach the minima. On the right side, Gradient Descent takes fewer steps to reach the minima, but the SGD algorithm introduces more noise and demands more iterations.

The code segment for SGD involves the addition of a loop over the training examples, where the gradient with respect to each example is evaluated.

for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
        params_grad = evaluate_gradient(loss_function, example, params)
        params = params - learning_rate * params_grad

Advantage:
  - Reduced memory requirements compared to the Gradient Descent (GD) algorithm, given that the derivative is computed using only one point at a time.

Disadvantages:
  - Longer time is needed to complete one epoch compared to the GD algorithm.
  - Slower convergence.
  - Susceptible to getting stuck in local minima.

### 1.3.3 Mini Batch Stochastic Gradient Descent (MB-SGD)
MB-SGD algorithm extends the capabilities of the SGD algorithm, addressing the issue of high time complexity associated with SGD. MB-SGD, rather than considering one point at a time, takes a batch or subset of points from the dataset to compute derivatives.

It is observed that, after a certain number of iterations, the derivative of the loss function for MB-SGD closely resembles the derivative of the loss function for Gradient Descent. However, MB-SGD requires more iterations to reach the minima compared to GD, and the computational cost is also higher.

The weight update in MB-SGD relies on the derivative of the loss computed for a batch of points. The updates in the case of MB-SGD exhibit more noise because the derivative does not consistently point towards the minima.

MB-SGD partitions the dataset into several batches, and after processing each batch, the algorithm updates the parameters.

θ = θ − α⋅∂(J(θ;B(i)))/∂θ

where {B(i)} are the batches of training examples.
In the code, rather than iterating over individual examples, we now iterate over mini-batches, each containing 50 examples:

for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
        params_grad = evaluate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad

Advantage:
  - Lower time complexity for convergence compared to the standard SGD algorithm.

Disadvantages:
  - The updates in MB-SGD are more noisy compared to the updates in the GD algorithm.
  - Longer time is required to converge compared to the GD algorithm.
  - Susceptible to getting stuck in local minima.

### 1.3.4 SGD with Momentum
A significant drawback of the MB-SGD algorithm is the noisy updates in weight. This issue is addressed by SGD with Momentum, which mitigates the noise in gradients. The weight updates depend on noisy derivatives, and by denoising these derivatives, the convergence time can be reduced.

The concept involves denoising the derivative through exponential weighting averages, assigning more weight to recent updates compared to previous ones. This approach accelerates convergence in the relevant direction while minimizing fluctuations in irrelevant directions. An additional hyperparameter, referred to as momentum and denoted by 'γ', is introduced in this method.

V(t) = γ.V(t−1) + α.∂(J(θ))/∂θ

Now, the weights are updated by θ = θ − V(t).

The typical choice for the momentum term, γ, is around 0.9 or a similar value.

Momentum at time 't' is calculated by considering all previous updates, assigning more importance to recent updates compared to previous ones. This strategy enhances the convergence speed.

In essence, when employing momentum, it's akin to rolling a ball down a hill. The ball accumulates momentum, gaining speed as it descends (until it reaches a terminal velocity if there is air resistance, i.e., when γ<1). A similar principle applies to our parameter updates: the momentum term increases for dimensions with gradients pointing in the same direction, reducing updates for dimensions with changing gradient directions. Consequently, this results in faster convergence and reduced oscillation.

The diagram depicted above demonstrates that SGD with momentum effectively denoises gradients, leading to faster convergence when compared to standard SGD.

Advantages:
  - Retains all the advantages of the SGD algorithm.
  - Achieves faster convergence than the GD algorithm.

Disadvantage:
  - Requires the computation of an additional variable for each update.

### 1.3.5 Nesterov Accelerated Gradient (NAG)
The concept behind the NAG algorithm is quite similar to SGD with momentum, with a subtle variation. In SGD with momentum, both momentum and gradient are computed based on the previously updated weights.

While momentum is a beneficial method, excessive momentum might cause the algorithm to overlook local minima and continue ascending. To address this issue, the NAG algorithm was introduced as a lookahead approach. By using γ.V(t−1) to adjust the weights, θ−γV(t−1) provides an approximate glimpse into the future location. Consequently, the algorithm calculates the cost based on this anticipated future parameter rather than the current one.

V(t) = γ.V(t−1) + α. ∂(J(θ − γV(t−1)))/∂θ

and then update the parameters using θ = θ − V(t).

Once again, the momentum term γ is typically set to a value around 0.9. While Momentum initially computes the current gradient (small brown vector in the Image below) and then takes a substantial step in the direction of the updated accumulated gradient (big brown vector), NAG follows a different sequence. NAG first takes a significant step in the direction of the previously accumulated gradient (green vector), evaluates the gradient, and then introduces a correction (red vector), culminating in the complete NAG update (red vector). This forward-looking update serves as a preventive measure to avoid excessive speed and enhances responsiveness, contributing significantly to the improved performance of RNNs across various tasks.

Both the NAG and SGD with momentum algorithms perform comparably well and exhibit the same set of advantages and disadvantages.

### 1.3.6 Adaptive Gradient Descent(AdaGrad)
In contrast to the previously discussed algorithms where the learning rate remains constant, AdaGrad introduces the concept of an adaptive learning rate for each weight. This approach involves making smaller updates for parameters linked to frequently occurring features and larger updates for parameters associated with infrequently occurring features.

For conciseness, the notation used includes gt to represent the gradient at time step t. gt,i as the partial derivative of the objective function w.r.t. with respect to the parameter θi at time step t, η as the learning rate, and ∇θ as the partial derivative of the loss function J(θi).

In its update rule, Adagrad adjusts the general learning rate η at each time step t for each parameter θi, taking into account the historical gradients for θi:

where Gt is the sum of the squares of the past gradients w.r.t to all parameters θ.

The advantage of AdaGrad lies in its ability to eliminate the necessity for manual tuning of the learning rate, with many practitioners opting for a default value of 0.01.

However, a notable weakness of AdaGrad is the accumulation of squared gradients (Gt) in the denominator. Due to the positivity of each added term, the accumulated sum continues to grow during training. This causes the learning rate to progressively shrink, eventually becoming exceedingly small and resulting in the vanishing gradient problem.

Advantage:
  - Adaptive adjustment of the learning rate with iterations eliminates the need for manual updates.

Disadvantage:
  - As the number of iterations increases, the learning rate decreases to an extremely small value, leading to slow convergence.

### 1.3.7 AdaDelta
The drawback with the previous algorithm, AdaGrad, was the diminishing learning rate with a high number of iterations, leading to sluggish convergence. To address this, the AdaDelta algorithm introduces the concept of taking an exponentially decaying average.

AdaDelta represents a more robust extension of Adagrad, adapting learning rates based on a moving window of gradient updates rather than accumulating all past gradients. This approach enables AdaDelta to continue learning effectively even after numerous updates. In the original version of AdaDelta, there is no need to set an initial learning rate.

Instead of inefficiently storing the previous w squared gradients, the sum of gradients is recursively defined as a decaying average of all past squared gradients. The running average  at time step t is dependent only on the previous average and the current gradient:

With AdaDelta, there is no necessity to specify a default learning rate, as it has been removed from the update rule.


### 1.3.8 RMSprop
RMSprop is essentially identical to the initial update vector of Adadelta that was derived earlier.

RMSprop also involves dividing the learning rate by an exponentially decaying average of squared gradients. Geoffrey Hinton recommends setting γ to 0.9, and a commonly suggested default value for the learning rate η is 0.001.

Both RMSprop and Adadelta were developed independently around the same time, driven by the shared objective of addressing the issue of Adagrad's rapidly diminishing learning rates.

### 1.3.9 Adaptive Moment Estimation (Adam)
Adam can be perceived as a fusion of RMSprop and Stochastic Gradient Descent with momentum.

Adam calculates adaptive learning rates for each parameter. In addition to maintaining an exponentially decaying average of past squared gradients vt like Adadelta and RMSprop, Adam also retains an exponentially decaying average of past gradients mt, similar to the concept of momentum. While momentum resembles a ball rolling down a slope, Adam behaves akin to a heavy ball with friction, displaying a preference for flat minima in the error surface.

The hyperparameters β1 and β2, both belonging to the range [0, 1), govern the exponential decay rates of these moving averages. The computation of the decaying averages of past gradients mt and past squared gradients vt is performed as outlined below:

The terms mt and vt represent estimations of the first moment (the mean) and the second moment (the uncentered variance) of the gradients, giving rise to the name of the method.

## 1.4 When to choose this algorithm?

As evident from the training cost, Adam consistently achieves the lowest values.

Now, considering the observations from the animation at the beginning of this article:
  - SGD (red) appears to be stuck at a saddle point, indicating its limited applicability to shallow networks.
  - All algorithms, excluding SGD, eventually converge. AdaDelta is the fastest, followed by momentum algorithms.
  - AdaGrad and AdaDelta algorithms are suitable for sparse data.
  - Momentum and NAG perform well across various cases but are relatively slower.
  - Although the animation for Adam is not available, the plot suggests that it is the fastest algorithm to converge to the minima.
  - Adam is regarded as the most effective algorithm among those discussed above.
