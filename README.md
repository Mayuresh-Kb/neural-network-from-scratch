# Neural Network From Scratch (NumPy)

A 3-layer neural network implemented from first principles using only NumPy — no TensorFlow, no PyTorch, no autograd. Forward propagation, backpropagation, and gradient descent are all derived and coded by hand.

This started as an exercise after finishing Andrew Ng's Machine Learning Specialization: I wanted to make sure I could actually implement what I'd learned, not just follow along with the course.

## Why two notebooks

The same network is built twice, in two different styles, on purpose.

**1. `backprop-from-scratch.ipynb`** — loop-based implementation. Every neuron and every weight update is computed with explicit `for` loops. This is slower and more verbose, but it makes the mechanics of backpropagation impossible to hand-wave past — you can see exactly which weight connects to which neuron at every step.

**2. `vectorized-backprop-numpy.ipynb`** — the same architecture and the same math, rewritten using matrix operations (`np.matmul`) instead of loops, with a training loop added on top. This is closer to how production code is actually written.

Building it the slow way first, then the fast way second, made the vectorized version genuinely make sense rather than feeling like syntax to memorize.

## Architecture

```
Input (2) → Dense (3, sigmoid) → Dense (2, sigmoid) → Dense (1, sigmoid)
```

A simple feedforward network with sigmoid activations throughout, trained with vanilla gradient descent on squared error loss.

## What's implemented

- Forward propagation through dense layers
- Backpropagation via the chain rule, derived layer by layer
- Gradient descent weight updates
- A training loop (vectorized notebook) tracking prediction and cost across epochs
