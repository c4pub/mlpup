# Multi-Layer Perceptron Universal Predictor
A simple, one-file implementation of a Multi-Layer Perceptron (MLP).

## Overview
This is a basic implementation of a Multi-Layer Perceptron. It's not optimized for performance.

## Features
* One-file, self-contained implementation
* Contains a theoretical derivation of the backpropagation algorithm using calculus
* No external libraries are used for core functionality (NumPy and others are used for input/output processing)
* Compatible with classifier usage as in scikit-learn
* Accepts a mix of continuous and categorical data
* Several activation and loss functions are provided, with the ability to add new ones
* Can also do basic regression

## Structure
The code is divided into two main parts:
1. A comment section describing the theoretical derivation of backpropagation using general calculus.
2. The implementation of the theoretical description, using several classes for encapsulation purposes. The `NeuNet` class contains the actual implementation of the theoretical concepts, with cues provided by `<theory-ref>` keyword comments.

## Usage
This implementation can be used to experiment with MLP and backpropagation.

## Theory
From comment section:

---
```

    (*) Theoretical background on back-propagation and weights update.

        Gradient descent: minimizing a loss function f_loss(x) by varying x so
        that it descends in small steps towards the minimum. The theory says
        that the most efficient update is:

        x_new = x_crt - rate_step * (d(f_loss(x))/d(x))(x_crt)

        (*) where:

            rate_step = learn_rate

        (*) The term "gradient" refers to the derivatives of the loss function
            with respect to each of the weight components, treating these
            weights as a vector:

                    d(f_loss(x))/d(x) <=> grad(f_loss(x))

    (*) In back-propagation:

        w_new = w_crt - learn_rate * (d(f_loss(w))/d(w))(w_crt)

        (*) The difference between the predictive and the expected output
            represents the residual.

                residual = y_out - y_expect
                residual(w, in) = f_out(w, in) - f_expect(in)

                    f_out() - also referred to as activation

        (*) Definitions:

            w - weights,
            in - inputs,
            f_out() - output function
            f_expect() - expected output function
            f_deriv_loss = d(f_loss)/d(f_out)

        (*) Example of loss function: Euclidean distance

                f_loss(y_out) = (1/2)*((y_out - y_expect)^2)
                d(f_loss)/d(x) = x - y_expect
                f_deriv_loss(x) = d(f_loss)/d(x)

    (*) A neuron (also known as a "node") processes an input, represented
        by a vector of numbers, into an output signal by means of "weights"
        and an "activation function".

                   w_1
            in_1 ---*---+
                        |
                   w_2  |
            in_2 ---*---+
                        |    |-------\       |-----------\
                        +--->|  Sum   |---> |  f_activ()  |---> output
                        |    |-------/       |-----------/
               ...      |
                        |
                   w_n  |
            in_n ---*---+

        For a neuron with multiple inputs:

            f_preact() = Sum[k](w_k * in_k)
            f_out() = f_activ(f_preact())
            f_deriv_act() = d(f_activ(z))/d(z)

            where:
                in_k        - is the input k
                w_k         - is the weight of input k
                f_preact    - is the preactivation of the neuron
                f_out       - is the output of the neuron

            (*) Examples of activation functions:

                (*) sigmoid

                sigmoid(x) = 1/(1 + e^(-x))

                f_activ(z) = sigmoid(z)
                f_deriv_act(z) = (d/d_z)(sigmoid)(z)
                               = sigmoid(z)(1 - sigmoid(z))
                               = f_activ(z)(1 - f_activ(z))

                (*) tanh

                tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

                f_activ(z) = tanh(z)
                f_deriv_act(z) = (d/d_z)(tanh)(z)
                               = 1 - (tanh(z))^2
                               = 1 - (f_activ(z))^2

            f_out(w, in) = f_activ(f_preact(w, in))

    (*) The topology of a multi-layer perceptron (MLP) consists of a
        sequence of neuron layers. The last layer (L), also referred to as the
        "output layer", provides the predictive output activation that is
        compared to the expected output. The difference between the predictive
        and the expected output represents the residual.

        A simple example of an MLP topology is provided in the following
        diagram.

             IN_A --->----- (C) --->-----\
                 \          /  \          (E) ->- \
                  \   /->--/    \   /->--/         \
                   \ /           \ /                \
                    \             \                  (G) -> OUTPUT
                   / \           / \                /
                  /   \->--\    /   \->--\         /
                 /          \  /          (F) ->- /
             IN_B --->----- (D) --->-----/

        Here, the output layer, or the last layer, consists of
        one neuron: G. It is preceded by a hidden layer consisting of neurons
        E and F. This is preceded by another layer formed by neurons C and D.
        Layer "L" (last) consists of neuron "G."

        In order to simplify, the present explanation will consider a
        simple topology with only one neuron per layer.

            IN ->- ... ->- (p) ->- (q) ->- (r) ->- ... ->- (L) -> OUTPUT

        The L node is the last node providing the output.

        In order to compute the gradient of each node's weight, a helper
        function is computed for each node/neuron. This is referred to
        as "delta," or alternatively, the "error term" of a neuron.

            delta() = d(f_loss)/d(f_preact)

        It represents the variation of the loss function with respect to
        the preactivation of the neuron.

            f_loss() - loss function of the last layer neurons, a function
                        of the neuron output

    (*) The delta function on a generic layer neuron (q):

        delta_q() = d(f_loss)/d(f_preact_q)

        (*) layer (q) is followed by layer (r) and preceded by (p):

        delta_q = (d(f_loss)/d(f_preact_r))*(d(f_preact_r)/d(f_preact_q))

            delta_r = d(f_loss)/d(f_preact_r)

        delta_q = (delta_r(f_preact_r))*(d(f_preact_r)/d(f_preact_q))

            d(f_preact_r)/d(f_preact_q) =
                    = (d(f_preact_r)/d(f_out_q))*(d(f_out_q)/d(f_preact_q))

                f_preact_r = w_r * in_r

                    (*) note that input to r is the output from q
                        in_r = f_out_q

                f_preact_r = w_r * f_out_q

                d(f_preact_r)/d(f_out_q) = d(w_r * f_out_q)/d(f_out_q)
                d(f_preact_r)/d(f_out_q) = w_r

            d(f_preact_r)/d(f_preact_q) = (w_r)*(d(f_out_q)/d(f_preact_q))

                (*) notation:
                    f_deriv_act = d(f_out)/d(f_preact)

            d(f_preact_r)/d(f_preact_q) = (w_r)*(f_deriv_act(f_preact_q))

        delta_q = delta_r * w_r * f_deriv_act
        delta_q(f_preact_q) = delta_r(f_preact_r) *w_r* f_deriv_act(f_preact_q)

    (*) The delta function on the last layer neuron:

        delta_L() = d(f_loss)/d(f_preact_L)
            = (d(f_loss)/d(f_out_L))*(d(f_out_L)/d(f_preact_L))
            = (d(f_loss)/d(f_out_L))*(f_deriv_act(f_preact_L))

                d(f_loss)/d(f_out) = f_deriv_loss(f_out)

        delta_L() = d(f_loss)/d(f_preact_L)
            = (d(f_loss)/d(f_out_L))*(f_deriv_act(f_preact_L))
            = (f_deriv_loss(f_out))*(f_deriv_act(f_preact_L))

        delta_L() = (f_deriv_loss(f_out()))*(f_deriv_act(f_preact_L))

    (*) The gradient of the loss function can be expressed as:

        d(f_loss)/d(w_q) = (d(f_loss)/d(f_preact_q))*(d(f_preact_q)/d(w_q))
        d(f_loss)/d(w_q) = (delta_q(f_preact_q))*(d(f_preact_q)/d(w_q))

            f_preact_q(w_q, in_q) = w_q * in_q
            d(f_preact_q)/d(w_q) = in_q

        d(f_loss)/d(w_q) = delta_q(f_preact_q) * in_q

    (*) Using the previous expressions, the gradient descent update of the
        weights can be computed as:

        w_k_new = w_k_crt - learn_rate * (d(f_loss)/d(w_k))(w_k_crt)

    (*) Note that the effect of the neurons' biases has not been detailed,
        but they do operate as regular weights with a constant input of one.

```
---
