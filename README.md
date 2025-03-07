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
