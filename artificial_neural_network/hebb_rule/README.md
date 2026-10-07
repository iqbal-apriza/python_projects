# Hebb Rule Algorithm
Hebbian Learning Rule, also known as Hebb Learning Rule, was proposed by Donald O Hebb. It is a single layer neural network, i.e. it has one input layer and one output layer. Hebbian rule works by updating the weights between neurons in the neural network for each training sample. (Source: https://www.geeksforgeeks.org/machine-learning/hebbian-learning-rule-with-implementation-of-and-gate/)

## Network Architecture
<p align="center">
    <img width="700" src="../assets/Hebb Rule Arch.png" alt="description" />
</p>

## Algorithm
<p align="center">
    <img width="700" src="../assets/Hebb Rule Main.png" alt="description" />
    <img width="400" src="../assets/Hebb Rule Val.png" alt="description" />
</p>

## Getting Started
To run the algorithm, run the command as shown

    $ python hebb_rule.py

The result will show

    --- Calculating for data 1 ---
    Weights	: [1 1]
    Bias	: -1

    --- Calculating for data 2 ---
    Weights	: [2 0]
    Bias	: -2

    --- Calculating for data 3 ---
    Weights	: [1 1]
    Bias	: -3

    --- Calculating for data 4 ---
    Weights	: [2 2]
    Bias	: -2

    ==== Final Result ====
    Weights	: [2 2]
    Bias	: -2

    --- Equation ---
    y = 2x0 + 2x1 + -2

    Input 1 Input 2  Target  Output
        -1      -1      -1    -1.0
        -1       1      -1    -1.0
         1      -1      -1    -1.0
         1       1       1     1.0
