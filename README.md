# NeuralNetwork-Java-v2
My second version of my own neural network, written in Java.<br>

Testing the neural network in a real project
---
The neural network is used in my project SignDetector.
<br>

Some sample results ([Test.java](NeuralNetwork/Testing/Test.java))
---
The Test file creates a new neural network with to input neurons, three hidden neurons and one output neuron.
It also prints this information out:
```
A neural network with 3 layer(s) including a bias.
Using activation function: Sigmoid function.
Network state: ready.
```
Afterwards the network is getting trained towards two DataSets:
```
1. inputVector = {0.5,  0.4} --> outputVector{0.9}
2. inputVector = {0.4, -0.4} --> outputVector{0.0}
```
