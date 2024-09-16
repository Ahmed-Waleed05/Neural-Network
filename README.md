# Neural-Network

for the NN file to work first extract the **archive.zip** folder then copy the path of the **mnist_train.csv** file and 
replace the **TRAIN PATH HERE** do the same with **mnist_test.csv** and **TEST PATH HERE**







a neural network is a way to simulate the the human thought processes and it consists of somthing called neurons and the more layers if neurons there is the more complex the network is.

the weight of the input means how important the input is to the output as in how much it changes it.

the threshold is a parameter of the neuron and it represents how easily the neruon will output a 1 and it is the same as Bais

for a neuron to fire(output) the fowlloing equation must be fullfied (input1 x weight1) + (input2 x weight2) > threshold(bais)

 Perceptrons output={ 0 if w⋅x+b≤0 }
                    { 1 if w⋅x+b>0 }   where x is the input w is the weight and b is the bais


we can devise learning algorithms which can automatically tune the weights and biases of a network of artificial neurons. This tuning happens in response to external stimuli, without direct intervention by a programmer

Perceptrons their output value can either be 0 or 1 while Sigmoid neurons their value can be anthing between 0 and 1 

Sigmoid neurons output σ(z)≡1/1+e^−z

As mentioned earlier, the leftmost layer in this network is called the input layer, and the neurons within the layer are called input neurons. The rightmost or output layer contains the output neurons, or, as in this case, a single output neuron. The middle layer is called a hidden layer

While the design of the input and output layers of a neural network is often straightforward, there can be quite an art to the design of the hidden layers. In particular, it's not possible to sum up the design process for the hidden layers with a few simple rules of thumb. Instead, neural networks researchers have developed many design heuristics for the hidden layers, which help people get the behaviour they want out of their nets. For example, such heuristics can be used to help determine how to trade off the number of hidden layers against the time required to train the network

we use 10 neurons for the output instead of just 4 because for this problem If we had 4
outputs, then the first output neuron would be trying to decide what the most significant bit of the digit was. And there's no easy way to relate that most significant bit to simple shapes like those shown above. It's hard to imagine that there's any good historical reason the component shapes of the digit will be closely related to (say) the most significant bit in the output.

the tenser implmentaion is much easier to do and has good prefromance and has some flexibility while the NN from scratch was alot more difficult and the preformance varries alot with the user and how they do it but it has alot more flexibility as you do everything as you like and you can control every bit of the network
