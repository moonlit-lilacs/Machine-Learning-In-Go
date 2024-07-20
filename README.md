# Machine Learning In Go
A re-implementation of my initial "Machine Learning In C" project, now using Go. Since re-implementing the original, I've switched from the initial Sigmoid function to a Leaky ReLU, added gradient snipping and added normalization values. In the future I'd like to make some adjustments so it can eventually handle the MNIST dataset of handwritten digits. It's *possible* it could handle it now, but I haven't attempted it since even training it to add two numbers up to 20 is quite slow and prone to idling at some values for long periods of time.


As this is based off of the original project I mentioned, credit goes to Tsoding for the base in C which I have developed off here. 