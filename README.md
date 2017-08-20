# Digit_recognizer

# Example implementation of Neural Network for Digit recognition

Digit recognizer implementation with different activation functions (sigmoid, ReLU, LeakyReLU and tanh) and parameter update techniques (Vanilla update, adagrad and momentum parameter update techniques). This implementation do not use any libraries for bacward and forward progration implemenation, other than using Linear Algebra subroutines for Matrix vector multiplication. This is an initial implemenation, so there might bugs! Any comments or suggestions are welcome!

# Steps to use design files:

step1: clone the git repository

step2: lanuch python intrepretter and then execute the below commands 

        import mnist_loader
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
 
step3: exit out of the python intrepretter 

step4: To run different network implemenation follow the below syntax

       python <script_name.py>
       
       example:
       
       python digit_recog.py
       
           
# Note: 
### mnist_loader.py script is borrowed from the below tutorial http://neuralnetworksanddeeplearning.com/chap1.html
### Implementation is done using python version Python 2.7.10 
### Make sure the dependencies, you might get errors in case some of the required packages are not installed      
    
    
    
    
    
License

MIT License

Copyright (c) 2017-2018 Hanumantharayappa, University of Tennessee Knoxville (UTK), EECS department

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


