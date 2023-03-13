# Blender_Neural_Network
Create an Artificial Neural Network Model in Blender

![MNIST](https://github.com/barckley75/Blender_Neural_Network/blob/f45a0d29556761b2a48d50d5a4f72baeabc4de8e/files/Image.GIF)

[Blender 2.93.1](https://www.blender.org](https://download.blender.org/release/Blender2.93/), [Animation Nodes 2.2.2](https://blender.community/c/graphicall/0hbbbc) (this version still in development).

Blender Neural Network is an addon for Blender that creates a panel in the 3D view. With this panel you can create the model of ANN.
Some examples:

Input Size = 15, Hidden Layer  = 10, Output = 2
![example_1](https://github.com/barckley75/Blender_Neural_Network/blob/f3a5fdac246f53f01116a1ac503aba048343dbd2/ANN_3.jpg)

Input Size = 784, Hidden Layer  = 100, Output = 10
![example_2](https://github.com/barckley75/Blender_Neural_Network/blob/f3a5fdac246f53f01116a1ac503aba048343dbd2/ANN_2.jpg)

Input Size = 784, Hidden Layer  = 100, Output = 9
![example_3](https://github.com/barckley75/Blender_Neural_Network/blob/f3a5fdac246f53f01116a1ac503aba048343dbd2/ANN_1.jpg)

<h2>The Panel has four sections</h2>

1. **<h3>Neuron Aspect</h3>**
   1. **Neurons Aspect Mesh** change the mesh of the neurons: 0 > Iconesphere, 1 > Sphere, 2 > Text, 3 > Square
   2. **Neuron Model Size** change the scale of the neurons.
   3. **Connections Visibility** Hide the connections
   4. **Connecitons Radius** change the thickness of the connections. 
2. **<h3>Neuron Model Size</h3>**
   1. Set the number of the neurons for each layer.
3. **<h3>Neuron Model Shape</h3>**
   1. Here it is possible to change the 'SHAPE' of the neurons (it is not connected with Numpy, it is just for aesthetic purpose). For instance, MNIST handwritten is a dataset of image 28 x 28, if you are reading an array with one dimension, you will have 28x28 = 784 pixels, and this case the size will be 784 and the shape 28.
4. **<h3>Training Data</h3>**
   1. this section is for training the model and still in development. At the moment you can add the path of the dataset and it will be read directly in Animation Nodes. In this case the the size input of the dataset will be the length of the columns. In the tree you will find this node, Training Model, here converge all the variables you need to update your model. This is the variables:

* Variables
  * dataset_path
  * inputs
  * hiddenLayer_1
  * output
  * epochs
  * minibatch
  * eta
   
   To import, for instance, Tensorflow in Blender, I suggest to install it inside the Blender site-packages. [Here](https://stackoverflow.com/questions/65076829/how-to-use-tensorflow-in-blender/68335409#68335409) how to do it.
   In this case, as you can read in the node script, we are reading training_data file from Text Area in Blender, in this file you have direct access to the viriables declared in the node. So, for example, you can pass the hidden neurons in this way: 
   >keras.Input(shape=(hiddenLayer_1,))
  Where hiddenLayer_1 is the variable red from the panel. 
   
   10. ![Data](https://github.com/barckley75/Blender_Neural_Network/blob/d7ec4d06196605717da7b04d550751890a0f32fc/AN_Script_training_model.png)

<h1>>The Panel</h1>
This is the aspect of the panel, you can pretty much control all the important visualization aspects. The panel still missing the possibility to control the distance between the layers. If you want to contribute, please, you are welcome.


![Panel](https://github.com/barckley75/Blender_Neural_Network/blob/1baf0de336c445f8e7f8b610caf3ad9039fe85b4/panel.png)
