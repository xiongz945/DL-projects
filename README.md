# DL-projects
Some of the DL projects completed in CMU
## MLP.ipynb
### HYPERPARAMETERS & EPOCH ###
The learning rate is set to 1e-3. For others, it's basically a result of trial-and-error. For example, the learning rate is decreased by a factor of 0.1 every 2 epochs, the batch size is 2048 if a GPU is presented, the context size is 24.

As for the number of epochs I ran to have the best result, 8 is what I used. Usually after 6 or 7 epochs, the model's performance plateaus, based on the running loss in training time. Therefore, out of concern of outfitting and to save unnecessary costs, I stop the training after 8 epochs.

### MODEL & DATA-LOADING ###
I used a pretty deep model. Based on the training outcome, it appears that a deeper model generally gives better accuracy. Therefore, the final model I used is comprised of 10 linear layers, which all except the last one are followed by a ReLU and a BatchNorm layer. The input and output numbers are set from 512 to 2048. It's a result of balance
between model's performance and time taken to train the model.

There are two Dataset classes that are used to load training and testing datasets. The process is during the init, I pad each utterance in the dataset with the context, and create a mapping which stores the index for the correct utterance when it's flatten in the getitem stage. In the getitem stage, things are generally easier. It just reads the correct
index from the mapping array, and returns it as a tensor pair.

## ResNet.ipynb
### Model Architecture ###
The model basically follows the ResNet, building on the basis of resnet block. More specifically, the building block contains 2 to 3 convolution layers, each followed by a batch normalization. Built with this building block, the complete ResNet contains 6 such blocks of 3 kinds of dimensions. Each has 64, 128, 256 as output channel size. After the blocks, there are a global average pool and a fully connected layer of size (256, 4000).

### Hyper Parameters ###
* Learning rate: 0.15, decays at 0.85 for every epoch
* Optimizer: SGD, with momentum of 0.9, and weight decay of 5e-5
* Batch size: 200
* Epoch: 18; after 18th epoch, the validation accuracy starts to decline

### Other details ###
A difference between my implementation and the one in the original paper is that I changed the first convolution layer of 7*7 kernel to a smaller kernel (3*3), therefore keeping more information in the first layer.

### Reference ###
* He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
* Zhang, Aston, et al. "Dive into deep learning." Unpublished Draft. Retrieved 19 (2019): 2019.

## LSTM.ipynb ##
###  Model Architecture ###
The model consists of a 4-layer LSTM and 2 FC layers after that. The LSTM layer has 13 dimension input size and 512 as hidden size, and is  bidirectional with a dropout of 0.5. The first FC layer has 1024 input size and 512 as output size. The second FC layer, which is also the output layer, has 512 as input size and 42 as output size.

### Loss function ###
The loss function used is CTCloss, which is a built-in loss function provided by PyTorch. 

### Hyper-parameters ###
Optimizer used is Adam, with learning rate being 1e-3 in the beginning, and a weight decay at 5e-6. The batch size used is 64.
There is also a CTCBeamDecoder, which is used to decode the output of the model. The beam width used is 25.

A learning rate scheduler 'ReduceLROnPlateau' is used on top of Adam. The scheduler keeps an eye on Levenshtein distance of the validation, and if it's not decreasing for 2 epochs, the learning rate will be lowered by a factor of 0.1.

## Attention.ipynb ##
### Model Architecture ###
The model consists of an encoder, an attention component, and a decoder. The encoder consists of a BiLSTM, and three pBLSTM layers as specified in the 'Listen, Attend and Spell' paper. After that, there are two outputs from two linear layers, key and value, which are fed into the attention component later. The attention component is basically a weighted average of the value, where weights are computed by applying a softmax to the product of the key and query. Query is the output from the decoder.
The decoder is composed of an embedding layer, two lstm cells and eventually a linear layer to output the probabilities of different characters.

### Loss Function ###
The loss function used is CrossEntropyLoss, which is a built-in loss function provided by PyTorch. But to properly calculate the loss between labels and the outputs, I added a 'ignore_index=0‘ parameter to properly leave out the position where padding is present.

### Hyper Parameters ###
Optimizer used is Adam, with learning rate being 2e-3 in the beginning, and a scheduler that lowers the learning rate by 0.6 every 5 iterations.
The batch size used is 64.
Teacher forcing is also implmented. The initial teacher-forcing rate is 0.95. After 20 epochs, when the training loss is about to plateau, the teacher-forcing rate will be decreased by 0.01 every 3 iterations.
