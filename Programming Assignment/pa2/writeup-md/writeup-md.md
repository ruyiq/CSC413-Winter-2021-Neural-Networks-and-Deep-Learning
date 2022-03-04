##  Programming Assignment 2
  
###  Part A: Pooling and Upsampling
  
####  A.1 Implementation of PoolUpsampleNet
  
<img src="PoolUpsampleNet.png" alt="PoolUpsampleNet" width="600"/>
  
####  A.2 Training Result
  
![training result](1a1a.png "Title")
![training result](1a1b.png "Title")
The shown figure is the result obtained from the main training loop of PoolUpsampleNet. It does not look good to me as the images are quite blurry with greyscale pixels present. Also, after 25 epoch, accuracy is around 41.4%, which is quite low.
  
####  A.3
  
Assume the kernel size is $k$
- when each input dimension (width/height) is not doubled (original input)
  - number of weights = $k^2 (NIC\cdot NF+ NF^2 + 2NF^2 + NF^2 + NC \cdot NF)$= $ k^2 (NIC\cdot NF+ 4NF^2 + NC \cdot NF)$
  - number of outputs = $2880NF + 3328NC$
  - number of connections = $k^2 (1024 \cdot NIC \cdot NF + 640 NF^2 +256NF \cdot NC)$
- when each input dimension (width/height) is doubled
  - number of weights = $k^2 (NIC\cdot NF+ NF^2 + 2NF^2 + NF^2 + NC \cdot NF)$= $ k^2 (NIC\cdot NF+ 4NF^2 + NC \cdot NF)$
  - number of outputs = $11520NF + 13312NC$
  - number of connections = $4k^2 (1024 \cdot NIC \cdot NF + 640 NF^2 +256NF \cdot NC)$
  
  
###  Part B: Strided and Transposed Dilated Convolutions
  
####  B.1 Implementation of ConvTransposeNet

<img src="ConvTransposeNet.png" alt="ConvTransposeNet" width="700"/>

  
####  B.2
  
![training result](1b2a.png "Title")
![training result](1b2b.png "Title")
####  B.3 
 The trained result from ConvTransposeNet model seems better from what we got from PoolUpsampleNet because 
 - the image seems less blurry 
 - the validation accuracy increases from 41.4% to 54.7%. 
 - the validation loss decreases from 1.5848 to 1.1565. 
 
 We know that Max-pooling can reduce the height and width of the output (i.e. reduce the input dimensionality) so that there will be less parameters to learn from for the next layer. Therefore, by removing MaxPool2d and Upsample from the model, ConvTransposeNet allows more features to be learnt within the system than what's of PoolUpsampleNet. Therefore, ConvTransposeNet yields a better result.
####  B.4
- if kernel size = 4,
  - padding parameter passed to the first two nn.Conv2d layers = 1
  - padding parameter passed to the nn.ConvTranspose2d layers= 1
  - output_padding parameter passed to the nn.ConvTranspose2d layers = 0
- if kernel size = 5,
  - padding parameter passed to the first two nn.Conv2d layers = 2
  - padding parameter passed to the nn.ConvTranspose2d layers = 2
  - output_padding parameter passed to the nn.ConvTranspose2d layers = 1
####  B.5
When fixing the number of epochs, increase in batch sizes will result in a increase in validation loss and a worse final image output quality. Vice versa.
###  Part C: Skip Connections
  
####  C.1 Implementation of UNet
  
<img src="UNet.png" alt="UNet" width="600"/>

####  C.2
  
![training result](1c2a.png "Title")
![training result](1c2b.png "Title")

####  C.3
First of all, comparing the result obatined from UNet to the results from Part A and Part B, we see that:
- the image obtained from skip connections seems to be the least blurry most among all three
- the validation loss from UNet is 1.0512 after 25 epoch, which is smaller than 1.5848 (from Part A) and 1.1565 (from Part B).
- the validation accuracy is 58.9% which is greater than 41.4% (from Part A) and 54.7% (from Part B).

Therefore, we can conclude that skip connections does improve the validation loss and accuracy.

Two reasons why skip connections might improve the performance of our CNN models are:
- There are more learning parameters in skip connection models, which results in a better performance. For example, output of first layer and third layer are both the source of input for the fourth layer. But for the previous models, they only have output from the third layer as the source for the fourth layer.
-  Skip connection concatenates the layers. By doing so, the features can be resused multiple time and thus a better result.
###  Part D: Object Detection
####  D.1 Fine-tuning from pre-trained models for object detection
<img src="129.png" alt="d1" width="600"/>

####  D.2 Implement the classification loss

##### D.2.1
<img src="97.png" alt="d1" width="600"/>
<img src="150.png" alt="d1" width="600"/>

##### D.2.2
<img src="d1.jfif" alt="d1" width="400"/>
<img src="d2.jfif" alt="d2" width="400"/>
