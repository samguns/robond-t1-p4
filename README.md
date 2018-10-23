## Deep Learning Project: Semantic Segmentation

---

[//]: # (Image References)

[image1]: ./images/network.png
[image2]: ./images/training_curve.png


### Explains the Neural Network.
#### 1. Describes techniques and concepts that are used in this project.
##### 1x1 Convolution
Most often, a `1x1 convolution` is used to reduce dimensionality in feature space, hence reducing computational costs. In essence, it's a linear matrix multiplication, which functions like a fully connected layer. But since it's a convolutional layer that gets spatial information preserved. This makes it an ideal bridge in connecting `encoder` to `decoder`.

##### Separable Convolutional layer
In general, a Separable Convolutional layer is a normal 2D convolution followed by a `1x1 convolution`. It extract feature maps with a reduced parameters, which is more efficient during training.

##### Bilinear Upsampling
The idea of Bilinear Upsampling is to calculate new pixel's intensity from its four nearest known neighbor pixels that located diagonally. Because of it just up-scales image from interpolation, it's imprecise and there's no learnable parameters in this layer.


#### 2. Describes the network architecture
##### Network architecture
The fully convolutional network for this project consists of an `encoder`, followed by a `decoder`. The `encoder` portion extracts features of input images. While the convolutional layer within `encoder` goes deeper, we get a higher dimensional feature vectors that define the input well in pixel-wise level. On the opposite side, the `decoder` construct a semantic segmentation mask image back to the original input size from the feature vectors.
![alt text][image1]

##### Encoder
The `encoder` in this project has three Separable convolutional layers. The filter depth goes from 32 to 256, and each layer down-samples its input in a scale of 2.

At the end of `encoder`, feature vectors are sent to the `decoder` through a `1x1 convolution`. So the `decoder` gets features as well as spatial information without bringing in expensive computational costs.


##### Decoder
Compare to the `encoder` portion, the `decoder` is a bit complicated. It has three blocks. In each block, a `Bilinear Upsample layer` does a 2x up-scale to the input, then concatenates the result with features that generated in encoding phase. This is also known as `skip connection`. Finally, I put two Separable convolutional layers that have the same filter depth to extract features from the concatenated vectors.

#### 3. Explains how hyper-parameters are selected.
##### Learning rate
The model uses Adam optimizer. From my previous limited experiences, the default 0.001 learning rate usually works well. So I didn't attempt to tune it.

##### Batch size
I used p2.xlarge AWS instance for the training and tried various batch sizes starting from 32. I ended up with 64, which is large enough and wouldn't exhaust all GPU memory.

##### Steps per Epoch
I first tried the 200 steps but he training process was too slow. Then I lowered it to be 100 and found out it didn't hurt much in accuracy, but with a relatively shorter training process.

##### Epochs
I first set a 20-Epoch training. The result was around 30%. From to the plots, I could see a decreasing trend both in training and validation costs. I decided to go drastically. I set it to be 100 and registered a check-point saving callback in the training iteration. The callback function saves weight by the end of every epoch. In the end, my model got a final score of about 42% at Epoch 38 with the provided training datasets.

### Reflections.
#### 1. Can this model and data work well for following another objects (dog, cat, car, etc) instead of a human?
This model and trained weight won't work for following objects other than a human. In our training datasets, there are only three classes, they are i.) background, ii.) our hero to follow, iii.) other humans. Besides, as shown in the visualized predictions and IOU metrics for our hero located in a distance away, this model doesn't perform well. Meaning it's hard to recognize relatively smaller figures. So even if we have the object defined in training set, the model will require further improvements in recognizing small objects.

#### 2. Future Enhancements
* From the plotted graphs shown below, I can see the model suffers overfitting from about 40th epoch, because training loss keeps decreasing but validation loss increases. To counteract this, collecting more data seems promising and I'd like to try out later.

![alt text][image2]

* The network is a bit shallower compared to those state-of-art models in the wild. I think it might be we need to make a trade off for efficiency. If I were to pursue this project later, making the network deeper is an intriguing approaching I'd like to try out.
