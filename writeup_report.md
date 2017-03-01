# Network Structure
The first layer is a 5 by 5 convolutional layer, followed by a max pooling layer. The next layer is also a 5 by 5 convolutional layer followed by a max pooling layer. After that, the output is flattened. The flattened output is then going though a dropout layer followed by a 84 node dense layer, and then a dropout layer followed by a 84 node dense layer, and then a dropout layer followed by a one node dense layer.

# Data Collection
I first drive continuously for several laps to get my dataset a baseline of all situations. I then intentionally make some minor mistakes, i.e. driving car to side of road and then drive back to center. I only record the part where I'm driving towards center. I now use the trained model to do autonomous mode to see how well my model is performing and identify any weak locations. I then focus on these weak locations and drive in these weak locations to add more samples to enhance my model's weakness.

# To prevent overfitting
I used several different strategies to prevent overfitting. The first is that I collected a lot of data. Before data preprocessing, my data set contains roughly 30,000 samples. I use a 0.2 validation split to train the model. I also use two dropout layers to prevent overfitting.


