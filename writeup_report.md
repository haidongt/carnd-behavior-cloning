# Network Structure


# Data Collection
My strategy for data collection is to drive continuously for several laps. I intentionally make some mistakes and then I control the vehicle to recover to the center of the lane. I have a pattern when I make mistakes so that I can easily remove the sample images where I'm making mistakes in my pre-processing step. When I make mistakes I just let go of the mouse and let the car drift to the side, so that the steering angle in this process is always zero, and I'll remove sample images with zero steering angle.

# Data Pre-processing
I have plotted a histogram of the steering angle distribution and I found that most steering angles are about zero. 


