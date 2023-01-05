# Stereo-Matching

Stereo Matching is the process of recovering 3D structures of real world from 2D images. It has been widely used in areas such as autonomous driving, augmented reality and robotics navigation. 

Given a pair of rectified stereo images, the goal of this project is to compute the disparity for each pixel in the reference image, where disparity is defined as the horizontal displacement between a pair of corresponding pixels in the left and right images.

## Example

Given a pair of rectified images below:

**Image left:**

![Cones left image](/images/cones/im2.png)

**Image right:**

![Cones right image](/images/cones/im6.png)

The main goal is to implement algorithms that is able to estimate the disparity of the scene in 3D from the above images. The performance of the algorithms is evaluated by comparing the results to the actual disparity map.

**Actual disparity map - Using left image as reference:**

![Disparity map](/images/cones/disp2.png)

The evaluation also considers the accuracy of occluded pixel detection. Occlusion detection in Stereo Matching refers to the process of detecting which areas of the images are occlusion boundaries or areas that appear occluded in views of the scene

**Actual occlusion map:**

![Occlusion map](/images/cones/occl.png)

## Algorithms

Nowadays, there exists several ideas using Machine Learning and Deep Learning to solve this problem, see [here](https://vision.middlebury.edu/stereo/eval3/) for more information.

However, this project aims to develop and optimize traditional algorithms Block Matching and Semi Global Block Matching without using Machine Learning or Deep Learning.

### Block Matching

Local block matching uses the information in the
neighboring patch based on the window size, for identifying the conjugate point in its stereo pair

![Block Matching](/images/block-matching.png)

**Command to run Block Matching algo:**

`python ./block-matching/stereomatch.py left_image_path right_image_path`

### Semi Global Block Matching

Semi-global matching uses information from neighboring pixels in multiple directions to calculate the disparity of a pixel. Analysis in multiple directions results in a lot of computation. Instead of using the whole image, the disparity of a pixel can be calculated by considering a smaller block of pixels for ease of computation. Thus, the Semi-Global Block Matching (SGBM) algorithm uses block-based cost matching that is smoothed by path-wise information from multiple directions.

![Semi Global Block Matching](/images/SGBM.png)

`python ./semi-global-block-matching/stereomatch.py left_image_path right_image_path`
