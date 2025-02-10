# HAHCP


Authors: Li X, Mo D, Deng Shanjing, Jiang XY

<!-- [[Paper Link]] -->





<!-- #### If you find the resource useful, please cite the following :- ) -->

<!-- ```
@article{
```   -->

### Prerequisites
- Matlab R2021a or above.


### Data Set


- O-Haze (45 pics)

 Images are available in https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/
  
### HAHCP

#### The HAHCP algorithm is based on Dr. He's DCP. （Kaiming He, Jian Sun and Xiaoou Tang, "Single image haze removal using dark channel prior," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 2009, pp. 1956-1963, doi: 10.1109/CVPR.2009.5206515.）

 For the related theory of the algorithm, please refer to: https://ieeexplore.ieee.org/document/5206515.

#### To make the operation more intuitive, we have divided the algorithm into two separate parts `hist.m` and `HAHCP.m`, but they can also be integrated into a single program. 


- `hist.m` calculates the image histogram and records the dominant gray level and its corresponding pixel count. 
 
- `HAHCP.m` implements the main algorithmic computations.

#### The flowchart of the algorithm and the specific steps are shown in the figure below.

![image](https://github.com/21ShH/HAHCP/blob/main/FLOWCHART.png)

![image](https://github.com/21ShH/HAHCP/blob/main/Algorithm%20Procedure.png)


