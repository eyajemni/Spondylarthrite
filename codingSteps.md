# General goal 
To visualize the movement in the file 

Inspired by the strategy to display Kinect skeleton

https://www.muratkirlioglu.com/pykinect-tutorial/

# read the input file 
* the file consists of timeseries of joint position (unit?): 
    - X
    - Y 
    - Z

* each line is an image 
* what is the time delay between 2 images ? 

# structure to manipulate the data 

## joint
* Cartesian coordinates (x, y, z) of N joints 
* N = ? 
* visualization as a circle or dot 

## segment 
A link between 2 joints: 

* starting joint
* ending joint
* visualization as a line 

# List of development steps 

* fist version as a single python file using standard procedural programming (no object oriented programming)

* what libraries are necessary ? 
    - numpy
    - matplotlib 
    - panda (only for reading the file?)

* define file name 
* read the file into a big array 
* split the big array into joints 
* define the segments (this does not depend on the file: this is given by Xsens)
* for each timestamp 
    * plot all joints 
    * plot all links 
* make an animation to show the movement
