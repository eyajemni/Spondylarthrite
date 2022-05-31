#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import os

image_folder = "C:\\Users\\Administrateur\\Documents\\GitHub\\images"
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".JPG")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()


# In[ ]:





# In[3]:


print('hello')


# In[ ]:


def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


# In[ ]:





# In[22]:


import os 
import cv2  
from PIL import Image  
  
print(os.getcwd())  
  
os.chdir("C:\\Users\\Administrateur\\Documents\\GitHub\\images")   
path = "C:\\Users\\Administrateur\\Documents\\GitHub\\images"
  
mean_height = 0
mean_width = 0
  
num_of_images = len(os.listdir('.')) 
  
for file in os.listdir('.'): 
    im = Image.open(os.path.join(path, file)) 
    width, height = im.size 
    mean_width += width 
    mean_height += height 
    
  
mean_width = int(mean_width / num_of_images) 
mean_height = int(mean_height / num_of_images) 
  
  
for file in os.listdir('.'): 
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"): 
        
        im = Image.open(os.path.join(path, file)) 
        PIL.Image.MAX_IMAGE_PIXELS = 178956960 

   
        
        width, height = im.size    
        print(width, height) 
  
        
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)  
        imResize.save( file, 'JPEG', quality = 95) 
        
        print(im.filename.split('\\')[-1], " is resized")  
  
  
def generate_video(): 
    image_folder = '.'
    video_name = 'mygeneratedvideo.avi'
    os.chdir("C:\\Users\\Administrateur\\Documents\\GitHub\\images") 
      
    images = [img for img in os.listdir(image_folder) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
     
    
    
    print(images)  
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    
    
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))  
  
    
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
      
    
    cv2.destroyAllWindows()  
    video.release()  
  
  
generate_video() 


# In[26]:


# importing libraries
import os
import cv2 
from PIL import Image 
  
# Checking the current directory path
print(os.getcwd()) 
  
# Folder which contains all the images
# from which video is to be generated
os.chdir("C:\\Users\\Administrateur\\Documents\\GitHub\\images")  
path = "C:\\Users\\Administrateur\\Documents\\GitHub\\images"
  
mean_height = 0
mean_width = 0
  

num_of_images = len(os.listdir('.'))
# print(num_of_images)
  
for file in os.listdir('.'):
    im = Image.open(os.path.join(path, file))
    PIL.Image.MAX_IMAGE_PIXELS = 178956960
    width, height = im.size
    mean_width += width
    mean_height += height
    # im.show()   # uncomment this for displaying the image
  
# Finding the mean height and width of all images.
# This is required because the video frame needs
# to be set with same width and height. Otherwise
# images not equal to that width height will not get 
# embedded into the video
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)
  
# print(mean_height)
# print(mean_width)
  
# Resizing of the images to give
# them same width and height 
for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        # opening image using PIL Image
        im = Image.open(os.path.join(path, file)) 
   
        # im.size includes the height and width of image
        width, height = im.size   
        print(width, height)
  
        # resizing 
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
        imResize.save( file, 'JPEG', quality = 95) # setting quality
        # printing each resized image name
        print(im.filename.split('\\')[-1], " is resized") 
  
  
# Video Generating function
def generate_video():
    image_folder = '.' # make sure to use your folder
    video_name = 'mygeneratedvideo.avi'
    os.chdir("C:\\Users\\Administrateur\\Documents\\GitHub\\images")
      
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
    # Array images should only consider
    # the image files ignoring others if any
    print(images) 
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated
  
  
# Calling the generate_video function
generate_video()


# In[28]:


import os
import moviepy.video.io.ImageSequenceClip
image_folder="C:\\Users\\Administrateur\\Documents\\GitHub\\images"
fps=1

image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('my_video.mp4')


# In[ ]:




