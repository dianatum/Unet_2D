# Use an official Python runtime as a base image
FROM python:3

# Add files to the dockerfile
ADD unet2D_main.py unet2D_main.py 
ADD unet_2d.py unet_2d.py 

# Install any needed packages 
RUN pip install tensorflow
RUN pip install numpy
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install nilearn

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run when the container launches
CMD ["python", "unet2D_main.ipynb"]