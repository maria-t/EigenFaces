# EigenFaces
Testing eigenface recognition, using the images form the FERET face database.

The implementation shows as images:
* the average face
* the eigenfaces corresponding to the 10 largest eigenvalues
* the eigenfaces corresponding to the 10 smallest eigenvalues

### Requirements
OpenCV library

Compiles with: 
g++ eigenfaces.cpp `pkg-config --cflags --libs opencv`
