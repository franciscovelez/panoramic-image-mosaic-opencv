panoramic-image-mosaic-opencv
=============================

Generation of panoramic image mosaics using cylindrical projection and Burt-Adelson's blending

This application has been developed as a project for the "Visión por Computador" course (Computer Vision) at the University of Granada. Course 2013/2014

How to use
----------

The program runs from the command line. It has three parameters to be introduced in the following order:
- Folder of input pictures: Path to the folder containing the pictures.
- Scaling factor: Value for the scale factor in the cylindrical warping.
- Output picture: Optional. Name of the picture to save.

Warnings:
- Pictures must be listed in the exact order in which they relate each one to the following.
- It is strongly recommended to use the parameter "Output picture". If not used, the image is displayed on the screen and, if it is too large, it will be cut

Example of use of the program: 
```
Panorama.exe c:\imgs 900 panorama.jpg
```

Results
-------
![panorama](https://raw.githubusercontent.com/franciscovelez/panoramic-image-mosaic-opencv/master/Pictures/pan_street_800.jpg "panorama")

Notes
-----

All documentation (in spanish) available on: https://github.com/franciscovelez/panoramic-image-mosaic-opencv/raw/master/Documentation%20and%20How%20to%20use.pdf

Thanks to Chaman Singh Verma and Mon-Ju for the pictures available on: http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/DataSet/

Authors
-------

* Yuri Garcés Ciemerozum
* Francisco Vélez Ocaña
