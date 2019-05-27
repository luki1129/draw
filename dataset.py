#!/usr/bin/env python

""""
Dataset loader module for reference DRAW implementation.

Author: Lukasz Stalmirski
"""

import numpy as _np
import os as _os
from PIL import Image as _PIL_Image


def image_proc_desaturate( image ):
    """
    Desaturate the image.

    The function uses Rec. 709 luma component factors:  
    OUT = 0.2126 R + 0.7152 G + 0.0722 B
    """
    output_image = []
    for row in image:
        output_row = []
        for pixel in row:
            output_row.append( 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2] )
        output_image.append( output_row )
    return output_image

def image_proc_resize( image, width, height ):
    """
    Resize the image.
    """
    output_image = []
    return output_image

def image_proc_normalize( image ):
    """
    Normalize image pixel component values to 0-1 range.
    """
    output_image = []
    for row in image:
        output_row = []
        for pixel in row:
            if isinstance( pixel, float ) or isinstance( pixel, int ):
                output_row.append( pixel / 255 )
            else:
                output_row.append( [component / 255 for component in pixel] )
        output_image.append( output_row )
    return output_image


def load_mnist_dataset( directory, download=False ):
    """
    Load mnist-like dataset from the directory.

    If the directory does not exist or is empty, and ```download``` is True, basic mnist dataset
    is downloaded. Otherwise, exception is raised.

    The function returns numpy array with (k, 784) shape, where k is number of images in the dataset.
    """
    pass


def load_image_dataset( directory, image_proc=[] ):
    """
    Load dataset from directory with raw images (saved as jpeg, png or other common image format).

    If the directory is empty, exception is raised.

    The function returns numpy array with (k, w*h) shape, where k is number of images in the directory,
    w and h are dimensions of each image (width and height, respectively).

    Each image may be processed by image_proc functions.
    """
    if not _os.path.exists( directory ):
        raise RuntimeError( directory + ': no such directory' )
    if not _os.path.isdir( directory ):
        raise RuntimeError( directory + ': not a directory' )
    images = []
    image_size = (0, 0)
    for filename in _os.listdir( directory ):
        try:
            img = _PIL_Image.open( _os.path.join( directory, filename ) )
            image_size = img.size
            img = _np.array( img )
            for image_proc_func in image_proc:
                img = image_proc_func( img )
            images.append( img )
        except Exception:
            pass
    image_count = len( images )
    if image_count == 0:
        raise RuntimeError( directory + ': directory does not contain any images')
    images = _np.asarray( images )
    images.reshape( (image_count, image_size[0] * image_size[1] ) )
    return images


def load_npy_dataset( path ):
    """
    Load dataset from npy array.

    If the file does not exist, exception is raised.

    The function returns numpy array with (k, w*h) shape, where k is number of images in the dataset,
    w and h are dimensions of each image (width and height, respectively).

    Each image may be processed by image_proc functions.
    """
    return _np.load( path )
