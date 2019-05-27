#!/usr/bin/env python

""""
Dataset loader module for reference DRAW implementation.

Author: Lukasz Stalmirski
"""

import numpy as _np
import os as _os
import urllib as _urllib
from shutil import copy as _copy
from tensorflow.python.platform import gfile
#from PIL import Image as _PIL_Image
import gzip

DEFAULT_MNIST_URL='https://storage.googleapis.com/cvdf-datasets/mnist/'


def image_proc_desaturate( image ):
    """
    Desaturate the image.

    The function uses Rec. 709 luma component factors:  
    OUT = 0.2989 R + 0.587 G + 0.114 B
    """
    output_image = []
    for row in image:
        output_row = []
        for pixel in row:
            output_row.append( 0.2989 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2] )
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


def _read32( bytestream ):
    """
    Read next 32-bit unsigned integer value from the byte stream.
    """
    dt = _np.dtype( _np.uint32 ).newbyteorder( '>' )
    return _np.frombuffer( bytestream.read(4), dtype=dt )[0]


def load_mnist_dataset( directory, download=False ):
    """
    Load mnist-like dataset from the directory.

    If the directory does not exist or is empty, and ```download``` is True, basic mnist dataset
    is downloaded. Otherwise, exception is raised.

    The function returns numpy array with (k, 784) shape, where k is number of images in the dataset.
    """
    dataset = 'train-images-idx3-ubyte.gz'
    dataset_path = _os.path.join( directory, dataset )
    if download:
        if not _os.path.exists( directory ):
            _os.makedirs( directory )
        if _os.path.exists( dataset_path ):
            _os.remove( dataset_path )
        temp_filename, _ = _urllib.request.urlretrieve( DEFAULT_MNIST_URL + dataset )
        _copy( temp_filename, dataset_path )
        with gfile.GFile( dataset_path ) as f:
            size = f.size()
        print( 'Successfully downloaded', dataset, size, 'bytes.' )
    if not _os.path.exists( directory ):
        raise RuntimeError( directory + ': No such directory' )
    if not _os.path.isdir( directory ):
        raise RuntimeError( directory + ': Not a directory' )
    print( 'Extracting', dataset )
    with gfile.Open( dataset_path, mode='rb' ) as f:
        with gzip.GzipFile( fileobj=f ) as bytestream:
            magic = _read32( bytestream )
            if magic != 2051:
                raise ValueError( 'Invalid magic number %d in MNIST image file: %s' % (magic, f.name) )
            num_images = _read32( bytestream )
            rows = _read32( bytestream )
            cols = _read32( bytestream )
            buf = bytestream.read( rows * cols * num_images )
            data = _np.frombuffer( buf, dtype=_np.uint8 )
            data = data.reshape( num_images, rows*cols, 1 )
            # normalize
            data = data.astype( _np.float32 )
            data = _np.multiply( data, 1 / 255 )
    return data


#def load_image_dataset( directory, image_proc=[] ):
#    """
#    Load dataset from directory with raw images (saved as jpeg, png or other common image format).
#
#    If the directory is empty, exception is raised.
#
#    The function returns numpy array with (k, w*h) shape, where k is number of images in the directory,
#    w and h are dimensions of each image (width and height, respectively).
#
#    Each image may be processed by image_proc functions.
#    """
#    if not _os.path.exists( directory ):
#        raise RuntimeError( directory + ': no such directory' )
#    if not _os.path.isdir( directory ):
#        raise RuntimeError( directory + ': not a directory' )
#    images = []
#    image_size = (0, 0)
#    for filename in _os.listdir( directory ):
#        try:
#            img = _PIL_Image.open( _os.path.join( directory, filename ) )
#            image_size = img.size
#            img = _np.array( img )
#            for image_proc_func in image_proc:
#                img = image_proc_func( img )
#            images.append( img )
#        except Exception:
#            pass
#    image_count = len( images )
#    if image_count == 0:
#        raise RuntimeError( directory + ': directory does not contain any images')
#    images = _np.asarray( images )
#    images.reshape( (image_count, image_size[0] * image_size[1] ) )
#    return images


def load_npy_dataset( path ):
    """
    Load dataset from npy array.

    If the file does not exist, exception is raised.

    The function returns numpy array with (k, w*h) shape, where k is number of images in the dataset,
    w and h are dimensions of each image (width and height, respectively).

    Each image may be processed by image_proc functions.
    """
    return _np.load( path )
