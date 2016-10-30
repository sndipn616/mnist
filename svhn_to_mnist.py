#!/usr/bin/env python
#
# Olivier Soares
# June 2016
#
# MNIST Dataset Renderer.
#
# The MIT License (MIT)
#
# Copyright (c)2016 Olivier Soares
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""
  SVHN to MNIST Dataset Converter.

  This script convert a SVHN dataset (Street View House Numbers)
  to a MNIST dataset format.

  See the SVHN dataset format here:
    http://ufldl.stanford.edu/housenumbers

  See the MNIST dataset format here:
    http://yann.lecun.com/exdb/mnist
"""


import os, array, argparse, numpy as np, scipy.io as spio, cv2


def int32(val):
  '''
  Convert an integer value to a 32-bit bitarray.
  
  Args:
    val: integer value

  Returns:
    32-bit bitarray
  '''

  return bytearray([(val >> i & 0xFF) for i in (24, 16, 8, 0)])


def svhn_to_mnist(svhn_path, output_dir, res, prefix):
  '''
  Convert a SVHN dataset to a MNIST dataset format.

  Args:
    svhn_path : path to the svhn mat file
    output_dir: output directory to store the MNIST dataset
    res       : output MNIST dataset resolution
    prefix    : prefix to the output MNIST dataset
  '''

  if not prefix:
    prefix = "mnist"

  try:
    mat    = spio.loadmat(svhn_path)
    pixels = mat["X"]
    labels = mat["y"]
  except Exception as e:
    print("Can't read '%s': %s" % (svhn_path, e))
    return

  # Input parameters
  num_img        = len(labels)
  img_src_width  = len(pixels)
  img_src_height = len(pixels[0])
  img_src_depth  = len(pixels[0][0])
  img_src_size   = img_src_width * img_src_height

  # Output parameters
  img_dst_width  = res
  img_dst_height = res

  if len(pixels[0][0][0]) != num_img:
    print("Invalid input file '%s'" % input_file)
    return

  # Check the output directory exists, create it if not
  if not os.path.isdir(output_dir):
    try:
      os.makedirs(output_dir)
    except Exception as e:
      print("Can't create '%s': %s" % (output_dir, e))
      return

  # Path to the image and label files
  image_path = os.path.join(output_dir, "%s-images-idx3-ubyte" % prefix)
  label_path = os.path.join(output_dir, "%s-labels-idx1-ubyte" % prefix)

  # Create the MNIST files
  try:
    image_file = open(image_path, "wb")
    label_file = open(label_path, "wb")
  except Exception as e:
    print("Can't write output MNIST files: %s" % e)
    return

  # Magic number
  image_file.write(int32(0x803))
  label_file.write(int32(0x801))

  # Number of images
  image_file.write(int32(num_img))
  label_file.write(int32(num_img))

  # Image width and height
  image_file.write(int32(img_dst_width))
  image_file.write(int32(img_dst_height))

  print("Encoding %d image(s) from '%s' to '%s' and '%s'" %\
       (num_img, svhn_path, image_path, label_path))

  perc      = 0
  perc_iter = 10
  perc_next = 10
  images    = []
  for i in range(num_img):
    perc = int(100 * float(i + 1) / num_img)
    if perc >= perc_next:
      print("%d%% done" % perc)
      perc_next += perc_iter
    # Read the image from the mat, resize it and add it to the list
    img = np.zeros((img_src_width, img_src_height), "uint8")
    if img_src_depth == 1:
      for y in range(img_src_height):
        for x in range(img_src_width):
          img[x][y] = pixels[x][y][0][i]
    elif img_src_depth == 3:
      for y in range(img_src_height):
        for x in range(img_src_width):
          # Get the luminosity
          R         = pixels[x][y][0][i]
          G         = pixels[x][y][1][i]
          B         = pixels[x][y][2][i]
          img[x][y] = 0.2126 * R + 0.7152 * G + 0.0722 * B
    else:
      print("Unknown image depth of %d" % img_src_depth)
      return
    img = cv2.resize(img, (img_dst_width, img_dst_height))
    img.tofile(image_file)
    label_file.write(bytearray([labels[i][0]]))
    image_file.flush()
    label_file.flush()

  # Cleanup
  image_file.close()
  label_file.close()


def main():
  '''
  Main function.
  '''

  # Arguments parser
  parser = argparse.ArgumentParser(description = "SVHN to MNIST Dataset.")
  parser.add_argument("-svhn", help = "Input SVHN",
                      type = str, required = True)
  parser.add_argument("-out", help = "Output directory",
                      type = str, required = True)
  parser.add_argument("-res", help = "Resolution of the output images",
                      type = int, default = 28)
  parser.add_argument("-prefix", help = "Output prefix", type = str)

  args = parser.parse_args()

  # Converter
  svhn_to_mnist(args.svhn, args.out, args.res, args.prefix)


if __name__ == "__main__":
  main()
