#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Quick animated GIF maker
"""

#------------------------------------------------------------------------------
# PROGRAM: make_gif.py
#------------------------------------------------------------------------------
# Version 0.1
# 28 February, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import os, glob
import imageio.v2 as imageio

#----------------------------------------------------------------------------
# MAKE GIF
#----------------------------------------------------------------------------

use_reverse_order = False
png_dir = 'tas_median_20*.png'
gif_str = 'tas_median_20xx.gif'
mp4_str = 'tas_median_20xx.mp4'

if use_reverse_order == True:
    a = glob.glob(png_dir)
    images = sorted(a, reverse=True)
else:
    images = sorted(glob.glob(png_dir))

var = [imageio.imread(file) for file in images]
imageio.mimsave(gif_str, var, fps = 10)

#----------------------------------------------------------------------------
# CLI --> MAKE GIF & MP4
#----------------------------------------------------------------------------

# PNG --> GIF:
# convert -delay 10 -loop 0 png_dir gif_str

# GIF --> MP4
# ffmpeg -i gif_str -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" mp4_str

# MERGE: century level GIFs (if too large for convert)

# sudo apt install gifsicle
# gifsicle  g1.gif  g2.gif  g3.gif > combo.gif 

# -----------------------------------------------------------------------------
print('** END')
