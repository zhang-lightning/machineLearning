# -*- coding: utf-8 -*-
"""
Download images from given url, resize them to 100 x 100, and rotate 45°.

@author: Zeshen
"""

import os
from PIL import Image
from bs4 import BeautifulSoup
from urllib.request import urlopen

IMG_PATH = "./assets/"
WEB_URL = "http://vis-www.cs.umass.edu/lfw/alpha_last_G.html"

# create image saving directory
if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)

# open url
webpage = urlopen(WEB_URL, timeout=300)

# extract image directories
soup = BeautifulSoup(webpage, features="html.parser")
elements = soup.findAll("img", {"alt": "person image"})

counter = 1
# download images in loop
for element in elements:
    # https://vis-www.cs.umass.edu/lfw/images/Louis_Van_Gaal/Louis_Van_Gaal_0001.jpg
    url = "http://vis-www.cs.umass.edu/lfw/" + element["src"]
    img = Image.open(urlopen(url))
    img = img.crop([20, 20, 120, 120])    # crop image
    img = img.resize((100, 100), Image.LANCZOS)   # resize image
    img = img.rotate(45)  # rotate image to 45°

    # save image
    fName = IMG_PATH + "IMG_%03d.jpg" % (counter, )
    img.save(fName)
    print(fName)
    counter += 1
