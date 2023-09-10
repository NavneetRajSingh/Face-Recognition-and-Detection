import cv2
import os
import numpy as np


count=0

for path, subdirnames, filenames in os.walk("trainingImages"):

    for filename in filenames:
      if filename.startswith("."):
        print("Skipping File:",filename)
        continue
      img_path=os.path.join(path, filename)
      print("img_path",img_path)
      id=os.path.basename(path)
      img = cv2.imread(img_path)
      if img is None:
        print("Image not loaded properly")
        continue
      resized_image = cv2.resize(img, (100, 100))
      new_path="resizedTrainingImages"+"/"+str(id)
      print("desired path is",os.path.join(new_path, "frame%d.jpg" % count))
      cv2.imwrite(os.path.join(new_path, "frame%d.jpg" % count),resized_image)
      count += 1
