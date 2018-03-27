In WinkDetect.py:
When I detect face, I use relatively small scaleFactor to search all the possible positions and use large minSize and large minNeighbors to eliminate false positive faces.
When I detect eyes, I use relatively small scaleFactor to search all the possible positions and use small minSize to search all the possible size of eyes. To eliminate false positives, I use large minNeighbors.
As for CascadeClassifiers, I use the default choices since it is better than others.

In ShushDetect.py
When I detect face, I use relatively small scaleFactor to search all the possible positions and use small minSize to search all the possible size of faces. To eliminate false positives, I use large minNeighbors.
When I detect eyes, I use relatively small scaleFactor to search all the possible positions and use small minSize to search all the possible size of eyes. To eliminate false positives, I use very large minNeighbors.
As for CascadeClassifiers, I use the default choices since it is better than others.