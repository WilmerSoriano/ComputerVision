Brief Summary: 
   1. the number of features extracted using each approach
      * SIFT: Avergae keypoint detected per image: 34.5 ± 15.8
      * HOG: 1 histogram per image of length: 1764

   2. the number of correct matches found (for keypoint_matching.py function)
      * Found 759 correct matches after ratio test

   3. the accuracy of the classifiers when evaluated on the test set
      * SIFT: accuracy of 0.34
      * HOG: accuracy of 0.48

Questions:
   1. First, HOG is less about detecting unique marks, I would use SIFT
   since it already allow us to use key point and descriptors to help
   us identify key points of interest.
   Next, to prevent HOG from scanning the entire image, we would crop up the
   original image down into smaller pieces using coordinate location around SIFT descriptors.
   Finally, we would perform HOG function just as normal for each cropped image and should return something similar to SIFT function. 

   2. I believe HOG outperformed SIFT because it limited by what it can do. So, HOG main strength lies in object detection only.
    It able to scan the entire image quicker then SIFT who looks for unique key points. However,  in terms of flexibility, HOG lack invariance in scaling, rotation, and being precise. SIFT is complex hence poor result, but in return is more flexible in what it can do such as rotation, scaling, and object movement.

