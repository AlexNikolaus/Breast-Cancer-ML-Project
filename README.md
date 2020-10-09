# Breast-Cancer-ML-Project

This is my first Machine Learning project. It consists of predicting breast cancer diagnosis (either benign or malignant) based on the parameters from digitized images of cancerous cell nuclei.

I have used three ML models: k-NN, Random forest and Gradient boosting, followed by a voting system to choose the final prediction for the test set in order to predict the diagnosis.


The parameters fitted in the model are the means of :
a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
