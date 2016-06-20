This code aims to predict the differential dose volume hisogram (DVH) from the contoured anatomical scan.

Neural network structure

Input (100x100x11) --> Convolutional-20 filters (5x5x11) --> Avg. pooling (2) --> Convolutional-20 filters (4x4) --> Avg. pooling (2) --> Convolutional-20 filters (3x3) --> Avg. pooling (2) --> Fully connected 25 nodes --> Output-10 nodes


There are 110 pateints, and for each patients only the contour slices where one of PTV. Rectum, or Bladder exists. This results in 2835 contour slices. 

To generate input for each slice the neighboring slices [-5,+5] is considered to end up with 2835 3D volumes of size 50x50x11.



Main function

"oar_diffdvh_crsentropy_prediction_2d.py" 
mian function to run

"3D_mat2np.py"
Converts the .mat files for the contours and differential DVHs to .numpy files using pickle

Data

"contour_bladder_tnsr.mat": 4D array 50x50x2835 w/ the contour slices for the Bladder across all 110 patients

"diff_dvh_bladder.mat": 2D array 2835X10 w/ the differential DVH for all the Bladder sluices; for each slice's DVH 10 bins used

"contour_rectum_tnsr.mat": 4D array 50x50x2835 w/ the contour slices for the Rectum across all 110 patients

"diff_dvh_rectum.mat": 2D array 2835X10 w/ the differential DVH for all the Rectum sluices; for each slice's DVH 10 bins used

"contour_PTV_tnsr.mat": 4D array 50x50x2835 w/ the contour slices for the PTV across all 110 patients

"diff_dvh_PTV.mat": 2D array 2835X10 w/ the differential DVH for all the PTV slices; for each slice's DVH 10 bins used
