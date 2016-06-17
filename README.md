This code aims to predict the differential dose volume hisogram (DVH) from the contoured anatomical scan.

"oar_diffdvh_crsentropy_prediction_2d_copy.py" 
mian function to run

"3D_mat2np.py"
Converts the .mat files for the contours and differential DVHs to .numpy files using pickle

data files
"contour_bladder_tnsr.mat": 4D array 50x50x2835 w/ the contour slices for the Bladder across all 110 patients

"diff_dvh_bladder.mat": 2D array 2835X10 w/ the differential DVH for all the Bladder sluices; for each slice's DVH 10 bins used

"contour_rectum_tnsr.mat": 4D array 50x50x2835 w/ the contour slices for the Rectum across all 110 patients

"diff_dvh_rectum.mat": 2D array 2835X10 w/ the differential DVH for all the Rectum sluices; for each slice's DVH 10 bins used

"contour_PTV_tnsr.mat": 4D array 50x50x2835 w/ the contour slices for the PTV across all 110 patients

"diff_dvh_PTV.mat": 2D array 2835X10 w/ the differential DVH for all the PTV slices; for each slice's DVH 10 bins used
