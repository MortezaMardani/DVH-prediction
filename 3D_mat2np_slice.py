#convert .mat files into numpy arrays

#import h5py 
#f = h5py.File('/home/morteza/Dropbox/Research/Research-Postdoc/treatment_planning/Data/prostate_data_186cases/vv.mat') 
#g = f['vv']
#data = np.array(g)

import numpy as np
import scipy.io as sio
import pickle

xx = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/num_slice_patient.mat')
num_slice_patient = xx['num_slice_patient']

#xx = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/scan_slice.mat')
#scan_slice = xx['scan_slice']

xx_ptv = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/contour_ptv_tnsr.mat')
contour_slice_ptv = xx_ptv['contour_ptv_tnsr']

xx_rectum = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/contour_rectum_tnsr.mat')

contour_slice_rectum = xx_rectum['contour_rectum_tnsr']

xx_bladder = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/contour_bladder_tnsr.mat')

contour_slice_bladder = xx_bladder['contour_bladder_tnsr']

yy_ptv = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/diff_dvh_ptv.mat')
diff_dvh_ptv_matrix = yy_ptv['diff_dvh_ptv']

yy_rectum = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/diff_dvh_rectum.mat')
diff_dvh_rectum_matrix = yy_rectum['diff_dvh_rectum']

yy_bladder = sio.loadmat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/diff_dvh_bladder.mat')
diff_dvh_bladder_matrix = yy_bladder['diff_dvh_bladder']

print type(num_slice_patient)
print np.size(num_slice_patient)
print num_slice_patient[0][1]

m=50
n=50
p=11
l=0
data_dict = []
for i in range(114):
       
        for k in range(num_slice_patient[0][i]):

              contour_ptv_3d = np.zeros((m,n,p))
              contour_rectum_3d = np.zeros((m,n,p))
              contour_bladder_3d = np.zeros((m,n,p))

              for j in range(np.maximum(0,k-5)-k,np.minimum(k+5,num_slice_patient[0][i])-k,1):

                   contour_ptv_3d[:,:,j+5] = contour_slice_ptv[:,:,l+j]
                   contour_rectum_3d[:,:,j+5] = contour_slice_rectum[:,:,l+j]
                   contour_bladder_3d[:,:,j+5] = contour_slice_bladder[:,:,l+j]

              diff_dvh_ptv = diff_dvh_ptv_matrix[:,l]
              diff_dvh_rectum = diff_dvh_rectum_matrix[:,l]
              diff_dvh_bladder = diff_dvh_bladder_matrix[:,l]

              l = l+1

              data_dict.append({'contour_ptv':contour_ptv_3d,'contour_rectum':contour_rectum_3d,'contour_bladder':contour_bladder_3d, 'diff_dvh_ptv':diff_dvh_ptv, 'diff_dvh_rectum':diff_dvh_rectum, 'diff_dvh_bladder':diff_dvh_bladder})
     
file_Name = '/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/data_slice_pickle'
fileObject = open(file_Name, 'wb') 
pickle.dump(data_dict, fileObject)

fileObject.close()
fileObject = open(file_Name,'r')  
b = pickle.load(fileObject)


