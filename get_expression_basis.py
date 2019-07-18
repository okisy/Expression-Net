import scipy.io
import numpy as np
import time
import utils
import os

start = time.time()
BFM_path = './Shape_Model/BaselFaceModel_mod.mat'  # 140MB
model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
model = model["BFM"]
faces = model.faces - 1
print("load BFM costs: {}s".format(time.time() - start))

basis_folder = './exp_basis'  # The location where .ply files are saved
if not os.path.exists(basis_folder):
    os.makedirs(basis_folder)

S = model.shapeMU + model.expMU
numVert = int(S.shape[0] / 3)
S = np.reshape(S, (numVert, 3))
basis_name = basis_folder + '/basis_mean' + '.ply'
utils.write_ply_textureless(basis_name, S, faces)

shape_param = np.zeros_like(model.shapeMU)
for i in range(29):
    start = time.time()
    exp_param = np.zeros_like(model.expMU)
    exp_param[i] = 3.0
    SE, _ = utils.projectBackBFM_withExpr(model, shape_param, exp_param)
    basis_name = basis_folder + '/basis_' + str(i) + '.ply'
    utils.write_ply_textureless(basis_name, SE, faces)
    print(str(i) + ': {}s'.format(time.time() - start))
