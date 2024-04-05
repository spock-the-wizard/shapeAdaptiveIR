
import sys
sys.path.append("/sss/InverseTranslucent/build")
sys.path.append("/sss/InverseTranslucent/build/lib")
import psdr_cuda
import enoki as ek
import cv2
import numpy as np
import math
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD, Vector3i
from enoki.cuda import Vector20f as Vector20fC
# from enoki.cuda_autodiff import Vector20fD as Vector20fD
from enoki.cuda import Float32 as FloatC
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import os
import sys
import time
import random
import imageio
import json
import datetime
from matplotlib import colormaps
import wandb
import time

import argparse
import glob
import os
import pickle
import sys
import time
from enum import Enum
# sys.path.insert(0,'./copy.py')
import copy
from multiprocessing import Queue

import numpy as np
import skimage
import skimage.io
import tensorflow as tf
from matplotlib import cm
import tqdm

import mitsuba
import mitsuba.render
import nanogui
# import utils.math
sys.path.append('../viz')
sys.path.append('../viz/vae')
# from viz import vae.config
# from viz import extract_shape_features
import vae.config
import vae.config_abs
# import vae.config_angular
# import vae.config_scatter
# import vae.datapipeline
import vae.utils
from mitsuba.core import *
from nanogui import (Button, ComboBox,
                     GroupLayout, ImageView, Label,
                     PopupButton, Widget, Window, entypo, glfw)
from utils.experiments import load_config
from utils.gui import (FilteredListPanel, FilteredPopupListPanel,
                       LabeledSlider, add_checkbox)
from vae.global_config import (DATADIR3D, FIT_REGULARIZATION, OUTPUT3D,
                               RESOURCEDIR, SCENEDIR3D, DATADIR)

import vae.model
# import vae.predictors
# from vae.model import generate_new_samples, sample_outgoing_directions
# from vae.predictors import (AbsorptionPredictor, #AngularScatterPredictor,
#                             ScatterPredictor)
# from viewer.datasources import PointCloud, VectorCloud
from viewer.utils import *
import viewer.utils
from viewer.viewer import GLTexture, ViewerApp
from utils.printing import printr, printg
import utils.mtswrapper



class Scatter3DViewer:
    """Viewer to visualize a given fixed voxelgrid"""

    def set_mesh(self, mesh_file):
        self.mesh_file = mesh_file
        self.mesh, self.min_pos, self.max_pos, self.scene, self.constraint_kd_tree, self.sampled_p, self.sampled_n = setup_mesh_for_viewer_2(
            mesh_file, self.sigma_t, self.g, self.albedo)
        self.shape = self.scene.getShapes()[0]
        self.computed_poly = False

    def get_poly_fit_options(self):
        self.fit_regularization = FIT_REGULARIZATION
        self.use_svd = False
        self.kdtree_threshold = 0.0
        self.use_legacy_epsilon = False
        return {'regularization': self.fit_regularization, 'useSvd': self.use_svd, 'globalConstraintWeight': 0.01,
                # 'order': self.scatter_config.poly_order(), 'kdTreeThreshold': self.kdtree_threshold,
                'order': 3, 'kdTreeThreshold': self.kdtree_threshold,
                'useSimilarityKernel': not self.use_legacy_epsilon, 'useLightspace': False}
        
    def extract_mesh_polys(self,albedo=None,sigma_t=None,g=None):
        if albedo is None:
            albedo = self.albedo
        if sigma_t is None:
            sigma_t = self.sigma_t
        if g is None:
            g = self.g
        coeff_list = [] 
        for i in tqdm.tqdm(range(self.mesh.mesh_positions.shape[1])):
            pos = self.mesh.mesh_positions[:, i].ravel()
            normal = self.mesh.mesh_normal[:, i].ravel()

            coeffs, _, _ = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree, pos, -normal, sigma_t,g, albedo,
                                                          self.fit_opts, normal=normal)
            # Rotate TS
            coeffs_ts = utils.mtswrapper.rotate_polynomial(coeffs, normal, 3)
            coeff_list.append(coeffs_ts)

        self.mesh_polys = np.array(coeff_list)

    def get_shape_features(self,pos,inDirection,sigma_t,g,albedo,normal):
        # pos, inDirection should be np.array 
        coeff_list = [] 
        for idx,(p,d) in enumerate(zip(pos,inDirection)):
            coeffs, p_const, c_const = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree, p, -d, sigma_t, g, albedo,
                                                          self.fit_opts, normal=normal)
            # Rotate TS
            coeffs_ts = utils.mtswrapper.rotate_polynomial(coeffs, d, 3)
            coeff_list.append(coeffs_ts)

        print(p_const.shape)
        return coeff_list

    def __init__(self,mesh_file,albedo,sigma_t,g):
        super(Scatter3DViewer, self).__init__()
        
        self.albedo = albedo
        self.sigma_t = sigma_t
        self.g = g
        
        self.set_mesh(mesh_file)
        self.mesh_polys = None

        self.fit_opts = self.get_poly_fit_options()
        self.fit_opts['useLightspace'] = False

        # self.extract_mesh_polys()


if __name__ == "__main__":
    
    # poly_data = np.load("test_poly_data.npz")
    # poly_data = np.load("test_poly_data.npy",allow_pickle=True)


    parser = argparse.ArgumentParser(description='''SSS Viewer''')
    parser.add_argument('--mesh', )
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args() #args)

    name = args.mesh.split('/')[-1].split('.obj')[0]
    num_med = 5
    # num_med = 2
    # set mesh
    list_alb = np.linspace(0.5,0.99,num=num_med)
    list_sig = np.linspace(40,120,num=num_med)
    # list_sig = np.linspace(40,120,num=num_med)
    list_g = np.linspace(0,0,num=1)
    
    mesh_file = args.mesh
    app = Scatter3DViewer(mesh_file,list_alb[0],list_sig[0],list_g[0])
    sampled_p = app.sampled_p#.points.T
    sampled_n = app.sampled_n

    coeff_list = [] #{idx:[] for idx in range(len(sampled_p))}
    poly_data = {f"{idx}": {
                        'coeffs':[],
                        'albedo':[],
                        'sigma_t':[],
                        'kernelEps':[],
                        'fitScaleFactor':[],
                        'g': [],
                        'maxCoeffs':[],
                        'p' : [],
                        'n': []
                    } for idx in range(len(sampled_p))}
    
    neighs = []
    maxKernelEps = -100
    for alb in list_alb:
        for sig in list_sig:
            for g in list_g:
                kernelEps = vae.utils.kernel_epsilon(g,sig,alb)
                distThreshold = 3*np.sqrt(kernelEps)
                print(distThreshold)
                if kernelEps > maxKernelEps:
                    max_alb = alb
                    max_sig = sig
                    max_g = g
                    maxKernelEps = kernelEps

    # Get max coeffs
    assert maxKernelEps == max([vae.utils.kernel_epsilon(_g,_s,_a) for _g in list_g for _a in list_alb for _s in list_sig])
    print(f"Max kernelEps: {maxKernelEps}")
    max_coeffs = app.get_shape_features(sampled_p,sampled_n,max_sig,max_g,max_alb,sampled_n)
    
    count = 0
    for alb in list_alb:
        for sig in list_sig:
            for g in list_g:
                print(count)
                app.g = g
                app.albedo = alb
                app.sigma_t = sig

                app.set_mesh(mesh_file)
                
                kernelEps = vae.utils.kernel_epsilon(g,sig,alb)
                fitScaleFactor = float(vae.utils.get_poly_scale_factor(kernelEps))
                coeffs = app.get_shape_features(sampled_p,sampled_n,sig,g,alb,sampled_n)
                for idx in range(len(coeffs)):
                    poly_data[f"{idx}"]['coeffs'].append(coeffs[idx])
                    poly_data[f"{idx}"]['albedo'].append(alb)
                    poly_data[f"{idx}"]['sigma_t'].append(sig)
                    poly_data[f"{idx}"]['kernelEps'].append(kernelEps)
                    poly_data[f"{idx}"]['fitScaleFactor'].append(fitScaleFactor)
                    poly_data[f"{idx}"]['maxCoeffs'].append(max_coeffs[idx])
                    poly_data[f"{idx}"]['g'].append(g)
                    poly_data[f"{idx}"]['p'].append(sampled_p[idx])
                    poly_data[f"{idx}"]['n'].append(sampled_n[idx])
                count += 1
    
    
    for key,val in poly_data.items():
        # val is dictionary
        for k, v in val.items():
            poly_data[key][k] = np.array(v)
    

    np.savez(f"{name}_poly_data",**poly_data) #coeff_list)
    # breakpoint()

        

