
"""
Code to render mock dataset with BSSRDF model rather than Vol. Path. (Mitsuba)
"""
import sys
import psdr_cuda
import enoki as ek
import cv2
import numpy as np
import math
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD, Vector3i
from enoki.cuda import Vector20f as Vector20fC
from psdr_cuda import Bitmap3fD
from enoki.cuda import Float32 as FloatC
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import argparse
import os
import sys
import time
import random
import imageio
import json
from datetime import datetime
from matplotlib import colormaps
import time

import pytorch_ssim
from enoki import *

from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import to_differential, from_differential

from loss import compute_image_matrix #mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss, mesh_cot_laplacian, mesh_uni_laplacian
from loss import hausdorff_distance
from pytorch_msssim import ssim
from AdamUniform import UAdam
from tool_functions import checkpath

from constants import REMESH_DIR, RESULT_DIR, TEXTURE_DIR, SCENES_DIR, ROOT_DIR, REAL_DIR, LIGHT_DIR, ESSEN_DIR
from constants import params_gt
sys.path.append(REMESH_DIR)

import argparse
import glob
import os
import pickle
import sys
import time
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import copy
from multiprocessing import Queue

import numpy as np
import skimage
import skimage.io
import tensorflow as tf
from matplotlib import cm
import tqdm
import trimesh

import mitsuba
import mitsuba.render
import nanogui


sys.path.append('../viz')
sys.path.append('../viz/vae')

os.chdir("/sss/InverseTranslucent/examples/python/scripts")
import vae.config
import vae.config_abs
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
# from viewer.utils import *
import viewer.utils
from viewer.viewer import GLTexture, ViewerApp
from utils.printing import printr, printg
import utils.mtswrapper

def reset_random():
    random_seed = 2
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed) # if use multi-GPU



class Mode(Enum):
    REF = 0
    PREDICTION = 1
    RECONSTRUCTION = 2
    POLYREF = 3
    POLYTRAIN = 4


class Scatter3DViewer:
    """Viewer to visualize a given fixed voxelgrid"""

    def set_mesh(self, mesh_file):
        self.mesh_file = mesh_file
        self.mesh, self.min_pos, self.max_pos, self.scene, self.constraint_kd_tree, self.sampled_p, self.sampled_n = viewer.utils.setup_mesh_for_viewer(
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
        
    def extract_mesh_polys(self,):
        coeff_list = [] 
        for i in tqdm.tqdm(range(self.mesh.mesh_positions.shape[1])):
            pos = self.mesh.mesh_positions[:, i].ravel()
            normal = self.mesh.mesh_normal[:, i].ravel()

            coeffs, _, _ = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree, pos, -normal, self.sigma_t, self.g, self.albedo,
                                                          self.fit_opts, normal=normal)
            # Rotate TS
            coeffs_ts = utils.mtswrapper.rotate_polynomial(coeffs, normal, 3)
            coeff_list.append(coeffs_ts)

        self.mesh_polys = np.array(coeff_list)

    def get_shape_features(self,pos,inDirection,sigma_t,g,albedo,normal):
        # pos, inDirection should be np.array 
        coeff_list = [] 
        for idx,(p,d) in enumerate(zip(pos,inDirection)):
            coeffs, _, _ = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree, p, -d, sigma_t, g, albedo,
                                                          self.fit_opts, normal=normal)
            # Rotate TS
            coeffs_ts = utils.mtswrapper.rotate_polynomial(coeffs, d, 3)
            coeff_list.append(coeffs_ts)

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

        self.extract_mesh_polys()


parser = argparse.ArgumentParser()
parser.add_argument('--scene',              type=str,      default='sphere1')
parser.add_argument('--stats_folder',       type=str,      default="test/debug")
parser.add_argument('--light_file',         type=str,      default="seal")
parser.add_argument('--ref_folder',         type=str,      default="exr_ref")
parser.add_argument('--scene_file',         type=str,    default="../../scenes/inverse/sphere1_out.xml")

parser.add_argument('--sigma_lr',           type=float,    default=0.04)
parser.add_argument('--eta_lr',             type=float,    default=0.01)
parser.add_argument('--albedo_lr',          type=float,    default=0.02)
parser.add_argument('--mesh_lr',            type=float,    default=0.05)
parser.add_argument('--rough_lr',            type=float,   default=0.02)
parser.add_argument('--epsM_lr',            type=float,   default=0.001)

parser.add_argument('--img_weight',         type=float,    default=1.0)
parser.add_argument('--range_weight',         type=float,    default=0.0)
parser.add_argument('--tot_weight',         type=float,    default=0.01)
parser.add_argument('--laplacian',          type=float,    default=60)
parser.add_argument('--sigma_laplacian',    type=float,    default=0)
parser.add_argument('--albedo_laplacian',   type=float,    default=0)
parser.add_argument('--rough_laplacian',   type=float,    default=0)

parser.add_argument('--n_iters',            type=int,      default=30000)
parser.add_argument('--n_resize',           type=int,      default=10000)
parser.add_argument('--n_reduce_step',      type=int,      default=200)
parser.add_argument('--n_dump',             type=int,      default=100)
parser.add_argument('--n_crops',             type=int,      default=1)

parser.add_argument('--seed',               type=int,      default=2)

parser.add_argument('--spp',                type=int,      default=4)
parser.add_argument('--sppe',               type=int,      default=0)
parser.add_argument('--sppse',              type=int,     default=8192)

parser.add_argument('--integrator',         type=str,      default='direct')

parser.add_argument('--albedo_texture',     type=int,      default=256)
parser.add_argument('--sigma_texture',      type=int,      default=256)
parser.add_argument('--rough_texture',      type=int,      default=256)
parser.add_argument('--epsM_texture',      type=int,      default=256)

parser.add_argument('--ref_spp',            type=int,      default=50)
parser.add_argument('--no_init',            type=str,     default="yes")
parser.add_argument('--d_type',             type=str,     default="custom")
parser.add_argument('--silhouette',         type=str,     default="no")

parser.add_argument('--render_gradient', action="store_true")
parser.add_argument('--sensor_id',         type=int,    default=0)
parser.add_argument('--debug', action="store_true", default=False)
parser.add_argument('--isSweep', action="store_true", default=False)
parser.add_argument('--opaque', action="store_true", default=False)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


import matplotlib as mpl
class MidpointNormalize(mpl.colors.Normalize):
    """
    class to help renormalize the color scale
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def saveArgument(ars, file):
    with open(file, 'w') as f:
        json.dump(ars.__dict__, f, indent=2)

def loadArgument(ars, file):
    with open(file, 'r') as f:
        ars.__dict__ = json.load(f)
    return ars

def rmse(predictions, targets):
    # Computes error map for each pixel
    return np.sqrt(((predictions - targets) ** 2).mean(axis=-1))

def rmse_total(predictions, targets):
    # Computes error map for each pixel
    return np.sqrt(((predictions - targets) ** 2).mean())

def enoki2img(arr,shape=(512,512,-1),filename='test.png'):
    img = arr.numpy().reshape(shape)
    cv2.imwrite(filename, img * 255.0)

def opt_task(isSweep=True):
    args = parser.parse_args()
    isSweep = args.isSweep

    # Forward mode, disable learning rate
    args.mesh_lr = 0.0
    args.sigma_lr = 0.0
    args.albedo_lr = 0.0
    args.g_lr = 0.0

    reset_random()
    if args.scene.endswith("het"):
        args.scene = args.scene[:-3]
    if args.scene == 'kiwi':
        args.d_type = 'real'
    if args.scene == 'soap':
        args.d_type = 'real'

    # write intermedia results to ... 
    # destdir = RESULT_DIR + "/{}/".format(args.scene)
    # checkpath(destdir)
    # statsdir = destdir + "/{}_{}/".format(args.stats_folder, args.seed)
    # checkpath(statsdir)

    # argsdir = destdir + "/{}_{}/settings_{}.txt".format(args.stats_folder, args.seed, datetime.now())
    # saveArgument(args, argsdir)

    # load scene
    sc = psdr_cuda.Scene()
    if os.path.exists(args.scene_file): #args.scene_file is not None:
        sc.load_file(args.scene_file)
    elif args.d_type == "syn":
        sc.load_file(SCENES_DIR + "/{}.xml".format(args.scene))
    elif args.d_type == "custom":
        sc.load_file(SCENES_DIR + "/{}_out.xml".format(args.scene))
    else: 
        sc.load_file(SCENES_DIR + "/{}_real.xml".format(args.scene))
    
    
    # sc.opts.cropheight = 256
    # sc.opts.cropwidth = 256
    # sc.opts.cropheight = 64
    # sc.opts.cropwidth = 64
    sc.opts.cropheight = sc.opts.height // args.n_crops
    sc.opts.cropwidth = sc.opts.width // args.n_crops
    
    ro = sc.opts
    
    ro.sppse = args.sppse
    ro.spp = args.spp
    ro.sppe = args.sppe
    ro.log_level = 0

    mesh_key = "Mesh[id=init]"
    material_key = "BSDF[id=opt]"
    num_sensors = sc.num_sensors
    isBaseline = sc.param_map[material_key].type_name() == "HeterSub"
    if args.albedo_lr == 0:
        sc.param_map[material_key].albedo = Bitmap3fD(params_gt[args.scene]['albedo'])
    if args.sigma_lr == 0:
        sc.param_map[material_key].sigma_t = Bitmap3fD(params_gt[args.scene]['sigmat'])

    if args.d_type == "syn":
        lightdir = LIGHT_DIR + '/lights-sy-{}.npy'.format(args.light_file)
    elif args.scene == "duck":
        lightdir = LIGHT_DIR + '/lights-head.npy'
    elif args.d_type == "custom":
        lightdir = LIGHT_DIR + '/lights-{}.npy'.format(args.scene)
    elif args.d_type == "real":
        lightdir = LIGHT_DIR + '/lights-gantry.npy'
    else:
        lightdir = LIGHT_DIR + '/essen-lights-gantry.npy'
    lights = np.load(lightdir)
    if args.scene == "duck":
        lights = lights[:25,:]
        lights[:,:3] = np.array([1.192, -1.3364, 0.889])
    if args.scene == "plane": 
        lights[:,:3] = np.array([0.0, 2.0,0.0])

    if args.integrator == 'direct':
        myIntegrator = psdr_cuda.DirectIntegrator()
    else:
        myIntegrator = psdr_cuda.ColocateIntegrator()
    silhouetteIntegrator = psdr_cuda.FieldExtractionIntegrator("silhouette")
    
    # load reference images
    if args.d_type == "syn":
        refdir = RESULT_DIR + "/{}/{}/".format(args.scene, args.ref_folder)
        maskdir = RESULT_DIR + "/{}/silhouette/".format(args.scene)
    elif args.d_type == "real":
        refdir = REAL_DIR + "/hdr{}/{}/".format(args.scene, args.ref_folder)
    elif args.d_type == "custom":
        # NOTE: tmp setting for sanity check
        refdir = REAL_DIR + "/{}/{}/".format(args.scene,args.ref_folder)
    else:
        refdir = ESSEN_DIR + "/hdr{}/{}/".format(args.scene, args.ref_folder)

    if args.scene == "cone4":
        mesh_file = "../../smoothshape/vicini/cone_subdiv.obj"
    elif 'head' in args.scene:
        mesh_file = "../../smoothshape/head_v2.obj"
    elif args.scene == "cylinder4":
        mesh_file = "../../smoothshape/vicini/cylinder_subdiv.obj"
    elif 'kettle' in args.scene: #== "kettle1":
        mesh_file = "../../smoothshape/final/kettle_.obj"
    elif args.scene == "duck":
        mesh_file = "../../smoothshape/duck_v2.obj"
    elif args.scene == "sphere1":
        mesh_file = "../../smoothshape/sphere_v2.obj"
    elif 'maneki' in args.scene: #== "maneki1":
        mesh_file = "../../smoothshape/final/maneki_.obj"
    elif args.scene == "pig1":
        mesh_file = "../../smoothshape/pig.obj"
    elif args.scene == "horse1":
        mesh_file = "../../smoothshape/horse.obj"
    elif args.scene == 'botijo':
        mesh_file = "../../smoothshape/final/botijo2_.obj"
        # mesh_file = "../../smoothshape/botijo_.obj"
    elif args.scene == 'buddha1':
        mesh_file = "../../smoothshape/final/buddha_.obj"
        # mesh_file = "../../smoothshape/buddha_.obj"
    elif 'botijo' in args.scene: # botijo2, botijo3
        mesh_file = "../../smoothshape/botijo2.obj"
    elif args.scene == "cube":
        mesh_file = "../../smoothshape/cube_subdiv25.obj"
    elif args.scene == "pyramid4":
        mesh_file = "../../smoothshape/vicini/pyramid_.obj"
    elif args.scene == "plane":
        mesh_file = "../../smoothshape/plane_almost_v2.obj"
    elif args.scene == "kiwi":
        mesh_file = "../../smoothshape/cube_3_init_uv.obj"
    elif args.scene == "soap":
        mesh_file = "../../smoothshape/soap_init.obj"
    else:
        mesh_file = params_gt[args.scene]['mesh']
        # raise NotImplementedError


    def precompute_mesh_polys():
        albedo = sc.param_map[material_key].albedo.data.numpy()
        sigma_t = sc.param_map[material_key].sigma_t.data.numpy()
        g = sc.param_map[material_key].g.data.numpy()[0]
        if sigma_t.size > 3:
            sigma_t = sigma_t.mean(axis=0,keepdims=True)
        if albedo.size > 3:
            albedo = albedo.mean(axis=0,keepdims=True)

        # Setup mesh and precompute mesh_polys
        alb = albedo[:,0]
        sig = sigma_t[:,0]

        time_start = time.time()
        app = Scatter3DViewer(mesh_file,alb,sig,g)
        sc.param_map[mesh_key].load_poly(app.mesh_polys,0)
        # DEBUG: testing poly bug
        # for i in range(1,3):
        #     alb = albedo[:,i]
        #     sig = sigma_t[:,i]
        #     app.sigma_t = sig
        #     app.alb = sig
        #     app.extract_mesh_polys()
        #     sc.param_map[mesh_key].load_poly(app.mesh_polys,i)

        if sc.param_map[material_key].monochrome:
            sc.param_map[mesh_key].load_poly(app.mesh_polys,1)
            sc.param_map[mesh_key].load_poly(app.mesh_polys,2)
        else:
            for i in range(1,3):
                alb = albedo[:,i]
                sig = sigma_t[:,i]
                app.sigma_t = sig
                app.alb = sig
                app.extract_mesh_polys()
                sc.param_map[mesh_key].load_poly(app.mesh_polys,i)

        time_end = time.time()
        print(f"Precomputing polynomials took {time_end - time_start} seconds")

    # FIXME
    # if not args.debug and not isBaseline:
    if not isBaseline:
        precompute_mesh_polys()

    tars = [] 
    maks = []
    tmeans = []
    for i in range(num_sensors):
        if args.d_type == "syn":
            filename = refdir + "/l{}_s{}.exr".format(i, i)
        elif args.d_type == "custom":
            filename = refdir + "{}.exr".format(i)
        else:
            filename = refdir + "{}_{:05d}.exr".format(args.scene, i)
        
        target = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # from BGR TO RGB
        t1 = torch.from_numpy(cv2.cvtColor(target, cv2.COLOR_RGB2BGR)).float()
        t1 = t1.reshape((-1, 3))
        t1[t1 < 0] = 0
        
        tars.append(t1)
        
        if args.d_type == "syn":
            maskfile = maskdir + "/s_{:04}.exr".format(i)
            # print(maskfile)
            mask = cv2.imread(maskfile, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = torch.from_numpy(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)).float().reshape((-1, 3)) 
        else:  
            mask = torch.from_numpy(cv2.cvtColor(target, cv2.COLOR_RGB2BGR)).float().reshape((-1, 3)) 
        mask[mask > 0] = 1

        maks.append(mask)

    def active_sensors(batch, num_sensors):
        indices = torch.tensor(random.sample(range(num_sensors), batch))
        return indices

    def active_light(batch, num_lights):
        indices = torch.tensor(random.sample(range(num_lights), batch))
        indices = torch.tensor([1]) #random.sample(range(num_lights), batch))
        return indices

    def texture_range_loss(A, S, R, G, weight):
        lossS = torch.pow(torch.where(S < 0, -S, torch.where(S > 150.0, S - 150.0, torch.zeros_like(S))), 2)
        lossA = torch.pow(torch.where(A < 0.0, -A, torch.where(A > 1.0, A - 1, torch.zeros_like(A))), 2)
        lossR = torch.pow(torch.where(R < 0.01, 0.02 - R, torch.where(R > 2.0, R - 2.0, torch.zeros_like(R))), 2)
        lossG = torch.pow(torch.where(G < 1.0, 2.0-G, torch.where(G > 10.0, G - 1, torch.zeros_like(G))), 2)
        loss = (lossA.mean() + lossR.mean() + lossG.mean() + lossS.mean()) * weight
        return loss

    def renderC(scene,integrator,sensor_id,coeffs=None,fit_mode='avg'):
        image = integrator.renderC(scene, sensor_id)
        its = None 
        coeff = None
        return image,coeffs

                
    def renderNtimes(scene, integrator, n, sensor_id):

        image,coeffs = renderC(scene,integrator,sensor_id)
        
        weight = 1.0 / n
        out = image.numpy().reshape((scene.opts.cropheight, scene.opts.cropwidth, 3))
        
        for i in range(1, n):
            image,_ = renderC(scene, integrator, sensor_id, coeffs=coeffs)
            out += image.numpy().reshape((scene.opts.cropheight, scene.opts.cropwidth, 3))
            
        out *= weight
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out
    
    render_batch = 1

    def renderDataset(sidxs,outDir):
        images = []
        
        for idx in sidxs:
            sc.setlightposition(Vector3fD(lights[idx][0], lights[idx][1], lights[idx][2]))
            print(lights[idx])
            img2 = renderNtimes(sc, myIntegrator, args.ref_spp, idx)
            
            # target2 = tars[idx].numpy().reshape((ro.height,ro.width,-1)) #[coy:coy+coh,cox:cox+cow]
            # target2 = cv2.cvtColor(target2, cv2.COLOR_RGB2BGR)
            # target2 = np.clip(target2,a_min=0.0, a_max=1.0)
            cv2.imwrite(f"{outDir}/{idx}.exr", img2)
        
        print(f"Wrote {len(sidxs)} images to {outDir}")



    def optTask():
        
        sc.opts.cropheight = sc.opts.height
        sc.opts.cropwidth = sc.opts.width
        sc.opts.crop_offset_x = 0 
        sc.opts.crop_offset_y = 0 
        sc.opts.spp = 1
        sc.opts.rgb = 0
        sc.opts.mode = 1 if args.opaque else 0
        sc.configure()

        all_sensors = np.arange(num_sensors)
        curtime = datetime.now().strftime("%m%d%H%M%S")
        tag_model = 'planar' if isBaseline else 'shapeada'
        tag_mode = '_opaque' if args.opaque else ''
        outDir = f"../../../data_kiwi_soap/realdata/{args.scene}_{tag_model}{tag_mode}_{curtime}"
        os.makedirs(outDir)
        renderDataset(all_sensors,outDir)
        

    optTask()
        

if __name__ == "__main__":
    opt_task()
