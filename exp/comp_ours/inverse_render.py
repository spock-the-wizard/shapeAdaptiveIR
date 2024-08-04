
import sys
import psdr_cuda
import enoki as ek
import cv2
import numpy as np
import math
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD, Vector3i as Vector3iD
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
import pickle
import imageio
import json
import datetime
import wandb
import time
from enoki import *
import glob
import matplotlib.pyplot as plt
import random
import copy
from multiprocessing import Queue

import numpy as np
import matplotlib as mpl


root_code="/sss/InverseTranslucent/examples/python/scripts"
RESULT_DIR=os.path.abspath('./results')
print(RESULT_DIR)
os.chdir(root_code)
sys.path.append(f"{root_code}/..")

from loss import compute_image_matrix #mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss, mesh_cot_laplacian, mesh_uni_laplacian
from loss import hausdorff_distance
from AdamUniform import UAdam
from tool_functions import checkpath

from poly_coeffs import * # Added
from constants import REMESH_DIR, TEXTURE_DIR, SCENES_DIR, ROOT_DIR, REAL_DIR, LIGHT_DIR, ESSEN_DIR
from constants import params_gt
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import to_differential, from_differential


# constants
scale_sig = 100

def reset_random():
    random_seed = 2
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed) # if use multi-GPU

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
parser.add_argument('--n_remesh',           type=int,      default=10000)
parser.add_argument('--n_reduce_step',      type=int,      default=200)
parser.add_argument('--n_dump',             type=int,      default=100)
parser.add_argument('--n_fitpoly',             type=int,      default=300)
parser.add_argument('--n_crops',            type=int,      default=1)
parser.add_argument('--n_vaeModeChange',    type=int,      default=10000, help='No. Iterations after which to change to vaeMode 1')
parser.add_argument('--vaeMode',            type=int,      default=0)

parser.add_argument('--seed',               type=int,      default=2)

parser.add_argument('--spp',                type=int,      default=4)
parser.add_argument('--sppe',               type=int,      default=0)
parser.add_argument('--sppse',              type=int,     default=8192)

parser.add_argument('--integrator',         type=str,      default='direct')

parser.add_argument('--maxDistScale',            type=float,      default=1.0)
parser.add_argument('--albedo_texture',     type=int,      default=256)
parser.add_argument('--sigma_texture',      type=int,      default=256)
parser.add_argument('--rough_texture',      type=int,      default=256)
parser.add_argument('--epsM_texture',      type=int,      default=256)

parser.add_argument('--ref_spp',            type=int,      default=50)
parser.add_argument('--no_init',            type=str,     default="yes")
parser.add_argument('--d_type',             type=str,     default="custom")
parser.add_argument('--silhouette',         type=str,     default="no")

parser.add_argument('--render_gradient', action="store_true")
parser.add_argument('--render_results', action="store_true")
parser.add_argument('--sensor_id',         type=int,    default=0)
parser.add_argument('--debug', action="store_true", default=False)
parser.add_argument('--isSweep', action="store_true", default=False)
parser.add_argument('--opaque', action="store_true", default=False)
parser.add_argument('--onlySSS', action="store_true", default=True)
parser.add_argument('--isFD', action="store_true", default=False)
parser.add_argument('--isBaseline', action="store_true", default=False)
parser.add_argument('--randomInit', action="store_true", default=False)
parser.add_argument('--sweep_num',      type=int,      default=0)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
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
    # isSweep = args.isSweep
    shadeMode = 1 if args.opaque else (2 if args.onlySSS else 0)

    if not args.debug:
        if isSweep:
            run = wandb.init(
                    project="shapeAdaptiveIR",
            )
            args.stats_folder = args.stats_folder+f'_{run.name}'
            wandb.config.update(args)
            for key,val in wandb.config.items():
                if hasattr(args,key):
                    setattr(args,key,val)
            args.scene = wandb.config['scene']
            # Update n_crops to accomodate larger spp
            # if args.spp > 8:
            #     args.n_crops = 2
            if args.isBaseline and args.isFD:
                print("Cannot be FD mode for baseline!")
                exit()
            wandb.config.update(args,allow_val_change=True)
        else:
            run = wandb.init(
                    project="shapeAdaptiveIR",
                    config=args
            )
            args.stats_folder = args.stats_folder+f'_{run.name}'
            wandb.config.update(args,allow_val_change=True)


    # write intermedia results to ... 
    destdir = RESULT_DIR + "/{}/".format(args.scene)
    checkpath(destdir)
    statsdir = destdir + "/{}_{}/".format(args.stats_folder, args.seed)
    # Save dir for rendered result
    if args.render_results:
        destdir = destdir.replace('results','results_render')
        statsdir = f"{destdir}/{datetime.datetime.now()}"
    checkpath(statsdir)
    argsdir = statsdir+ "/settings_{}.txt".format(datetime.datetime.now())
    saveArgument(args, argsdir)

    # load scene
    sc = psdr_cuda.Scene()
    if os.path.exists(args.scene_file): #args.scene_file is not None:
        sc.load_file(args.scene_file)
    else:
        scene_file = f"{SCENES_DIR}/{args.scene}_out.xml"
        if args.isBaseline:
            from replace_xml import create_baseline_xml
            scene_file_bs = f"{SCENES_DIR}/{args.scene}_out_baseline.xml"
            create_baseline_xml(scene_file,scene_file_bs)
            scene_file = scene_file_bs
        sc.load_file(scene_file)
    
    
    sc.opts.cropheight = sc.opts.height // args.n_crops
    sc.opts.cropwidth = sc.opts.width // args.n_crops
    
    ro = sc.opts
    ro.sppse = args.sppse
    ro.spp = args.spp
    ro.sppe = args.sppe
    ro.log_level = 0

    mesh_key = "Mesh[id=init]"
    material_key = "BSDF[id=opt]"
    # Use 49 images [0-48] due to missing 49.exr for head1
    num_sensors = sc.num_sensors - 1
    isBaseline = sc.param_map[material_key].type_name() == "HeterSub"

    if args.randomInit:
        # Randomly initialize scattering parameters
        import pickle
        with open("../../random_init_240717_050427.pickle",'rb') as f:
            random_inits = pickle.load(f)
        
        init_albedo = random_inits['init_albedo'][args.sweep_num] 
        init_sigmat = random_inits['init_sigmat'][args.sweep_num] 
        # init_albedo = np.random.uniform(0.75,0.99,(3))
        # init_sigmat = np.random.uniform(30,90,(3))
        print(f"Init albedo: {init_albedo}")
        print(f"Init sigmat: {init_sigmat}")
        # init_sigmat /= scale_sig
        sc.param_map[material_key].albedo = Bitmap3fD(init_albedo)
        sc.param_map[material_key].sigma_t = Bitmap3fD(init_sigmat)
        
    # Fix random seed for reproducibility
    reset_random()
        
    
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
    if args.scene == 'maneki2':
        lights[:,:3] = np.array([0.529, 1.5237, 1.2932]) # lights 

    if args.integrator == 'direct':
        myIntegrator = psdr_cuda.DirectIntegrator()
    else:
        myIntegrator = psdr_cuda.ColocateIntegrator()
    silhouetteIntegrator = psdr_cuda.FieldExtractionIntegrator("silhouette")
    
    # load reference images
    refdir = REAL_DIR + "/{}/{}/".format(args.scene,args.ref_folder)

    if args.n_iters == 1: # forward mode
        mesh_file = params_gt[args.scene]['mesh']
    else:
        if 'cone' in args.scene:
            mesh_file = "../../smoothshape/vicini/cone_subdiv.obj"
        elif 'gargoyle' in args.scene:
            mesh_file = "../../smoothshape/final/gargoyle.obj"
        elif 'dragon' in args.scene:
            mesh_file = "../../smoothshape/final/dragon.obj"
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
            mesh_file = "../../smoothshape/final/botijo2_.obj"
            # botijo2.obj"
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
        elif 'croissant' in args.scene:
            mesh_file = "../../smoothshape/final/croissant.obj"
        else:
            mesh_file = "../../smoothshape/init/source.obj"


    def precompute_mesh_polys(mesh_file=mesh_file):
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
        
        sc.configure()

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
        
        # tmeans.append(meanvalue)

    def active_sensors(batch, num_sensors):
        indices = torch.tensor(random.sample(range(num_sensors), batch))
        return indices

    def active_light(batch, num_lights):
        indices = torch.tensor(random.sample(range(num_lights), batch))
        indices = torch.tensor([1]) #random.sample(range(num_lights), batch))
        return indices

    def texture_range_loss(A, S, R, G, weight):
        lossS = torch.pow(torch.where(S < 0, -S, torch.where(S > 1.50, S - 1.50, torch.zeros_like(S))), 2)
        lossA = torch.pow(torch.where(A < 0.0, -A, torch.where(A > 1.0, A - 1, torch.zeros_like(A))), 2)
        lossR = torch.pow(torch.where(R < 0.01, 0.02 - R, torch.where(R > 2.0, R - 2.0, torch.zeros_like(R))), 2)
        lossG = torch.pow(torch.where(G < 1.0, 2.0-G, torch.where(G > 10.0, G - 1, torch.zeros_like(G))), 2)
        loss = (lossA.mean() + lossR.mean() + lossG.mean() + lossS.mean()) * weight
        return loss

    def renderC(scene,integrator,sensor_id,coeffs=None,fit_mode='avg'):
        image = integrator.renderC(scene, sensor_id)
        coeff = None
        return image,coeffs
                
    def renderNtimes(scene, integrator, n, sensor_id):
        image = integrator.renderC(scene, sensor_id)
        
        weight = 1.0 / n
        out = image.numpy().reshape((scene.opts.cropheight, scene.opts.cropwidth, 3))
        
        for i in range(1, n):
            image = integrator.renderC(scene, sensor_id)
            out += image.numpy().reshape((scene.opts.cropheight, scene.opts.cropwidth, 3))
            
        out *= weight
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out

    def compute_render_loss(our_img1, our_img2, ref_img, weight,rgb=0,weight_map=None):
        size = our_img1.numpy().shape
        size = (int(np.sqrt(size[0])),int(np.sqrt(size[0])))
        if weight_map is None:
            weight_map = FloatD(1.0)
        loss = 0
        if rgb == 0:
            for i in range(3): 
                I1_ = our_img1[i]
                I2_ = our_img2[i]
                T_ = ref_img[i] 
                diff1 = ek.hmean((I1_ - T_) * (I2_ - T_))
                loss += ek.hmean(diff1) / 3.0 # + ek.hmean(diff2) / 3.0
        else:
            i = rgb-1
            I1_ = our_img1[i] / 3.0
            I2_ = our_img2[i] / 3.0
            T_ = ref_img[i] 
            # tmp = T_.numpy().reshape((512,512,-1)) * 255
            # tmp2 = I1_.numpy().reshape((512,512,-1)) * 255
            # cv2.imwrite('test1.png',tmp)
            # cv2.imwrite('test2.png',tmp2)
            # breakpoint()
            # print(I1_.numpy().max())
            # print(T_.numpy().max())
            diff1 = (I1_ - T_) * (I2_ - T_)
            
            # Clipping
            # I1 = ek.select(I1_ > 1, 1.0, I1_)
            # I2 = ek.select(I2_ > 1, 1.0, I2_)
            # T = ek.select(T_ > 1, 1.0, T_) 
            # enoki2img(I1,filename='img_out.png',shape=size)
            # enoki2img(T,filename='img_gt.png',shape=size)
            # breakpoint()
            # diff1 = (I1 - T) * (I2 - T)
            diff1 = diff1 * weight_map

            loss = ek.hmean(diff1)
            
        return loss * weight

    def compute_silheuette_loss(render, reference, weight):
        loss = 0
        for i in range(1):
            I1_ = render[i]
            T_ = reference[i] 
            diff = ek.hmean((I1_ - T_) * (I1_ - T_))
            loss += ek.hmean(diff)
        return loss * weight

    def total_variation_loss(img, weight, width, chanel):
        print(img.shape)
        w_variance = torch.mean(torch.pow(img.reshape((width, width, chanel))[:-1, :, :] - img.reshape((width, width, chanel))[1:,:,:], 2))
        h_variance = torch.mean(torch.pow(img.reshape((width, width, chanel))[:,:-1,:] - img.reshape((width, width, chanel))[:,1:,:], 2))
        loss = weight * (h_variance + w_variance)
        return loss

    def compute_FD(A,S,sensor_id,delta=1.0,poly_fit=False,chunk=False):
        # Render image
        seed = 2
        npixels = ro.cropheight * ro.cropwidth
        sc.opts.spp = 1
        # set to 1 for fixing random seed
        sc.opts.debug = 0
        sc.setseed(seed*npixels)
        sc.setlightposition(Vector3fD(lights[sensor_id][0], lights[sensor_id][1], lights[sensor_id][2]))
        if not chunk:
            sc.opts.crop_offset_x = 0
            sc.opts.crop_offset_y = 0
            sc.opts.cropheight = sc.opts.height
            sc.opts.cropwidth = sc.opts.width
        if poly_fit and not isBaseline:
            precompute_mesh_polys()

        # Create single differentiable variable
        if A is not None:
            sc.param_map[material_key].albedo.data   = A + delta #a
        elif S is not None:
            sc.param_map[material_key].sigma_t.data   = S + delta #a
        else:
            raise NotImplementedError
        sc.configure()

        img_pos = renderNtimes(sc, myIntegrator, args.ref_spp, sensor_id)
        
        # Create single differentiable variable
        if A is not None:
            sc.param_map[material_key].albedo.data   = A - delta #a
        elif S is not None:
            sc.param_map[material_key].sigma_t.data   = S - delta #a
        sc.configure()
        img_neg = renderNtimes(sc, myIntegrator, args.ref_spp, sensor_id)
        
        grad = (img_pos - img_neg) / (2*delta) 

        return grad 


    def compute_forward_derivative(A,S,sensor_id,idx_param=0,FD=False,poly_fit=False,chunk=False):

        if poly_fit and not isBaseline:
            precompute_mesh_polys()
        seed = 2
        # Create single differentiable variable
        if A is not None:
            albedo     = Vector3fD(A)
            a = albedo
            ek.set_requires_gradient(a,True)
            sc.param_map[material_key].albedo.data   = a
        elif S is not None:
            sigma_t    = Vector3fD(S)
            a = sigma_t
            ek.set_requires_gradient(a,True)
            sc.param_map[material_key].sigma_t.data   = a
        else:
            raise NotImplementedError

        # Render image
        npixels = ro.cropheight * ro.cropwidth
        sc.opts.spp = args.spp if not FD else 1
        sc.setseed(seed*npixels)
        sc.setlightposition(Vector3fD(lights[sensor_id][0], lights[sensor_id][1], lights[sensor_id][2]))
        # sc.opts.debug = 1 # Fix random seed for comparison w. FD
        sc.configure()

        if not FD:
            grad_imgs= []
            if chunk:
                img = myIntegrator.renderD(sc, sensor_id)
                grad_fors = []
                for i in range(3):
                    ek.forward(a[i],free_graph=False)
                    try:
                        grad_for = ek.gradient(img[i])
                        grad_fors.append(grad_for.numpy().reshape(sc.opts.cropheight,sc.opts.cropwidth,-1))
                    except:
                        grad_fors.append(np.zeros((sc.opts.cropheight,sc.opts.cropwidth,1)))
                grad_img = np.concatenate(grad_fors,axis=-1)#.reshape(sc.opts.cropheight,sc.opts.cropwidth,-1)
                grad_img *= 3 # n_channel
                
                if grad_img.shape[-1] == 0:
                    grad_img = np.zeros((sc.opts.cropheight,sc.opts.cropwidth,3))
                grad_imgs.append(grad_img)
            else:
                for batch_idx in range(args.n_crops*args.n_crops):
                    ix = batch_idx // args.n_crops
                    iy = batch_idx % args.n_crops
                    
                    # image_loss = renderV, A, S, R, G, render_batch, i)
                    sc.opts.crop_offset_x = ix * sc.opts.cropwidth
                    sc.opts.crop_offset_y = iy * sc.opts.cropheight
                    sc.configure()

                    img = myIntegrator.renderD(sc, sensor_id)
                    # ek.forward(a)
                    grad_fors = []
                    for i in range(3):
                        ek.forward(a[i],free_graph=False)
                        try:
                            grad_for = ek.gradient(img[i])
                            grad_fors.append(grad_for.numpy().reshape(sc.opts.cropheight,sc.opts.cropwidth,-1))
                        except:
                            print(f"Zero gradient for {batch_idx}")
                            grad_fors.append(np.zeros((sc.opts.cropheight,sc.opts.cropwidth,1)))
                    grad_img = np.concatenate(grad_fors,axis=-1)#.reshape(sc.opts.cropheight,sc.opts.cropwidth,-1)
                    grad_img *= 3 # n_channel
                   
                    if grad_img.shape[-1] == 0:
                        grad_img = np.zeros((sc.opts.cropheight,sc.opts.cropwidth,3))
                    grad_imgs.append(grad_img)

            # (16,w,h) in column major order
            img = np.array(grad_imgs)
            # (4,w,4,h)
            if chunk:
                img = img.reshape((sc.opts.cropheight,sc.opts.cropwidth,-1))
            else:
                img = img.reshape((args.n_crops,args.n_crops,sc.opts.cropheight,sc.opts.cropwidth,-1))
                img = img.transpose((1,0,2,3,4))
                # (4,4,w,h) 
                img = img.transpose((0,2,1,3,4))
                img = img.reshape((sc.opts.height,sc.opts.width,-1))
            # (4w, 4h)
            return img
        else:
            if not chunk:
                sc.opts.crop_offset_x = 0
                sc.opts.crop_offset_y = 0
                sc.opts.cropheight = sc.opts.height
                sc.opts.cropwidth = sc.opts.width
            sc.configure()
            img = renderNtimes(sc, myIntegrator, args.ref_spp, sensor_id)
            return img 
        
    # isMonochrome = sc.param_map[material_key].monochrome
    class Renderer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, V, A, S, R, G, E, batch_size, seed, M, crop_idx=0,):
            # Roughness = R
            _vertex     = Vector3fD(V)
            _albedo     = Vector3fD(A)
            _sigma_t    = Vector3fD(S)
            _rough      = FloatD(R)
            _eta        = FloatD(G)
            _epsM       = Vector3fD(E)

            ek.set_requires_gradient(_vertex,       V.requires_grad)
            ek.set_requires_gradient(_albedo,       A.requires_grad)
            ek.set_requires_gradient(_sigma_t,      S.requires_grad)
            ek.set_requires_gradient(_rough,        R.requires_grad)
            ek.set_requires_gradient(_eta,          G.requires_grad)
            ek.set_requires_gradient(_epsM,         E.requires_grad)

        
            ctx.input1 = _vertex
            ctx.input2 = _albedo
            ctx.input3 = _sigma_t
            ctx.input4 = _rough
            ctx.input5 = _eta
            ctx.input6 = _epsM

        
            albedo  = ek.select(_albedo     > 0.99995,     0.99995, ek.select(_albedo < 0.0, 0.0, _albedo))
            sigma_t = ek.select(_sigma_t    < 0.0,      0.01,  _sigma_t)
            roughness = ek.select(_rough    < 0.01,      0.01,  _rough)
            eta       = ek.select(_eta      < 1.0,       1.0,   _eta)
            epsM       = ek.select(_epsM      < 0.0 ,       0.01,   ek.select(_epsM > 10.0, 10.0,_epsM))


            sc.param_map[mesh_key].vertex_positions  = _vertex
            sc.param_map[material_key].albedo.data   = albedo
            sc.param_map[material_key].sigma_t.data  = sigma_t
            sc.param_map[material_key].alpha_u.data  = roughness
            sc.param_map[material_key].eta.data      = eta
            sc.param_map[material_key].epsM.data      = epsM

            if not isBaseline:
                ctx.mono = sc.param_map[material_key].monochrome
            else:
                ctx.mono = True

            print("------------------------------------seed----------",seed)
            npixels = ro.cropheight * ro.cropwidth
            sc.opts.spp = args.spp
            sc.setseed(seed*npixels)
            
            # isFD =  sc.opts.sppse < 0
            # naiveFD =  sc.opts.sppse < 0
            
            # Random select channel
            if isBaseline:
                sc.opts.rgb = 0
            else:
                sc.opts.rgb = 0 # blue only
            sc.opts.mode = shadeMode
            sc.opts.vaeMode = args.vaeMode
            sc.opts.isFD = args.isFD
            sc.opts.maxDistScale = args.maxDistScale
            sc.configure()
            
            render_loss= 0    
            
            sensor_indices = active_sensors(batch_size, num_sensors)
            ctx.sensor_indices = sensor_indices
            # print("sensor indices: ", sensor_indices)
            for sensor_id in sensor_indices:
                sc.setlightposition(Vector3fD(lights[sensor_id][0], lights[sensor_id][1], lights[sensor_id][2]))

                cox,coy,coh,cow = ro.crop_offset_x,ro.crop_offset_y,ro.cropheight,ro.cropwidth
                tar_img = Vector3fD(tars[sensor_id].reshape((ro.height,ro.width,-1))[coy:coy+coh,cox:cox+cow,:].reshape(-1,3).cuda())
                our_imgA = myIntegrator.renderD(sc, sensor_id)
                if isBaseline:
                    our_imgB = myIntegrator.renderD(sc, sensor_id)
                else:
                    # our_imgB = myIntegrator.renderD(sc, sensor_id)
                    our_imgB = our_imgA


                render_loss += compute_render_loss(our_imgA, our_imgB, tar_img, args.img_weight,sc.opts.rgb) / batch_size
                # if naiveFD:
                #     fd_delta= 1
                #     param_delta = torch.ones(3).cuda()
                #     param_delta= param_delta *fd_delta
                #     fd = compute_FD(None,S,sensor_id,fd_delta,chunk=True)
                #     ctx.d_pixel = fd.reshape(-1,3)
                #     ctx.diff = our_imgA - tar_img

            if args.silhouette == "yes":
                sc.opts.spp = 1
                sc.configure()
                for sensor_id in sensor_indices:
                    silhouette = silhouetteIntegrator.renderD(sc, sensor_id)
                    ref_sil = Vector3fD(maks[sensor_id].cuda())
                    render_loss += compute_silheuette_loss(silhouette, ref_sil, args.img_weight) / batch_size                

            ctx.output = render_loss
            out_torch = ctx.output.torch()

            return out_torch 

        @staticmethod
        def backward(ctx, grad_out):
            try:
                ek.set_gradient(ctx.output, FloatC(grad_out))
            except:
                result = (None, None, None, None, None, None, None, None, None)
        
            # print("--------------------------------------------------------")
            FloatD.backward()
            # print("-------------------------V-----------------------------")
            # gradV = ek.gradient(ctx.input1).torch()
            # gradV[torch.isnan(gradV)] = 0.0
            # gradV[torch.isinf(gradV)] = 0.0
            gradV = None
            # print("-------------------------A-----------------------------")
            gradA = ek.gradient(ctx.input2).torch()
            gradA[torch.isnan(gradA)] = 0.0
            gradA[torch.isinf(gradA)] = 0.0
            # FIXME: fix albedo 
            # gradA = None
            
            if not isBaseline and ctx.mono:
                gradA[...,:] = torch.mean(gradA,dim=-1)
            else:
                print("shold not be happening! Not monochorme")
            # gradA= None
            # result = compute_forward_derivative(A=A, S=S, sensor_id=sensor_id,idx_param=idx_param,poly_fit=True)
            # print("-------------------------S-----------------------------")
            # naiveFD = sc.opts.sppse < 0
            # if naiveFD:
            #     gradS = torch.from_numpy(ctx.d_pixel * 2*(ctx.diff)).cuda().mean(dim=0,keepdim=True)
            # else:
            #     gradS = ek.gradient(ctx.input3).torch()
            gradS = ek.gradient(ctx.input3).torch()
            gradS[torch.isnan(gradS)] = 0.0
            gradS[torch.isinf(gradS)] = 0.0
            if not isBaseline and ctx.mono:
                gradS[...,:] = torch.mean(gradS,dim=-1,)

            # print("-------------------------R-----------------------------")
            gradR = ek.gradient(ctx.input4).torch()
            gradR[torch.isnan(gradR)] = 0.0
            gradR[torch.isinf(gradR)] = 0.0
            # print("-------------------------G-----------------------------")
            gradG = ek.gradient(ctx.input5).torch()
            gradG[torch.isnan(gradG)] = 0.0 
            gradG[torch.isinf(gradG)] = 0.0

            # print("-------------------------E-----------------------------")
            gradE = ek.gradient(ctx.input6).torch()
            gradE[torch.isnan(gradE)] = 0.0
            gradE[torch.isinf(gradE)] = 0.0
            if not isBaseline and ctx.mono:
                gradE[...,:] = torch.mean(gradE,dim=-1,keepdim=True)

            result = (gradV, gradA, gradS, gradR, gradG, gradE, None,None,None)
            del ctx.output, ctx.input1, ctx.input2, ctx.input3, ctx.input4, ctx.input5, ctx.input6
            print("==========================")
            print("Estimated gradients")
            print("gradA ",gradA,"gradS ",gradS,"gradV ", gradV)
            # assert(torch.all(gradA==0.0))
            # assert(torch.all(gradS == 0.0))
            return result

    
    render = Renderer.apply
    render_batch = 1
    # render_batch = 5
    # render_batch = 6

    def resize(MyTexture, texturesize, textstr, i, channel=3):
        TextureMap = MyTexture.detach().cpu().numpy().reshape((texturesize, texturesize, channel))
        if texturesize < 512:
            texturesize = texturesize * 2
            TextureMap = cv2.cvtColor(TextureMap, cv2.COLOR_RGB2BGR)
            TextureMap = cv2.resize(TextureMap, (texturesize, texturesize))
            cv2.imwrite(statsdir + "/{}_resize_{}.exr".format(textstr, i+1), TextureMap)
        return texturesize

    def renderPreview(i, sidxs):
        # cmap = colormaps.get('inferno')
        images = []
        rmse_list = []
        # ssim_list = []
        
        for idx in sidxs:
            sc.setlightposition(Vector3fD(lights[idx][0], lights[idx][1], lights[idx][2]))
            img2 = renderNtimes(sc, myIntegrator, args.ref_spp, idx)
            
            target2 = tars[idx].numpy().reshape((ro.height,ro.width,-1)) #[coy:coy+coh,cox:cox+cow]
            target2 = cv2.cvtColor(target2, cv2.COLOR_RGB2BGR)
            target2 = np.clip(target2,a_min=0.0, a_max=1.0)
            # cv2.imwrite(statsdir + "/iter_{}_{}_out.png".format(i+1,idx), img2*255.0)
            # cv2.imwrite(statsdir + "/iter_{}_{}_gt.png".format(i+1,idx), target2*255.0)
            cv2.imwrite(statsdir + "/iter_{}_{}_out.exr".format(i+1,idx), img2)
            cv2.imwrite(statsdir + "/iter_{}_{}_gt.exr".format(i+1,idx), target2)
            absdiff2 = np.abs(img2 - target2)

            output2 = np.concatenate((img2, target2, absdiff2))
            images.append(output2)
                                
            # # Debug: RMSE image
            # rmse2 = rmse(img2,target2)
            # plt.imshow(rmse2, cmap='inferno')
            # plt.tight_layout()
            # plt.colorbar()
            # plt.axis('off')
            # outfile_error = statsdir + "/error_{}_{}.png".format(i+1,idx)
            # plt.savefig(outfile_error,bbox_inches='tight')
            # plt.close()
            
            
            # X: (N,3,H,W`) a batch of non-negative RGB images (0~255)
            # Y: (N,3,H,W)  
            # X = torch.from_numpy(img2.transpose(2,0,1)[None]) #torch.zeros((1,3,256,256))
            # Y = torch.from_numpy(target2.transpose(2,0,1)[None]) #torch.zeros((1,3,256,256))
            # calculate ssim & ms-ssim for each image
            # ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
            # wandb_log({
            #     "images/gt": wandb.Image(cv2.cvtColor(target2,cv2.COLOR_BGR2RGB)),
            #     "images/out" : wandb.Image(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)),
            #     "loss/rmse_image": rmse(img2,target2).mean(),
            #     "loss/ssim_image": ssim_val,
            # })
            
            rmse_list.append(rmse(img2,target2).mean())
            # ssim_list.append(ssim_val)

        output = np.concatenate((images), axis=1)                
        cv2.imwrite(statsdir + "/iter_{}.exr".format(i+1), output)
        rmse_avg_img = sum(rmse_list) / len(rmse_list)
        wandb_log({
            "loss/rmse_avg_image": rmse_avg_img, #rmse(img2,target2).mean(),
            # "loss/ssim_avg_image": sum(ssim_list) / len(rmse_list), #rmse(img2,target2).mean(),
        },step=i+1)
        return rmse_avg_img

    def wandb_log(*wandb_args,**kwargs):
        if not args.debug:
            wandb.log(*wandb_args,**kwargs)

    def optTask():

        rmsehistory = []
        losshistory = [] 
        albedohistory = []
        sigmahistory = []
        roughhistory = []
        etahistory = []

        alb_texture_width = args.albedo_texture
        sig_texture_width = args.sigma_texture
        rgh_texture_width = args.rough_texture
        epsM_texture_width = args.epsM_texture
        
        sc.opts.isFull = 1
        sc.opts.vaeMode = args.vaeMode
        sc.configure()

        
        S = Variable(sc.param_map[material_key].sigma_t.data.torch() / scale_sig, requires_grad=True)
        # S = Variable(torch.log(sc.param_map[material_key].sigma_t.data.torch()), requires_grad=True)
        A = Variable(sc.param_map[material_key].albedo.data.torch(), requires_grad=True)
        R = Variable(sc.param_map[material_key].alpha_u.data.torch(), requires_grad=True)
        G = Variable(sc.param_map[material_key].eta.data.torch(), requires_grad=True)
        E = Variable(sc.param_map[material_key].epsM.data.torch(), requires_grad=True)

        V = Variable(sc.param_map[mesh_key].vertex_positions.torch(), requires_grad=not(args.mesh_lr==0))
        F = sc.param_map[mesh_key].face_indices.torch().long()
        M = compute_matrix(V, F, lambda_ = args.laplacian)
        u = to_differential(M,V)
        U = Variable(u, requires_grad=not(args.mesh_lr==0))

        def saveHistory(filedir):
            np.save(filedir+"/loss.npy", np.concatenate(losshistory, axis=0))
            # np.save(filedir+"/parameters_rmse.npy", np.concatenate(rmsehistory, axis=0))
            if len(sigmahistory) > 0:
                np.save(filedir+"/sigmas.npy", np.concatenate(sigmahistory, axis=0))
            if len(albedohistory) > 0:
                np.save(filedir+"/albedo.npy", np.concatenate(albedohistory, axis=0))
            if len(roughhistory) > 0:
                np.save(filedir+"/rough.npy", np.concatenate(roughhistory, axis=0))
            if len(etahistory) > 0:
                np.save(filedir+"/eta.npy", np.concatenate(etahistory, axis=0))
        
        def init_optimizer(U,M,A,S,R,G,
                           alb_texture_width,
                           sig_texture_width,
                           rgh_texture_width,):
            params = []
            params.append({'params': U, 'lr': args.mesh_lr, "I_Ls": M, 'largestep': True})
            # params.append({'params': V, 'lr': args.mesh_lr, "I_Ls": M, 'largestep': True})
                        
            if  args.albedo_texture > 0 and args.albedo_laplacian > 0:
                AM = compute_image_matrix(alb_texture_width, args.albedo_laplacian)
                params.append({'params': A, 'lr': args.albedo_lr, "I_Ls": AM, 'largestep': True})
            else:
                params.append({'params': A, 'lr': args.albedo_lr})

            if  args.sigma_texture > 0 and args.sigma_laplacian > 0:
                SM = compute_image_matrix(sig_texture_width, args.sigma_laplacian)
                params.append({'params': S, 'lr': args.sigma_lr, "I_Ls": SM, 'largestep': True})
            else:
                params.append({'params': S, 'lr': args.sigma_lr})

            if args.rough_texture > 0 and args.rough_laplacian > 0:
                RM = compute_image_matrix(rgh_texture_width, args.rough_laplacian)
                params.append({'params': R, 'lr': args.rough_lr, "I_LS": RM, 'largestep': True})
            else:
                params.append({'params': R, 'lr': args.rough_lr})

            params.append({'params': G, 'lr': args.eta_lr})
                
            optimizer = UAdam(params)
            return optimizer

        optimizer = init_optimizer(U,M,A,S,R,G,
                           alb_texture_width,
                           sig_texture_width,
                           rgh_texture_width,)
        param_alb = A.detach().cpu().numpy()[0]
        param_sig = S.detach().cpu().numpy()[0]
        wandb_log({
            "loss/rmse_param_alb": rmse(param_alb,params_gt[args.scene]['albedo']),
            "loss/rmse_param_sig": rmse(param_sig, np.array(params_gt[args.scene]['sigmat'])/scale_sig),
            # "loss/rmse_param_sig": rmse(param_sig * scale_sig, params_gt[args.scene]['sigmat']),
            # "loss/rmse_param_sigSCALED": rmse(param_sig, params_gt[args.scene]['sigmat'] / scale_sig),

            # "param/eta": G.detach().cpu().numpy(),
            # "param/rough": R.detach().cpu().numpy(),
            "param/albedo_r" : param_alb[0],
            "param/albedo_g" : param_alb[1],
            "param/albedo_b" : param_alb[2],
            "param/sigmaT_r" : param_sig[0] * scale_sig,
            "param/sigmaT_g" : param_sig[1] * scale_sig,
            "param/sigmaT_b" : param_sig[2] * scale_sig,
        },step=0)

        for i in range(args.n_iters):
            loss = 0
            image_loss = 0
            range_loss = 0
            sc.opts.cropheight = sc.opts.height // args.n_crops
            sc.opts.cropwidth = sc.opts.width // args.n_crops
            batch_idxs = torch.randperm(args.n_crops*args.n_crops)
            V = from_differential(M,U)
            
            n_crops = 0
            optimizer.zero_grad()
            gradS = torch.zeros((1,3)).cuda()

            list_times = []
            start = time.time() 
            for batch_idx in batch_idxs:
                
                ix = batch_idx // args.n_crops
                iy = batch_idx % args.n_crops
                
                if args.n_crops > 2:
                    if ix ==0 or ix == args.n_crops -1:
                    # if ix < (args.n_crops //4) or ix >= (args.n_crops // 4 * 3):
                        print(f"idx: {ix}, skipping")
                        continue
                    if iy ==0 or iy == args.n_crops -1:
                    # if iy < (args.n_crops //4) or iy >= (args.n_crops // 4 * 3):
                        print(f"idx: {iy}, skipping")
                        continue
                
                # image_loss = renderV, A, S, R, G, render_batch, i)
                sc.opts.crop_offset_x = ix * sc.opts.cropwidth
                sc.opts.crop_offset_y = iy * sc.opts.cropheight
                image_loss_ = render(V, A, S * scale_sig, R, G, E, render_batch, i,M)
                # image_loss_ = render(V, A, torch.exp(S), R, G, E, render_batch, i,M)
                range_loss_ = texture_range_loss(A, S, R, G, args.range_weight)
                loss_ = image_loss_ + range_loss_

                # total variation loss_
                if args.albedo_texture > 0:
                    loss_ += total_variation_loss(A, args.tot_weight, args.albedo_texture, 3)
                if args.sigma_texture > 0:
                    print(S.shape)
                    loss_ += total_variation_loss(S, args.tot_weight,  args.sigma_texture, 3)
                if args.rough_texture > 0:
                    print(R.shape)
                    loss_ += total_variation_loss(R, args.tot_weight, args.rough_texture, 1)
                loss_.backward()

                loss += loss_.item()
                image_loss += image_loss_.item()
                range_loss += range_loss_.item()
                    
                # del loss_, range_loss_, image_loss_
                n_crops += 1
                
            optimizer.step()
            optimizer.zero_grad()

            loss /= n_crops
            image_loss /= n_crops
            range_loss /= n_crops

        
            # Time Measurement
            end = time.time()
            elapsed_time = end - start
            list_times.append(elapsed_time)
            avg_time = sum(list_times) / len(list_times)
            print(f"average time: {avg_time}")
            
            print("------ Iteration ---- ", i, ' image loss: ', loss, ' eta: ', G.detach().cpu().numpy())   
            
            if args.albedo_texture == 0:
                print("\n albedo: ", A.detach().cpu().numpy()) 
                albedohistory.append([A.detach().cpu().numpy()])
            if args.sigma_texture == 0:
                print("\n sigma: ", S.detach().cpu().numpy())
                sigmahistory.append([S.detach().cpu().numpy()])   
            if args.rough_texture == 0:
                print("\n rough: ", R.detach().cpu().numpy())
                roughhistory.append(R.detach().cpu().numpy())    
            param_alb = A.detach().cpu().numpy()[0]
            param_sig = S.detach().cpu().numpy()[0]

            wandb_log({
                "time/periter": elapsed_time,
                "time/average" : avg_time,

                "loss/train_loss": loss,
                "loss/range_loss": range_loss,
                "loss/image_loss": image_loss,
                "loss/rmse_param_alb": rmse(param_alb,params_gt[args.scene]['albedo']),
                "loss/rmse_param_sig": rmse(param_sig, np.array(params_gt[args.scene]['sigmat'])/scale_sig),
                # "loss/rmse_param_sig": rmse(param_sig * scale_sig, params_gt[args.scene]['sigmat']),
                # "loss/rmse_param_sigSCALED": rmse(param_sig, params_gt[args.scene]['sigmat'] / scale_sig),

                # "param/eta": G.detach().cpu().numpy(),
                # "param/rough": R.detach().cpu().numpy(),
                "param/albedo_r" : param_alb[0],
                "param/albedo_g" : param_alb[1],
                "param/albedo_b" : param_alb[2],
                "param/sigmaT_r" : param_sig[0] * scale_sig,
                "param/sigmaT_g" : param_sig[1] * scale_sig,
                "param/sigmaT_b" : param_sig[2] * scale_sig,
            },step=i+1)

            etahistory.append([G.detach().cpu().numpy()])
            losshistory.append([loss])
            
            if ((i+1) % args.n_reduce_step) == 0:
                print('reduce n_step')
                lrs = []
                args.albedo_lr = args.albedo_lr * 0.85
                args.sigma_lr = args.sigma_lr * 0.85
                args.rough_lr = args.rough_lr * 0.95
                args.epsM_lr = args.epsM_lr * 0.95
                args.eta_lr = args.eta_lr * 0.95
                lrs.append(args.mesh_lr)
                lrs.append(args.albedo_lr)
                lrs.append(args.sigma_lr)
                lrs.append(args.rough_lr)
                lrs.append(args.epsM_lr)
                lrs.append(args.eta_lr)

                optimizer.setLearningRate(lrs)
            computed_poly = False

            torch.cuda.empty_cache()
            ek.cuda_malloc_trim()

            if (i % args.n_dump == args.n_dump -1) or (i==10):
            # if (i % args.n_dump == 0):
                sc.opts.cropheight = sc.opts.height
                sc.opts.cropwidth = sc.opts.width
                sc.opts.crop_offset_x = 0 
                sc.opts.crop_offset_y = 0 
                sc.opts.spp = 1
                sc.opts.rgb = 0
                sc.opts.mode = shadeMode
                sc.opts.isFD = 0
                sc.configure()

                print(f"[DEBUG] Rendering selected images from training set")
                renderPreview(i, np.array([0, 1, 4, 10, 19], dtype=np.int32))
        
            if ((i + 1) % args.n_vaeModeChange) == 0:
                args.vaeMode = 0
            
            if ((i+1) % args.n_fitpoly) == 0:
                if not isBaseline:
                    precompute_mesh_polys()

            torch.cuda.empty_cache()
            ek.cuda_malloc_trim()

    if args.render_results:
        sc.opts.isFull = 1
        sc.opts.vaeMode = args.vaeMode
        sc.opts.cropheight = sc.opts.height
        sc.opts.cropwidth = sc.opts.width
        sc.opts.crop_offset_x = 0 
        sc.opts.crop_offset_y = 0 
        sc.opts.spp = 1
        sc.opts.rgb = 0
        sc.opts.mode = shadeMode
        sc.opts.isFD = 0
        sc.configure()

        print(f"[DEBUG] Rendering selected images from training set")
        renderPreview(0, np.array([1,], dtype=np.int32))
        # renderPreview(0, np.array([0, 1, 4, 10, 19], dtype=np.int32))

    elif args.render_gradient:


        fd_delta = 5e-2 #2
        fd_delta = 5
        # fd_delta = 2
        sensor_id = args.sensor_id #30 #0
        idx_param = 2 #0
        param_delta = torch.ones(3).cuda()
        param_delta = torch.zeros(3).cuda()
        param_delta[idx_param] = 1.0
        param_delta = param_delta * fd_delta
        # param_delta[idx_param] = fd_delta
        isAlbedo = True
        isAlbedo = False
        sc.opts.rgb = 0
        # sc.opts.rgb = idx_param + 1
        sc.opts.debug = 1 # Fix random seed for comparison w. FD
        # sc.opts.debug = 0 # Fix random seed for comparison w. FD
        sc.opts.vaeMode = args.vaeMode
        sc.opts.isFD = args.isFD
        sc.configure()
        
        GRAD_DIR = f"{statsdir}/grad" #"../../grad2"
        grad_dir = GRAD_DIR
        os.makedirs(GRAD_DIR,exist_ok=True)

        
        A,S = None,None
        if isAlbedo:
            A = Variable(sc.param_map[material_key].albedo.data.torch(), requires_grad=True)
            exp_name = f"sensor{sensor_id}_albedo{sc.param_map[material_key].albedo.data[idx_param][0]}_{sc.param_map[material_key].type_name()}_delta{fd_delta}"
        else:
            S = Variable(sc.param_map[material_key].sigma_t.data.torch(), requires_grad=True)
            exp_name = f"sensor{sensor_id}_sigmaT{sc.param_map[material_key].albedo.data[idx_param][0]}_{sc.param_map[material_key].type_name()}_delta{fd_delta}"

        filename = os.path.join(GRAD_DIR,f"{exp_name}_Ours.png")

        # Compute Image Derivatives for validation
        reset_random()
        result = compute_forward_derivative(A=A, S=S, sensor_id=sensor_id,idx_param=idx_param,poly_fit=True)
        try: 
            img = np.mean(result,axis=-1)
        except:
            img = np.mean(result.numpy(),axis=-1)
            
        print(f"Number of nonzero pixels: {np.count_nonzero(img)}")
        fname=f"{grad_dir}/{exp_name}_Ours"
        np.save(fname,img)
        # print(f"Write gradient image to {fname}.npy")
        
        # Gradient estimate using finite differences
        filename = filename.replace("Ours","FD",)

        A,S = None, None
        if isAlbedo:
            A0 = Variable(sc.param_map[material_key].albedo.data.torch() - param_delta, requires_grad=True)
            A1 = Variable(sc.param_map[material_key].albedo.data.torch() + param_delta, requires_grad=True)
            result0 = compute_forward_derivative(A0,S=S,sensor_id=sensor_id,idx_param=idx_param,FD=True)
            # fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_0"
            # np.save(fname,result0.mean(axis=-1))
            # print(f"Write gradient image to {fname}")
            # img0 = result0 * 255.0
            # cv2.imwrite(f"{filename.replace('FD','FD_0')}",img0)
            result1 = compute_forward_derivative(A1,S=S,sensor_id=sensor_id,idx_param=idx_param,FD=True)
            # fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_1"
            # np.save(fname,result1.mean(axis=-1))
            # print(f"Write gradient image to {fname}")
            # img1 = result1 * 255.0
            # cv2.imwrite(f"{filename.replace('FD','FD_1')}",img1)
        
        else:
            S0 = Variable(sc.param_map[material_key].sigma_t.data.torch() - param_delta, requires_grad=True)
            reset_random()
            result0 = compute_forward_derivative(A,S=S0,sensor_id=sensor_id,idx_param=idx_param,FD=True)
            # fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_0"
            # np.save(fname,result0.mean(axis=-1))
            # print(f"Write gradient image to {fname}")
            # img0 = result0 * 255.0
            # cv2.imwrite(f"{filename.replace('FD','FD_0')}",img0)
            S1 = Variable(sc.param_map[material_key].sigma_t.data.torch() + 2*param_delta , requires_grad=True)
            reset_random()
            result1 = compute_forward_derivative(A,S=S1,sensor_id=sensor_id,idx_param=idx_param,FD=True)
            # fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_1"
            # np.save(fname,result1.mean(axis=-1))
            # print(f"Write gradient image to {fname}")
            # img1 = result1 * 255.0
            # cv2.imwrite(f"{filename.replace('FD','FD_1')}",img1)
        
        breakpoint()
        result_fd = (result1[...,idx_param] - result0[...,idx_param]) / (2*fd_delta)
        # img = np.mean(result_fd,axis=-1)
        fname=f"{grad_dir}/{exp_name}_FD"
        np.save(fname,result_fd)
        print(f"Write gradient image to {fname}")

        # # Joon added: mode for gradient evaluation
        # mode = 0 # fd 0
        # mode = 1 # fd 1
        # mode = 2 # diff
        # mode = 3 # ours (img derivative)
        # mode = 4 # all
        
        # # grad_dir = statsdir
        # GRAD_DIR = f"{statsdir}/grad" #"../../grad2"
        # grad_dir = GRAD_DIR
        
        # os.makedirs(GRAD_DIR,exist_ok=True)
        
        # A,S = None,None
        # if isAlbedo:
        #     A = Variable(sc.param_map[material_key].albedo.data.torch(), requires_grad=True)
        #     filename = os.path.join(GRAD_DIR,f"sensor{sensor_id}_albedo{sc.param_map[material_key].albedo.data[idx_param]}_{sc.param_map[material_key].type_name()}_delta{fd_delta}_Ours.png")
        # else:
        #     S = Variable(sc.param_map[material_key].sigma_t.data.torch(), requires_grad=True)
        #     filename = os.path.join(GRAD_DIR,f"sensor{sensor_id}_sigmaT{sc.param_map[material_key].sigma_t.data[idx_param]}_{sc.param_map[material_key].type_name()}_delta{fd_delta}_Ours.png")

        # if mode >= 3:
            
        #     reset_random()
        #     result = compute_forward_derivative(A=A, S=S, sensor_id=sensor_id,idx_param=idx_param,poly_fit=True)
        #     # img = result[...,idx_param]
        #     # img = torch.mean(result,dim=-1)
        #     try: 
        #         img = np.mean(result,axis=-1)
        #     except:
        #         img = np.mean(result.numpy(),axis=-1)
                
        #     print(f"Number of nonzero pixels: {np.count_nonzero(img)}")
        #     fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}"
        #     np.save(fname,img)
        #     print(f"Write gradient image to {fname}")

        #     cmax = max(img.max(),abs(img.min()))
        #     norm = MidpointNormalize(vmin=-cmax,vmax=cmax,midpoint=0.0)
        #     plt.imshow(img, cmap='RdBu_r',norm=norm)
        #     plt.tight_layout()
        #     # plt.colorbar()
        #     plt.axis('off')
        #     print(f"Write gradient image to {filename}")
        #     plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        #     plt.close()

        #     cmap_key='RdBu_r'
        #     # Apply normalization and cmap to image
        #     extent = (0,img.shape[0],img.shape[1],0)
        #     # filename = filename.replace('.png','_nolegend.png')
        #     mapper = cm.ScalarMappable(norm=norm, cmap=cmap_key)
        #     img_ours = np.copy(mapper.to_rgba(img))#.clone()
        #     plt.tight_layout()
        #     # plt.colorbar()
        #     plt.axis('off')
        #     plt.imshow(img_ours,extent=extent)
        #     plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        #     plt.close()
        #     if mode != 4:
        #         return

        
        # filename = filename.replace("_nolegend.png",".png")
        # filename = filename.replace("deriv","FD",)
        # ## Gradient estimate using finite differences
        # # if os.path.exists(filename):
        # #     return

        # A,S = None, None
        # if isAlbedo:
        #     A0 = Variable(sc.param_map[material_key].albedo.data.torch() - param_delta, requires_grad=True)
        #     A1 = Variable(sc.param_map[material_key].albedo.data.torch() + param_delta, requires_grad=True)
        #     result0 = compute_forward_derivative(A0,S=S,sensor_id=sensor_id,idx_param=idx_param,FD=True)
        #     fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_0"
        #     np.save(fname,result0.mean(axis=-1))
        #     print(f"Write gradient image to {fname}")
        #     img0 = result0 * 255.0
        #     cv2.imwrite(f"{filename.replace('FD','FD_0')}",img0)
        #     result1 = compute_forward_derivative(A1,S=S,sensor_id=sensor_id,idx_param=idx_param,FD=True)
        #     fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_1"
        #     np.save(fname,result1.mean(axis=-1))
        #     print(f"Write gradient image to {fname}")
        #     img1 = result1 * 255.0
        #     cv2.imwrite(f"{filename.replace('FD','FD_1')}",img1)
        
        # else:
        #     if mode != 1:
        #         S0 = Variable(sc.param_map[material_key].sigma_t.data.torch() - param_delta, requires_grad=True)
        #         reset_random()
        #         result0 = compute_forward_derivative(A,S=S0,sensor_id=sensor_id,idx_param=idx_param,FD=True)
        #         fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_0"
        #         np.save(fname,result0.mean(axis=-1))
        #         print(f"Write gradient image to {fname}")
        #         img0 = result0 * 255.0
        #         cv2.imwrite(f"{filename.replace('FD','FD_0')}",img0)
        #     if mode != 0:
        #         S1 = Variable(sc.param_map[material_key].sigma_t.data.torch() + 2*param_delta , requires_grad=True)
        #         reset_random()
        #         result1 = compute_forward_derivative(A,S=S1,sensor_id=sensor_id,idx_param=idx_param,FD=True)
        #         fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd_1"
        #         np.save(fname,result1.mean(axis=-1))
        #         print(f"Write gradient image to {fname}")
        #         img1 = result1 * 255.0
        #         cv2.imwrite(f"{filename.replace('FD','FD_1')}",img1)
        
        # if mode == 2 or mode == 4:
        #     result_fd = (result1 - result0) / (2*fd_delta)
        #     img = np.mean(result_fd,axis=-1)
        #     fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_fd"
        #     np.save(fname,img)
        #     print(f"Write gradient image to {fname}")

        #     # Vis image error btw estimate and GT
        #     target = f"../../../data_kiwi_soap/realdata/{args.scene}/exr_ref/{sensor_id}.exr"
        #     tar = cv2.imread(target)
        #     fname=f"{grad_dir}/{args.scene}_{sensor_id}_{fd_delta}_diff"
        #     # breakpoint()
        #     diff = tar - result0
        #     np.save(fname,diff)
        #     print(f"Write gradient image to {fname}")
        

        # cmax = max(img.max(),abs(img.min()))
        # norm = MidpointNormalize(vmin=-cmax,vmax=cmax,midpoint=0.0)
        # plt.imshow(img, cmap='RdBu_r',norm=norm)
        # plt.tight_layout()
        # plt.colorbar()
        # plt.axis('off')
        # plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        # plt.close()

        # filename = filename.replace('.png','_nolegend.png')
        # mapper = cm.ScalarMappable(norm=norm, cmap=cmap_key)
        # img_fd = np.copy(mapper.to_rgba(img))
        # plt.tight_layout()
        # plt.axis('off')
        # plt.imshow(img_fd,extent=extent)
        # plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        # plt.close()
        
        # # Visualize absolute difference between FD and ours
        # filename = filename.replace('_nolegend.png','.png')
        # filename = filename.replace('FD','absdiff')
        # img_diff = np.abs(img_fd[...,:3] - img_ours[...,:3])# * 255.0
        # img_diff = img_diff[...,[2,1,0]].sum(axis=-1)
        # plt.imshow(img_diff,cmap='viridis',extent=extent)
        # plt.tight_layout()
        # plt.colorbar()
        # plt.axis('off')
        # # print(f"Write gradient image to {filename}")
        # plt.savefig(filename,bbox_inches='tight')
        # plt.close()
    else:
        optTask()
        
if __name__ == "__main__":
    opt_task(isSweep=False)
