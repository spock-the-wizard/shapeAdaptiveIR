import sys
import psdr_cuda
import enoki as ek
import cv2
import numpy as np
import math
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD, Vector3i
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
import datetime
from matplotlib import colormaps
import wandb

import pytorch_ssim


# from largesteps.optimize import AdamUnifom
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import to_differential, from_differential

from loss import compute_image_matrix #mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss, mesh_cot_laplacian, mesh_uni_laplacian
from AdamUniform import UAdam
from tool_functions import checkpath

from constants import REMESH_DIR, RESULT_DIR, TEXTURE_DIR, SCENES_DIR, ROOT_DIR, REAL_DIR, LIGHT_DIR, ESSEN_DIR
sys.path.append(REMESH_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--scene',              type=str,      default='soap')
parser.add_argument('--stats_folder',       type=str,      default="joint_sss_temp")
parser.add_argument('--light_file',         type=str,      default="seal")
parser.add_argument('--ref_folder',         type=str,      default="exr_ref")
parser.add_argument('--scene_file',         type=str,    default=None)

parser.add_argument('--sigma_lr',           type=float,    default=0.04)
parser.add_argument('--eta_lr',             type=float,    default=0.01)
parser.add_argument('--albedo_lr',          type=float,    default=0.02)
parser.add_argument('--mesh_lr',            type=float,    default=0.05)
parser.add_argument('--rough_lr',            type=float,   default=0.02)

parser.add_argument('--img_weight',         type=float,    default=1.0)
parser.add_argument('--range_weight',         type=float,    default=10.0)
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

parser.add_argument('--seed',               type=int,      default=4)

parser.add_argument('--spp',                type=int,      default=32)
parser.add_argument('--sppe',               type=int,      default=32)
parser.add_argument('--sppse',              type=int,     default=500)

parser.add_argument('--integrator',         type=str,      default='direct')

parser.add_argument('--albedo_texture',     type=int,      default=256)
parser.add_argument('--sigma_texture',      type=int,      default=256)
parser.add_argument('--rough_texture',      type=int,      default=256)

parser.add_argument('--ref_spp',            type=int,      default=50)
parser.add_argument('--no_init',            type=str,     default="yes")
parser.add_argument('--d_type',             type=str,     default="real")
parser.add_argument('--silhouette',         type=str,     default="yes")

parser.add_argument('--render_gradient', action="store_true")

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

args = parser.parse_args()

import matplotlib as mpl
import matplotlib.pyplot as plt
# class MidpointNormalize(mpl.colors.Normalize):
#     """
#     class to help renormalize the color scale
#     """
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
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

def opt_task(args):
    # write intermedia results to ... 
    destdir = RESULT_DIR + "/{}/".format(args.scene)
    checkpath(destdir)
    statsdir = destdir + "/{}_{}/".format(args.stats_folder, args.seed)
    checkpath(statsdir)

    argsdir = destdir + "/{}_{}/settings_{}.txt".format(args.stats_folder, args.seed, datetime.datetime.now())
    saveArgument(args, argsdir)

    params_gt = {
        'duck': {
            'albedo': [0.88305,0.183,0.011],
            'sigmat': [25.00, 25.00, 25.00],
        },
        'head': {
            'albedo': [0.9, 0.9, 0.9],
            'sigmat': [109.00, 109.00, 52.00],
        },
        'head1': {
            'albedo': [0.9, 0.9, 0.9],
            'sigmat': [109.00, 109.00, 52.00],
        },
        'sphere1': {
            'albedo': [0.9, 0.9, 0.9],
            'sigmat': [54.00, 72.00, 98.00],
        },
        'cone1': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [1.00, 1.00, 1.00],
        },
        'cone2': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [5.00, 5.00, 5.00],
        },
        'cone3': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [20.00, 20.00,20.00],
        },
        'cone4': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [50.00, 50.00,50.00],
        },
        'cone5': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [100.00, 100.00,100.00],
        },
        'pyramid4': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [50.00, 50.00,50.00],
        },
        'pyramid5': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [100.00, 100.00,100.00],
        },
        'cylinder4': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [50.00, 50.00,50.00],
        },
        'botijo': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [50.00, 50.00,50.00],
        },
        'cylinder5': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [100.00, 100.00,100.00],
        },
        'kettle1': {
            'albedo': [0.98, 0.98, 0.98],
            'sigmat': [90.00, 60.00,100.00],
        },
        'buddha1': {
            'albedo': [0.90, 0.90, 0.90],
            'sigmat': [40.00, 40.00, 100.00],
        },
    }

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

    if args.d_type == "syn":
        lightdir = LIGHT_DIR + '/lights-sy-{}.npy'.format(args.light_file)
    elif args.scene == "duck":
        # TODO: fix this...
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

    tars = [] 
    maks = []
    tmeans = []
    for i in range(num_sensors):
        if args.d_type == "syn":
            filename = refdir + "/l{}_s{}.exr".format(i, i)
        elif args.d_type == "custom":
            # TODO : change this
            filename = refdir + "{}.exr".format(i)
        else:
            filename = refdir + "{}_{:05d}.exr".format(args.scene, i)
        
        print(filename)
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
        # print(indices)
        return indices

    def active_light(batch, num_lights):
        indices = torch.tensor(random.sample(range(num_lights), batch))
        return indices

    def texture_range_loss(A, S, R, G, weight):
        lossS = torch.pow(torch.where(S < 0, -S, torch.where(S > 100.0, S - 100.0, torch.zeros_like(S))), 2)
        lossA = torch.pow(torch.where(A < 0.0, -A, torch.where(A > 1.0, A - 1, torch.zeros_like(A))), 2)
        lossR = torch.pow(torch.where(R < 0.01, 0.02 - R, torch.where(R > 2.0, R - 2.0, torch.zeros_like(R))), 2)
        lossG = torch.pow(torch.where(G < 1.0, 2.0-G, torch.where(G > 10.0, G - 1, torch.zeros_like(G))), 2)
        loss = (lossA.mean() + lossR.mean() + lossG.mean() + lossS.mean()) * weight
        return loss


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

    def compute_render_loss(our_img1, our_img2, ref_img, weight):
        loss = 0
        for i in range(3): 
            I1_ = our_img1[i]
            I2_ = our_img2[i]
            T_ = ref_img[i] 
            # I1 = ek.select(I1_ > 1, 1.0, I1_)
            # I2 = ek.select(I2_ > 1, 1.0, I2_)
            # T = ek.select(T_ > 1, 1.0, T_) 
            # #TODO: Calibrate light value
            diff1 = ek.hmean((I1_ - T_) * (I2_ - T_))
            loss += ek.hmean(diff1) / 3.0 # + ek.hmean(diff2) / 3.0
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


    def compute_forward_derivative(A,S,sensor_id,idx_param=0,FD=False,):

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
        sc.configure()
        sc.setlightposition(Vector3fD(lights[sensor_id][0], lights[sensor_id][1], lights[sensor_id][2]))

        # Forward-propagate gradients from single input to multi-outputs
        # ek.set_label(a,"sigma_t")
        # ek.set_label(a,"albedo")
        # ek.set_label(img,"img")
        # print(ek.graphviz(img))
        # with open("test_graph.dot", "w") as f:
        #     f.write(ek.graphviz(img))
    
        if not FD:
            img = myIntegrator.renderD(sc, sensor_id)
            ek.forward(a[idx_param])
            grad_for = ek.gradient(img)
            return grad_for
        else:
            img = renderNtimes(sc, myIntegrator, args.ref_spp, sensor_id)
            # img = myIntegrator.renderC(sc, sensor_id)
            # img_np = img.numpy().reshape(512,512,-1)
            # del img
            return img 
            # return img_np
        
    class Renderer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, V, A, S, R, G, batch_size, seed,crop_idx=0):
            # Roughness = R
            _vertex     = Vector3fD(V)
            _albedo     = Vector3fD(A)
            _sigma_t    = Vector3fD(S)
            _rough      = FloatD(R)
            _eta        = FloatD(G)

            ek.set_requires_gradient(_vertex,       V.requires_grad)
            ek.set_requires_gradient(_albedo,       A.requires_grad)
            ek.set_requires_gradient(_sigma_t,      S.requires_grad)
            ek.set_requires_gradient(_rough,        R.requires_grad)
            ek.set_requires_gradient(_eta,          G.requires_grad)

        
            ctx.input1 = _vertex
            ctx.input2 = _albedo
            ctx.input3 = _sigma_t
            ctx.input4 = _rough
            ctx.input5 = _eta

        
            albedo  = ek.select(_albedo     > 0.99995,     0.99995, ek.select(_albedo < 0.0, 0.0, _albedo))
            sigma_t = ek.select(_sigma_t    < 0.0,      0.01,  _sigma_t)
            roughness = ek.select(_rough    < 0.01,      0.01,  _rough)
            eta       = ek.select(_eta      < 1.0,       1.0,   _eta)
            # sigma_t = ek.exp(_s)


            sc.param_map[mesh_key].vertex_positions  = _vertex
            sc.param_map[material_key].albedo.data   = albedo
            sc.param_map[material_key].sigma_t.data  = sigma_t
            sc.param_map[material_key].alpha_u.data  = roughness
            sc.param_map[material_key].eta.data      = eta
            
            print("------------------------------------seed----------",seed)
            npixels = ro.cropheight * ro.cropwidth
            sc.opts.spp = args.spp
            sc.setseed(seed*npixels)
            sc.configure()
            
            render_loss= 0    
            
            sensor_indices = active_sensors(batch_size, num_sensors)
            # print("sensor indices: ", sensor_indices)
            for sensor_id in sensor_indices:
                sc.setlightposition(Vector3fD(lights[sensor_id][0], lights[sensor_id][1], lights[sensor_id][2]))

                cox,coy,coh,cow = ro.crop_offset_x,ro.crop_offset_y,ro.cropheight,ro.cropwidth
                tar_img = Vector3fD(tars[sensor_id].reshape((ro.height,ro.width,-1))[coy:coy+coh,cox:cox+cow,:].reshape(-1,3).cuda())
                # tar_img = Vector3fD(tars[sensor_id].cuda())
                # weight_img = Vector3fD(tmeans[sensor_id].cuda()f)
                our_imgA = myIntegrator.renderD(sc, sensor_id)
                # FIXME: tmp disabled for debugging
                our_imgB = our_imgA
                # our_imgB = myIntegrator.renderD(sc, sensor_id)
                render_loss += compute_render_loss(our_imgA, our_imgB, tar_img, args.img_weight) / batch_size
                # render_loss += compute_render_loss(our_imgA, our_imgA, tar_img, args.img_weight) / batch_size
                
                # Save RMSE loss for logging
                # render_loss += compute_render_loss(our_imgA, our_imgB, tar_img, args.img_weight) / batch_size

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
                result = (None, None, None, None, None, None, None)
        
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
            # print("-------------------------S-----------------------------")
            gradS = ek.gradient(ctx.input3).torch()
            gradS[torch.isnan(gradS)] = 0.0
            gradS[torch.isinf(gradS)] = 0.0
            # print("-------------------------R-----------------------------")
            gradR = ek.gradient(ctx.input4).torch()
            gradR[torch.isnan(gradR)] = 0.0
            gradR[torch.isinf(gradR)] = 0.0
            # print("-------------------------G-----------------------------")
            gradG = ek.gradient(ctx.input5).torch()
            gradG[torch.isnan(gradG)] = 0.0
            gradG[torch.isinf(gradG)] = 0.0

            result = (gradV, gradA, gradS, gradR, gradG, None, None)
            del ctx.output, ctx.input1, ctx.input2, ctx.input3, ctx.input4, ctx.input5
            print("==========================")
            print("Estimated gradients")
            print("gradA ",gradA,"gradS ",gradS)
            # assert(torch.all(gradA==0.0))
            # assert(torch.all(gradS == 0.0))
            return result

    
    render = Renderer.apply
    render_batch = 1

    def resize(MyTexture, texturesize, textstr, i, channel=3):
        TextureMap = MyTexture.detach().cpu().numpy().reshape((texturesize, texturesize, channel))
        if texturesize < 512:
            texturesize = texturesize * 2
            TextureMap = cv2.cvtColor(TextureMap, cv2.COLOR_RGB2BGR)
            TextureMap = cv2.resize(TextureMap, (texturesize, texturesize))
            cv2.imwrite(statsdir + "/{}_resize_{}.exr".format(textstr, i+1), TextureMap)
        return texturesize

    def renderPreview(i, sidxs):
        cmap = colormaps.get('inferno')
        images = []
        error_map = []
        rmse_list = []
        ssim_list = []
        
        # print(sidxs)
        for idx in sidxs:
            # TODO: tmp setting
            sc.setlightposition(Vector3fD(lights[idx][0], lights[idx][1], lights[idx][2]))
            img2 = renderNtimes(sc, myIntegrator, args.ref_spp, idx)
            img2 = np.clip(img2,a_min=0.0, a_max=1.0)
            # target2 = tars[idx].numpy().reshape((ro.height,ro.width,-1))[coy:coy+coh,cox:cox+cow]
            target2 = tars[idx].numpy().reshape((ro.height,ro.width,-1)) #[coy:coy+coh,cox:cox+cow]
            target2 = cv2.cvtColor(target2, cv2.COLOR_RGB2BGR)
            target2 = np.clip(target2,a_min=0.0, a_max=1.0)
            # breakpoint()
            cv2.imwrite(statsdir + "/iter_{}_{}_out.png".format(i+1,idx), img2*255.0)
            cv2.imwrite(statsdir + "/iter_{}_{}_gt.png".format(i+1,idx), target2*255.0)
            absdiff2 = np.abs(img2 - target2)
                                
            import matplotlib.pyplot as plt
            rmse2 = rmse(img2,target2)
            plt.imshow(rmse2, cmap='inferno')
            plt.tight_layout()
            plt.colorbar()
            plt.axis('off')
            # plt.show()

            outfile_error = statsdir + "/error_{}_{}.png".format(i+1,idx)
            plt.savefig(outfile_error,bbox_inches='tight')
            plt.close()
            

            # print(rmse2[0,0])
            output2 = np.concatenate((img2, target2, absdiff2))
            images.append(output2)
            
            from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
            # X: (N,3,H,W`) a batch of non-negative RGB images (0~255)
            # Y: (N,3,H,W)  
            X = torch.from_numpy(img2.transpose(2,0,1)[None]) #torch.zeros((1,3,256,256))
            Y = torch.from_numpy(target2.transpose(2,0,1)[None]) #torch.zeros((1,3,256,256))
            # calculate ssim & ms-ssim for each image
            ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
            wandb.log({
                "images/gt": wandb.Image(cv2.cvtColor(target2,cv2.COLOR_BGR2RGB)),
                "images/out" : wandb.Image(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)),
                "images/error": wandb.Image(cv2.cvtColor(cv2.imread(outfile_error),cv2.COLOR_BGR2RGB)),              
                "loss/rmse_image": rmse(img2,target2).mean(),
                "loss/ssim_image": ssim_val,
            })
            
            rmse_list.append(rmse(img2,target2).mean())
            ssim_list.append(ssim_val)

        # rmse2 = cmap(rmse(img2,target2)).astype(np.float32)[...,:3] * 255.0
        output = np.concatenate((images), axis=1)                
        # error = np.concatenate((error_map), axis=1)                
        cv2.imwrite(statsdir + "/iter_{}.exr".format(i+1), output)
        # cv2.imwrite(statsdir + "/error_{}.png".format(i+1), error)
        wandb.log({
            "loss/rmse_avg_image": sum(rmse_list) / len(rmse_list), #rmse(img2,target2).mean(),
            "loss/ssim_avg_image": sum(ssim_list) / len(rmse_list), #rmse(img2,target2).mean(),
        })


    def optTask(args):

        run = wandb.init(
                # Set the project where this run will be logged
                project="inverse-learned-sss",
                # Track hyperparameters and run metadata
                config=args,
        )
        
        

        rmsehistory = []
        losshistory = [] 
        albedohistory = []
        sigmahistory = []
        roughhistory = []
        etahistory = []

        alb_texture_width = args.albedo_texture
        sig_texture_width = args.sigma_texture
        rgh_texture_width = args.rough_texture

        if args.albedo_texture > 0:
            if args.no_init == "yes":
                print("excuting----------------albedo-------------------------", args.no_init)
                init_albedo = np.zeros((alb_texture_width, alb_texture_width, 3), dtype=np.float32)
                init_albedo[:, :, :] = [0.9, 0.9, 0.9]
                cv2.imwrite(destdir + "/albedo_resize_init.exr", init_albedo)
                del init_albedo
                sc.param_map[material_key].setAlbedoTexture(destdir + "/albedo_resize_init.exr")
            
        if args.sigma_texture > 0:
            if args.no_init == "yes":
                print("excuting-------------------sigma----------------------", args.no_init)
                init_sigma = np.zeros((sig_texture_width, sig_texture_width, 3), dtype=np.float32)
                init_sigma[:,:,:] = [1.5, 1.5, 1.5]
                cv2.imwrite(destdir + "/sigma_resize_init.exr", init_sigma)
                del init_sigma
                sc.param_map[material_key].setSigmaTexture(destdir + "/sigma_resize_init.exr")

        if args.rough_texture > 0:
            if args.no_init == "yes":
                print("excuting-------------------sigma----------------------", args.no_init)
                init_rough = np.zeros((rgh_texture_width, rgh_texture_width, 1), dtype=np.float32)
                init_rough[:,:,:] = [0.015]
                cv2.imwrite(destdir + "/rough_resize_init.exr", init_rough)
                del init_rough
                sc.param_map[material_key].setAlphaTexture(destdir + "/rough_resize_init.exr")
            
        
        S = Variable(torch.log(sc.param_map[material_key].sigma_t.data.torch()), requires_grad=True)
        A = Variable(sc.param_map[material_key].albedo.data.torch(), requires_grad=True)
        R = Variable(sc.param_map[material_key].alpha_u.data.torch(), requires_grad=True)
        G = Variable(sc.param_map[material_key].eta.data.torch(), requires_grad=True)

        # TODO: disabled for now
        V = Variable(sc.param_map[mesh_key].vertex_positions.torch(), requires_grad=not(args.mesh_lr==0))
        F = sc.param_map[mesh_key].face_indices.torch().long()
        M = compute_matrix(V, F, lambda_ = args.laplacian)


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

        params = []

        params.append({'params': V, 'lr': args.mesh_lr, "I_Ls": M, 'largestep': True})
                    
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

        for i in range(args.n_iters):
            # NOTE: added chunk training
            loss = 0
            image_loss = 0
            range_loss = 0
            sc.opts.cropheight = sc.opts.height // args.n_crops
            sc.opts.cropwidth = sc.opts.width // args.n_crops
            for ix in range(args.n_crops):
                for iy in range(args.n_crops):
                
                    optimizer.zero_grad()
                    # image_loss = renderV, A, S, R, G, render_batch, i)
                    # breakpoint()
                    sc.opts.crop_offset_x = ix * sc.opts.cropwidth
                    sc.opts.crop_offset_y = iy * sc.opts.cropheight

                    image_loss_ = render(V, A, torch.exp(S), R, G, render_batch, i,)
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
                    optimizer.step()
                    loss += loss_.item()
                    image_loss += image_loss_.item()
                    range_loss += range_loss_.item()

                    del loss_, range_loss_, image_loss_

            loss /= args.n_crops

        
            # print("------ Iteration ---- ", i, ' image loss: ', loss.item(), ' eta: ', G.detach().cpu().numpy())   
            print("------ Iteration ---- ", i, ' image loss: ', loss, ' eta: ', G.detach().cpu().numpy())   
            
            if args.albedo_texture == 0:
                print("\n albedo: ", A.detach().cpu().numpy()) 
                albedohistory.append([A.detach().cpu().numpy()])
            if args.sigma_texture == 0:
                print("\n sigma: ", S.detach().cpu().numpy())
                # sigmahistory.append([S.detach().cpu().numpy()])   
                sigmahistory.append([np.exp(S.detach().cpu().numpy())])   
            if args.rough_texture == 0:
                print("\n rough: ", R.detach().cpu().numpy())
                roughhistory.append(R.detach().cpu().numpy())    
            
            # wandb.log({
            #     "image loss": loss.item(),
            #     "eta": G.detach().cpu().numpy(),
            #     "albedo" : A.detach().cpu().numpy(),
            #     "sigmaT" : S.detach().cpu().numpy(),
            #     "rough": R.detach().cpu().numpy(),
            # })
            log_alb = A.detach().cpu().numpy()[0]
            log_sig = S.detach().cpu().numpy()[0]

            wandb.log({
                "loss/rmse_param_alb": rmse(log_alb,params_gt[args.scene]['albedo']),
                "loss/rmse_param_sig": rmse(np.exp(log_sig), params_gt[args.scene]['sigmat']),
                "loss/rmse_param_sigT": rmse(log_sig, np.log(params_gt[args.scene]['sigmat'])),

                "param/eta": G.detach().cpu().numpy(),
                "param/rough": R.detach().cpu().numpy(),
                "param/albedo_r" : log_alb[0],
                "param/albedo_g" : log_alb[1],
                "param/albedo_b" : log_alb[2],
                "param/sigmaT_r" : log_sig[0],
                "param/sigmaT_g" : log_sig[1],
                "param/sigmaT_b" : log_sig[2],
            })


            etahistory.append([G.detach().cpu().numpy()])
            
            # losshistory.append([loss.detach().cpu().numpy()]) 
            losshistory.append([loss])
            
            torch.cuda.empty_cache()
            ek.cuda_malloc_trim()
            
        
            if i == 0 or ((i+1) %  args.n_dump) == 0:
                wandb.log({
                        # "loss/total": loss.item(),
                        # "loss/image": image_loss.item(),
                        "loss/total": loss,
                        "loss/image": image_loss,
                })
            # del loss, range_loss, image_loss
            
            if ((i+1) % args.n_reduce_step) == 0:
                lrs = []
                args.mesh_lr = args.mesh_lr * 0.95
                args.albedo_lr = args.albedo_lr * 0.95
                args.sigma_lr = args.sigma_lr * 0.95
                args.rough_lr = args.rough_lr * 0.95
                args.eta_lr = args.eta_lr * 0.95
                lrs.append(args.mesh_lr)
                lrs.append(args.albedo_lr)
                lrs.append(args.sigma_lr)
                lrs.append(args.rough_lr)
                lrs.append(args.eta_lr)

                optimizer.setLearningRate(lrs)

            # if (i==0 and args.n_iters == 1) or ((i+1) %  args.n_dump) == 0:
            if i == 0 or ((i+1) %  args.n_dump) == 0:
                # sensor_indices = active_sensors(1, num_sensors)
                # renderPreview(i, np.array([0], dtype=np.int32))
                sc.opts.cropheight = sc.opts.height
                sc.opts.cropwidth = sc.opts.width
                sc.opts.crop_offset_x = 0 
                sc.opts.crop_offset_y = 0 
                sc.configure()
                renderPreview(i, np.array([0, 1, 3, 6, 20], dtype=np.int32))
                # renderPreview(i, np.array([0, 1, 4, 10, 19], dtype=np.int32))
                # renderPreview(i, np.array([0, 1, 25, 3, 35], dtype=np.int32))
                if args.albedo_texture > 0:
                    albedomap = A.detach().cpu().numpy().reshape((alb_texture_width, alb_texture_width, 3))
                    albedomap = cv2.cvtColor(albedomap, cv2.COLOR_RGB2BGR)
                    albedomap[albedomap >= 1.0] = 0.9999
                    albedomap[albedomap <= 0.0] = 0.0
                    cv2.imwrite(statsdir + "/albedo_{}.exr".format(i+1), albedomap)

                if args.sigma_texture > 0:
                    print(S.detach().cpu().numpy().shape)
                    sigmamap = S.detach().cpu().numpy().reshape((sig_texture_width, sig_texture_width, 3))
                    sigmamap = cv2.cvtColor(sigmamap, cv2.COLOR_RGB2BGR)
                    sigmamap = np.exp(sigmamap)
                    cv2.imwrite(statsdir + "/sigma_{}.exr".format(i+1), sigmamap)


                if args.rough_texture > 0:
                    roughmap = R.detach().cpu().numpy().reshape((rgh_texture_width, rgh_texture_width, 1))
                    # roughmap = 1.0 / np.power(roughmap, 2.0)
                    cv2.imwrite(statsdir + "/rough_{}.exr".format(i+1), roughmap)

                saveHistory(statsdir)
                sc.param_map[mesh_key].dump(statsdir+"obj_%d.obj" % (i+1))
                # dumpPly(statsdir+"obj_%d" % (i+1), V, F)

            if ((i + 1) % args.n_resize) == 0:
                update = False
                if args.albedo_texture > 0:
                    oldlength = alb_texture_width
                    alb_texture_width = resize(A, alb_texture_width, "albedo", i)
                    if oldlength < alb_texture_width: 
                        update = True
                        sc.param_map[material_key].setAlbedoTexture(statsdir+"/albedo_resize_{}.exr".format(i+1)) 
                        A = Variable(sc.param_map[material_key].albedo.data.torch(), requires_grad=True)

                if args.sigma_texture > 0:
                    oldlength = sig_texture_width
                    sig_texture_width = resize(S, sig_texture_width, "sigma", i)
                    if oldlength < sig_texture_width: 
                        update = True
                        sc.param_map[material_key].setSigmaTexture(statsdir+"/sigma_resize_{}.exr".format(i+1))
                        S = Variable(sc.param_map[material_key].sigma_t.data.torch(), requires_grad=True)

                if args.rough_texture > 0:
                    oldlength = rgh_texture_width
                    rgh_texture_width = resize(R, rgh_texture_width, "rough", i, 1)
                    if oldlength < rgh_texture_width: 
                        update = True
                        sc.param_map[material_key].setRoughTexture(statsdir+"/rough_resize_{}.exr".format(i+1))
                        R = Variable(sc.param_map[material_key].alpha_u.data.torch(), requires_grad=True)


                if update:
                    del optimizer
                    
                    params = []
                    params.append({'params': V, 'lr': args.mesh_lr, "I_Ls": M, 'largestep': True})
                                
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
            torch.cuda.empty_cache()
            ek.cuda_malloc_trim()

    if args.render_gradient:
        GRAD_DIR = "../../grad"

        fd_delta = 5e-2 #2
        # fd_delta = 5 #2
        sensor_id = 0
        idx_param = 2 #0
        param_delta = torch.zeros(3).cuda()
        param_delta[idx_param] = fd_delta

        isAlbedo = True
        # isAlbedo = False
        
        A,S = None,None
        if isAlbedo:
            A = Variable(sc.param_map[material_key].albedo.data.torch(), requires_grad=True)
            filename = os.path.join(GRAD_DIR,f"{args.scene}_sensor{sensor_id}_albedo{sc.param_map[material_key].albedo.data}_param{idx_param}_{sc.param_map[material_key].type_name()}_deriv.png")
        else:
            S = Variable(torch.log(sc.param_map[material_key].sigma_t.data.torch()), requires_grad=True)
            filename = os.path.join(GRAD_DIR,f"{args.scene}_sensor{sensor_id}_sigma_t{sc.param_map[material_key].sigma_t.data}_param{idx_param}_{sc.param_map[material_key].type_name()}_deriv.png")
        
        result = compute_forward_derivative(A=A, S=S, sensor_id=sensor_id,idx_param=idx_param)
        img = result.numpy().reshape(sc.opts.cropheight,sc.opts.cropwidth,-1)[...,idx_param]

        # norm = MidpointNormalize(midpoint=0.0)

        norm = MidpointNormalize(vmin=img.min(), vmax=img.max(), midpoint=0)
        # plt.imshow(img, cmap='RdBu_r',norm=norm)
        plt.imshow(img, cmap='RdBu_r',vmin=-1.0, vmax=1.0)#norm=norm)
        plt.tight_layout()
        plt.colorbar()
        plt.axis('off')

        print(f"Write gradient image to {filename}")
        plt.savefig(filename,bbox_inches='tight')
        plt.close()
        del result, img
        
        filename = filename.replace("deriv","FD",)
        ## Gradient estimate using finite differences
        A,S = None, None
        if isAlbedo:
            A0 = Variable(sc.param_map[material_key].albedo.data.torch() - param_delta, requires_grad=True)
            A1 = Variable(sc.param_map[material_key].albedo.data.torch() + param_delta, requires_grad=True)
            result0 = compute_forward_derivative(A0,S=S,sensor_id=sensor_id,idx_param=idx_param,FD=True)
            result1 = compute_forward_derivative(A1,S=S,sensor_id=sensor_id,idx_param=idx_param,FD=True)
        else:
            S0 = Variable(sc.param_map[material_key].sigma_t.data.torch() - fd_delta, requires_grad=True)
            S1 = Variable(sc.param_map[material_key].sigma_t.data.torch() + fd_delta, requires_grad=True)
            result0 = compute_forward_derivative(A,S=S0,sensor_id=sensor_id,idx_param=idx_param,FD=True)
            result1 = compute_forward_derivative(A,S=S1,sensor_id=sensor_id,idx_param=idx_param,FD=True)
        
        img0 = result0 * 255.0
        img1 = result1 * 255.0
        # img0 = result0.numpy().reshape(512,512,-1) * 255
        # img1 = result1.numpy().reshape(512,512,-1) * 255

        cv2.imwrite(f"{filename.replace('FD','FD_0')}",img0)
        cv2.imwrite(f"{filename.replace('FD','FD_1')}",img1)
        result_fd = (result1 - result0) / (2*fd_delta)
        img = result_fd[...,idx_param]

        # norm = MidpointNormalize(midpoint=0.0)
        
        # divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0., vmax=1.0)
        norm = MidpointNormalize(vmin=img.min(), vmax=img.max(), midpoint=0)
        plt.imshow(img, cmap='RdBu_r',vmin=-1.0, vmax=1.0)#norm=norm)
        # plt.imshow(img, cmap='RdBu',norm=divnorm)
        plt.tight_layout()
        plt.colorbar()
        plt.axis('off')

        print(f"Write gradient image to {filename}")
        plt.savefig(filename,bbox_inches='tight')
        plt.close()

        del result0, result1, result_fd
    else:
        optTask(args)

if __name__ == "__main__":

    opt_task(args)
    
    # img1 = cv2.imread("../../../data_kiwi_soap/results/head1/exp1/for_deng_clip_2/iter_1_0_out.png")
    # img2 = cv2.imread("../../../data_kiwi_soap/results/head1/exp1/for_deng_clip_2/iter_1_0_gt.png")
    
    # tmp1 = img1 / 255.0
    # tmp2 = img2 / 255.0
    # error2 = rmse_total(tmp1,tmp2)
    # print(error2)

    # breakpoint()
