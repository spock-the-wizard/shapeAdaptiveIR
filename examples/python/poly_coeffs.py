from enum import Enum
import numpy as np 
import sys
import tqdm
import os


curdir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f"{curdir}/viz")
sys.path.append(f"{curdir}/viz/vae")

from vae.global_config import (DATADIR3D, FIT_REGULARIZATION, OUTPUT3D,
                               RESOURCEDIR, SCENEDIR3D, DATADIR)
import viewer.utils
import viewer.utils
from viewer.viewer import GLTexture, ViewerApp
import utils.mtswrapper



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

