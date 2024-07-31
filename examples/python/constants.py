import os

# replace this as home directory
HOME_DIR = "/sss/InverseTranslucent"

ROOT_DIR = os.path.join(HOME_DIR)
DATA_DIR = os.path.join(HOME_DIR, "data_kiwi_soap")
SCENES_DIR = os.path.join(ROOT_DIR, "examples/scenes/inverse/")

RAW_TEXTURE_DIR = os.path.join(ROOT_DIR, "examples/data/textures_raw/")
TEXTURE_DIR = os.path.join(DATA_DIR, "examples/textures")
REMESH_DIR = os.path.join(HOME_DIR, "ext/botsch-kobbelt-remesher-libigl/build")
RESULT_DIR = os.path.join(DATA_DIR, "results")
BLEND_SCENE = os.path.join(SCENES_DIR, "render.blend")
BLENDER_EXEC = "blender2.8" # Change this if you have a different blender installation
REAL_DIR = os.path.join(DATA_DIR, "realdata")
# data folders
ESSEN_DIR = os.path.join(DATA_DIR, "essen/")
LIGHT_DIR = os.path.join(SCENES_DIR, "../light/")
SHAPE_DIR = os.path.join(DATA_DIR, "smoothshape/")

params_gt = {
    'duck': {
        'albedo': [0.88305,0.183,0.011],
        'sigmat': [25.00, 25.00, 25.00],
        'mesh': "../../smoothshape/duck_v2.obj",
    },
    'head1': {
        'albedo': [0.9, 0.9, 0.9],
        'sigmat': [109.00, 109.00, 52.00],
        'mesh': '../../smoothshape/head_v2.obj',
    },
    'cone4': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [50.00, 50.00,50.00],
        'mesh': "../../smoothshape/vicini/cone_subdiv.obj",
    },
    'cone3': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [50.00, 50.00,50.00],
        'mesh': "../../smoothshape/vicini/cone_subdiv.obj",
    },
    'cone2': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [50.00, 50.00,50.00],
        'mesh': "../../smoothshape/vicini/cone_subdiv.obj",
    },
    'pyramid4': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [50.00, 50.00,50.00],
    },
    'cylinder4': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [50.00, 50.00,50.00],
    },
    'botijo2': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [80.00, 80.00,80.00],
        'mesh': "../../smoothshape/final/botijo2_.obj" 
    },
    'botijo3': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [50.00, 100.00, 100.00],
        'mesh': "../../smoothshape/final/botijo2_.obj" 
    },
    'kettle1': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [90.00, 60.00,100.00],
        'mesh' : "../../smoothshape/final/kettle_.obj",
    },
    'kettle2': {
        'albedo': [0.98, 0.98, 0.98],
        'sigmat': [60.00, 90.00,80.00],
        'mesh' : "../../smoothshape/final/kettle_.obj",
    },
    'buddha1': {
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [40.00, 40.00, 100.00],
    },
    'maneki1':{
        'albedo': [0.89, 0.89, 0.89],
        'sigmat': [78.37, 54.169, 83.51],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki3':{
        'albedo': [0.89, 0.89, 0.89],
        'sigmat': [78.37, 54.169, 83.51],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki2':{
        'albedo': [0.891524, 0.891524, 0.891524],
        'sigmat': [78.37, 54.169, 83.51], # invalid
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki4':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [100.0,100.0,100.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki5':{
        'albedo': [0.40, 0.90, 0.40],
        'sigmat': [40.0,20.0,50.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki6':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [20.0,50.0,50.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki7':{
        'albedo': [0.40, 0.90, 0.40],
        'sigmat': [40.0,20.0,50.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki8':{
        'albedo': [0.40, 0.90, 0.40],
        'sigmat': [40.0,20.0,50.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki9':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [20.0,50.0,50.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'maneki10':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [20.0,50.0,50.0],
        'mesh': '../../smoothshape/final/maneki_.obj',
    },
    'gargoyle1':{
        'albedo': [0.82, 0.82, 0.70],
        'sigmat': [80.0, 70.0, 50.0],
        'mesh': '../../smoothshape/final/gargoyle.obj',
    },
    'gargoyle2':{
        'albedo': [0.82, 0.82, 0.70],
        'sigmat': [80.0, 70.0, 50.0],
        'mesh': '../../smoothshape/final/gargoyle.obj',
    },
    'gargoyle3':{
        'albedo': [0.82, 0.82, 0.70],
        'sigmat': [80.0, 70.0, 50.0],
        'mesh': '../../smoothshape/final/gargoyle.obj',
    },
    'dragon1':{
        'albedo': [0.70, 0.90, 0.60],
        'sigmat': [70.0, 40.0, 40.0],
        'mesh': '../../smoothshape/final/dragon.obj',
    },
    'dragon2':{
        'albedo': [0.70, 0.90, 0.60],
        'sigmat': [70.0, 40.0, 40.0],
        'mesh': '../../smoothshape/final/dragon.obj',
    },
    'dragon3':{
        'albedo': [0.70, 0.90, 0.60],
        'sigmat': [70.0, 40.0, 40.0],
        'mesh': '../../smoothshape/final/dragon.obj',
    },
    'croissant1':{
        'albedo': [0.95, 0.90, 0.80],
        'sigmat': [30.0, 40.0, 50.0],
        'mesh': '../../smoothshape/final/croissant.obj',
    },
    'croissant2':{
        'albedo': [0.95, 0.90, 0.80],
        'sigmat': [30.0, 40.0, 50.0],
        'mesh': '../../smoothshape/final/croissant.obj',
    },
    'croissant3':{
        'albedo': [0.95, 0.90, 0.80],
        'sigmat': [30.0, 40.0, 50.0],
        'mesh': '../../smoothshape/final/croissant.obj',
    },
    'croissant4':{
        'albedo': [0.90, 0.90, 0.90,],
        'sigmat': [50.0, 50.0, 50.0],
        'mesh': '../../smoothshape/final/croissant.obj',
    },
    'croissant5':{
        'albedo': [0.90, 0.90, 0.90,],
        'sigmat': [50.0, 50.0, 50.0],
        'mesh': '../../smoothshape/final/croissant.obj',
    },
    'torus1':{
        'albedo': [0.80, 0.80, 0.80],
        'sigmat': [20.0, 80.0, 50.0],
        'mesh': '../../smoothshape/final/torus.obj',
    },
    'torus2':{
        'albedo': [0.80, 0.80, 0.80],
        'sigmat': [20.0, 80.0, 50.0],
        'mesh': '../../smoothshape/final/torus.obj',
    },
    'torus3':{
        'albedo': [0.80, 0.80, 0.80],
        'sigmat': [20.0, 80.0, 50.0],
        'mesh': '../../smoothshape/final/torus.obj',
    },
    'head4':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [100.0,100.0,100.0],

        'mesh': '../../smoothshape/head_v2.obj',
    },
    'kettle4':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [100.0,100.0,100.0],
        'mesh' : "../../smoothshape/final/kettle_.obj",
    },
    'kiwi':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [70.0,70.0,70.0],
    },
    'soap':{
        'albedo': [0.90, 0.90, 0.90],
        'sigmat': [70.0,70.0,70.0],
    },
    'pig1':{
        'albedo': [0.52584,0.102227,0.51597],
        'sigmat': [25.0,25.0,25.0],
    },
}