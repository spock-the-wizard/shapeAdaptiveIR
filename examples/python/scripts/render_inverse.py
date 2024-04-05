
# import cv2
# import os

# def make_video(image_folder, video_name, fps=30):
#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#     images.sort()  # Sort the images by name
    
#     # Assuming all images are the same size, get dimensions of the first image
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 file
#     video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

#     for image in images:
#         video.write(cv2.imread(os.path.join(image_folder, image)))

#     cv2.destroyAllWindows()
#     video.release()
import imageio
import os
import glob

def create_looping_gif(image_folder, output_gif_path, idx=0,fps=10):
    # Get list of images in folder sorted by name
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg"))])
    images = glob.glob(f"{image_folder}/iter_*_{idx}_out.png")
    
    # Read images
    with imageio.get_writer(output_gif_path, mode='I', fps=fps) as writer:
        for filename in images:
            image = imageio.imread(filename)#os.path.join(image_folder, filename))
            writer.append_data(image)
            
        # To make the GIF loop you don't need to do anything specific,
        # imageio handles that by default for GIFs

# Usage example
# create_looping_gif('path/to/image/folder', 'output_looping.gif')


# Usage
scene = "cylinder4"
idx = 0
create_looping_gif(f'../../../data_kiwi_soap/results/{scene}/exp3/var30_2', f'{scene}_deng.mp4',idx=idx)
create_looping_gif(f'../../../data_kiwi_soap/results/{scene}/exp3/var34_2', f'{scene}_ours.mp4',idx=idx)
