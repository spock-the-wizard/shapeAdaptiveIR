import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import json
import argparse

class SimpleFCN2(nn.Module):
    def __init__(self,stats_pth,k_mean,k_stdinv):
        super().__init__()

        hd = 32
        self.fc1 = nn.Linear(1,hd)
        self.fc2 = nn.Linear(hd,hd)
        self.fc3_= nn.Linear(hd,20)
        # self.fc3 = nn.Linear(hd,20)
        # self.fc2 = nn.Linear(hd,hd)
        # self.fc2_1 = nn.Linear(hd,hd)
        # self.fc2_2 = nn.Linear(hd,hd)
        # self.fc2_3 = nn.Linear(hd,hd)
        # self.fc2_1= nn.Linear(16, 16) # Hidden layer
        # self.fc2_2= nn.Linear(16, 16) # Hidden layer
        # self.fc2_3= nn.Linear(16, 16) # Hidden layer
        # self.fc2_4= nn.Linear(16, 16) # Hidden layer
        # self.fc2_5= nn.Linear(16, 16) # Hidden layer
        self.fc3 = nn.Linear(hd, 20) # Output layer, matching the 20D coeffs
        
        # Load poly and med coeffs
        self.stats_pth = stats_pth
        import json
        feat_name = 'mlsPolyLS3'
        self.stats_med = json.load(open(self.stats_pth,"r"))
        # Load mean, stdinv
        self.coeffs_mean= torch.tensor(self.stats_med[feat_name+"_mean"],device='cuda')
        self.coeffs_stdinv= torch.tensor(self.stats_med[feat_name+"_stdinv"],device='cuda')
        
        self.k_mean = k_mean.cuda()
        self.k_stdinv = k_stdinv.cuda()

    def forward(self, kernelEps, maxCoeffs ,denormalize=True,sqrtKernelEps=False):
        # Normalize maxCoeffs
        maxCoeffs = (maxCoeffs - self.coeffs_mean) * self.coeffs_stdinv
        # Normalize kernelEps
        if not sqrtKernelEps:
            kernelEps = (kernelEps - self.k_mean) * self.k_stdinv
        
        x = torch.cat((kernelEps, maxCoeffs),1).cuda()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc2_2(x))
        x = torch.relu(self.fc2_3(x))
        # x = torch.relu(self.fc2_4(x))
        # x = torch.relu(self.fc2_5(x))
        x = self.fc3(x)
        if not sqrtKernelEps and denormalize:
            x = x / self.coeffs_stdinv + self.coeffs_mean
        return x
    

class SimpleFCN(nn.Module):
    def __init__(self,stats_pth,k_mean,k_stdinv,sqrtKernelEps=False):
        super().__init__()
        self.sqrtKernelEps = sqrtKernelEps

        hd = 64
        self.fc1 = nn.Linear(21, hd)  # Input layer (20D constant + 1D kernelEps = 21)
        self.fc2 = nn.Linear(hd,hd)
        self.fc2_1 = nn.Linear(hd,hd)
        self.fc2_2 = nn.Linear(hd,hd)
        self.fc2_3 = nn.Linear(hd,hd)
        # self.fc2_1= nn.Linear(16, 16) # Hidden layer
        # self.fc2_2= nn.Linear(16, 16) # Hidden layer
        # self.fc2_3= nn.Linear(16, 16) # Hidden layer
        # self.fc2_4= nn.Linear(16, 16) # Hidden layer
        # self.fc2_5= nn.Linear(16, 16) # Hidden layer
        self.fc3 = nn.Linear(hd, 20) # Output layer, matching the 20D coeffs
        
        # Load poly and med coeffs
        self.stats_pth = stats_pth
        feat_name = 'mlsPolyLS3'
        self.stats_med = json.load(open(self.stats_pth,"r"))
        # Load mean, stdinv
        self.coeffs_mean= torch.tensor(self.stats_med[feat_name+"_mean"],device='cuda')
        self.coeffs_stdinv= torch.tensor(self.stats_med[feat_name+"_stdinv"],device='cuda')
        
        self.k_mean = k_mean.cuda()
        self.k_stdinv = k_stdinv.cuda()

    # FIXME: tmp setting
    def forward(self, kernelEps, maxCoeffs ,denormalize=True,):
        # Normalize maxCoeffs
        maxCoeffs = (maxCoeffs - self.coeffs_mean) * self.coeffs_stdinv
        # Normalize kernelEps
        if not self.sqrtKernelEps:
            kernelEps = (kernelEps - self.k_mean) * self.k_stdinv
        
        x = torch.cat((kernelEps, maxCoeffs),1).cuda()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc2_2(x))
        x = torch.relu(self.fc2_3(x))
        # x = torch.relu(self.fc2_4(x))
        # x = torch.relu(self.fc2_5(x))
        x = self.fc3(x)
        if not self.sqrtKernelEps and denormalize:
            x = x / self.coeffs_stdinv + self.coeffs_mean
        return x
    
class NPZDataset(Dataset):
    def __init__(self, data_path,num_verts):
        super(NPZDataset, self).__init__()
        # Initialize an empty list to hold individual data points
        self.data_points = []
        
        # Determine whether we're dealing with a single file or a directory
        if os.path.isdir(data_path):
            file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]
        elif os.path.isfile(data_path) and data_path.endswith('.npz'):
            file_paths = [data_path]
        else:
            raise ValueError("Provided path is neither a .npz file nor a directory containing .npz files.")
        
        self.poly_data = [] 
        self.kernelEps = []
        # Load data from files
        count = 0
        for file_path in file_paths:
            with np.load(file_path, allow_pickle=True) as data:
                # self.poly_data.append()
                for key,val in data.items():
                    v = val.tolist()
                    kernelEps  = torch.tensor(v['kernelEps'],device='cuda',dtype=torch.float32)
                    self.poly_data.append({
                        'albedo': torch.tensor(v['albedo'],device='cuda',dtype=torch.float32),
                        'sigma_t': torch.tensor(v['sigma_t'],device='cuda',dtype=torch.float32),
                        'g': torch.tensor(v['g'],device='cuda',dtype=torch.float32),
                        'coeffs': torch.tensor(v['coeffs'],device='cuda',dtype=torch.float32),
                        'kernelEps': kernelEps,
                        'fitScaleFactor': torch.tensor(v['fitScaleFactor'],device='cuda',dtype=torch.float32),
                        'maxCoeffs': torch.tensor(v['maxCoeffs'],device='cuda',dtype=torch.float32),
                        })
                    self.kernelEps.append(kernelEps)

                    count +=1 
                    if num_verts == count:
                        break
            # FIXME: flatten kernelEps for multi vertices
            self.kernelEps = torch.cat(self.kernelEps)
            self.k_mean = torch.mean(self.kernelEps)
            self.k_stdinv = 1/torch.std(self.kernelEps)

            self.num_items = len(self.poly_data) #data['albedo'].shape[0]  # Assuming 'albedo' represents the items
            self.num_verts = len(self.poly_data)
            self.num_med = self.poly_data[0]['coeffs'].shape[0]

    def __len__(self):
        return self.num_verts * self.num_med #self.num_items #self.albedo.shape[0] #len(self.data_points)

    def __getitem__(self, idx):
        idx_med = idx % self.num_med #idx_med # lf.poly_num_med
        idx_vert = idx // self.num_med
        kernelEps = self.poly_data[idx_vert]['kernelEps'][idx_med]
        coeffs = self.poly_data[idx_vert]['coeffs'][idx_med]
        maxCoeffs = self.poly_data[idx_vert]['maxCoeffs'][idx_med]

        return kernelEps,maxCoeffs, coeffs

def save_mlp_coeffs(file_path,model,out_path):
    print(f"Running Inference... saving to {out_path}")
    
    out = {}
    with np.load(file_path, allow_pickle=True) as data:
        # self.poly_data.append()
        for key,val in data.items():
            v = val.tolist()

            # for k,v in val.items():
            coeffs = torch.tensor(v['coeffs'],device='cuda')
            kernelEps = torch.tensor(v['kernelEps'],device='cuda')[...,None]
            maxCoeffs = torch.tensor(v['maxCoeffs'],device='cuda')
            
            
            # inputs = torch.cat((kernelEps, maxCoeffs),1).cuda()
            outputs = model(kernelEps.cuda(),maxCoeffs.cuda())
            
            v['coeffs'] = outputs.detach().cpu().numpy()
            out[key] = np.array(v)
            # data[key].set(v)
    np.savez(out_path,**out)

if __name__ == "__main__":


    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',              default="test/noname")
    parser.add_argument('--scene',              default="head_v2")
    parser.add_argument('--num_verts',              type=int,default=-1)

    parser.add_argument('--sqrtKernelEps', action="store_true")
    parser.add_argument('--diff', action="store_true")
    args = parser.parse_args()

    exp_name = args.exp_name.split("/")[-1]
    num_verts = args.num_verts
    batch_size = 5000
    n_ckpt = 100
    n_print = 1 #100

    # ckpt_pth = f"./head_v2_mlp_{exp_name}_{num_verts}.pth"
    # out_path = f"./head_v2_poly_data_{exp_name}_{num_verts}.npz"
    ckpt_pth = f"./{args.scene}_mlp_{exp_name}_{num_verts}.pth"
    out_path = f"./{args.scene}_poly_data_{exp_name}_{num_verts}.npz"
    data_pth = f"../viz/{args.scene}_poly_data.npz"
    stats_pth = "../../../data_stats.json"
    run = wandb.init(
            project="inverse-learned-sss-descriptor",
            config=args,
    )
            

    dataset =NPZDataset(data_pth,num_verts=num_verts)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=(num_verts == -1)) #,num_workers=10,) #pin_memory=True)

    k_mean = dataset.k_mean
    k_stdinv = dataset.k_stdinv
    model = SimpleFCN(stats_pth=stats_pth,k_mean=k_mean,k_stdinv=k_stdinv,sqrtKernelEps=args.sqrtKernelEps).cuda()
    if os.path.exists(ckpt_pth):
        model.load_state_dict(torch.load(ckpt_pth))
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)



    # Training loop
    num_epochs = 15000  # Number of epochs
    for epoch in range(num_epochs):
        for kernelEps,in_coeffs,out_coeffs in tqdm(dataloader,desc=f"Epoch {epoch+1}/{num_epochs}"):  # Assuming 'dataloader' is your DataLoader instance
            out_coeffs = out_coeffs.cuda()
            kernelEps = kernelEps.unsqueeze(-1)
            
            # Concatenate kernelEps with the constant input
            # Forward pass
            outputs = model.forward(kernelEps, in_coeffs)
            loss = criterion(outputs, out_coeffs)
            # print("[DEBUG]", loss)
            # Backward and optimize
            optimizer.zero_grad()
            # print("[DEBUG]", outputs)
            
            loss.backward()
            optimizer.step()

        if epoch % n_print == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            wandb.log({"Loss" : loss.item(), "epoch": epoch,
                    "values/out": outputs[0].detach().cpu().numpy(),
                    "values/target": out_coeffs[0].detach().cpu().numpy(),
                    })

        if epoch % n_ckpt == 0:
            torch.save(model.state_dict(),ckpt_pth)

    torch.save(model.state_dict(),ckpt_pth)

    # Test loop
    with torch.no_grad():
        save_mlp_coeffs(data_pth,model,out_path)