import torch
import pickle
import os.path as osp
from qm9 import utils as qm9_utils
from qm9.data.collate import collate_fn
from qm9.models import EGNN
from qm9 import dataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
names = locals()
for num in ['9']:
    dir = "/work/01/gq54/p23002/guyuhan/wisteria_guyuhan/data_preprocess/826_Alchemy" + num + "_for_dimenet.npz"
    name = '906_egnn_inference_al' + num + 'test.npz'
    data=np.load(dir)
    # print(data.files)
    n = len(data['mu'])
    # n = 100

    prop = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
            'zpve', 'U0', 'U','H', 'G', 'Cv' ]
    split = [data['N'][0]]
    for i in tqdm(range(len(data['N'])-2)):
            split.append(split[-1] + data['N'][i + 1])
    # print(len(split))
    num_atoms_split = data['N']
    max = num_atoms_split.max()
    charges_split = np.split(data['Z'],split)
    charge_scale = data['Z'].max()
    positions_split = np.split(data['R'],split)
    index_split = data['id']
    for _ in prop:
            names[_ + '_split'] = data[_]

    dict_keys1 =  ['index','num_atoms', 
            'mu', 'alpha', 'homo', 'lumo', 
            'gap', 'r2', 'zpve', 'U0', 
            'U', 'H', 'G', 'Cv',]
    dict_keys2 = ['charges', 'positions',]
    dict_keys3 = ['one_hot', 'atom_mask', 'edge_mask']

    all_species = torch.tensor(data['Z']).unique(sorted=True)
    if all_species[0] == 0:
            all_species = all_species[1:]
    included_species = all_species

    for i in range(n):
        dict_i = {}
        _ = np.pad(charges_split[i],((0), (max - charges_split[i].size)),'constant')
        dict_i['charges'] = torch.tensor(_)
        if i == 0:
            all_charges = dict_i['charges'].unsqueeze(0)
        else:
            all_charges = torch.cat((all_charges,dict_i['charges'].unsqueeze(0)),0)

    all_one_hot = all_charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
    # print(all_one_hot.shape)
    dataset_alchemy = []
    for i in range(n):
        dict_i = {}
        for _ in dict_keys1:
            dict_i[_] = torch.tensor(names[_ + '_split'][i])
        _ = np.pad(charges_split[i],((0), (max - charges_split[i].size)),'constant')
        dict_i['charges'] = torch.tensor(_)
        if i == 0:
            all_charges = dict_i['charges'].unsqueeze(0)
        else:
            all_charges = torch.cat((all_charges,dict_i['charges'].unsqueeze(0)),0)
        _ = np.pad(positions_split[i],((0,max - positions_split[i].shape[0]), (0,0)),'constant')
        dict_i['positions'] = torch.tensor(_)
        # dict_i['one_hot'] = dict_i['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        dict_i['one_hot'] = all_one_hot[i]
        dataset_alchemy.append(dict_i)

    class al_dataset(Dataset):
        def  __init__(self,name):
            self.set = name
            
        def __len__(self):
            return len(dataset_alchemy)
        
        def __getitem__(self, idx):
            return self.set[idx]

    al9_set = al_dataset(dataset_alchemy)

    dataloaders1 = {'altest': DataLoader(al9_set,
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=collate_fn)
                            }

    infer_props = ['mu', 'alpha','homo', 'lumo', 'gap', 'r2','zpve', 'U0', 'H', 'G']
    for ip in infer_props:
        best_model = torch.load('/work/01/gq54/p23002/egnn_' + ip + '/exp_1_' + ip + '/best_model',map_location='cpu')
        device = 'cpu'
        dtype = torch.float32
        model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0,
                    attention=1, node_attr=0)
        model.load_state_dict(best_model['model_state_dict'])

        pred_all = [] 
        for i, data in enumerate(dataloaders1['altest']):
                model.eval()
                with torch.no_grad():
                    batch_size, n_nodes, _ = data['positions'].size()
                    atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
                    atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
                    edge_mask = data['edge_mask'].to(device, dtype)
                    one_hot = data['one_hot'].to(device, dtype)
                    charges = data['charges'].to(device, dtype)
                    nodes = qm9_utils.preprocess_input(one_hot, charges, 2, charge_scale, device)
                    nodes = nodes.view(batch_size * n_nodes, -1)
                    # nodes = torch.cat([one_hot, charges], dim=1)
                    edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
                    # label = data[args.property].to(device, dtype)
                    pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                                n_nodes=n_nodes)
                    if i == 0:
                        pred_all = pred
                    else:
                        pred_all = torch.cat((pred_all,pred),dim = 0)

        pred_all = pred_all.detach().numpy()
        labels=np.load(dir)
        label = labels[ip]

        dataloaders, charge_scale = dataset.retrieve_dataloaders(96, 2)
        meann, mad = qm9_utils.compute_mean_mad(dataloaders, ip)
        meann = meann.item()
        mad = mad.item()
        pred_all = pred_all * mad + meann
        mae = abs((pred_all)[0:n] - label[0:n]).mean()
        coef = np.corrcoef(pred_all[0:n],label[0:n])
        print(f'alchemy{num}', ip, 'all number:',n)
        print(' mae:', mae.round(7))
        print('coef:',coef[0][1].round(7))
        names['labels_total' + ip] = label[0:n]
        names['preds_total' + ip] = pred_all[0:n]
    
    np.savez(name,
            mu=names['preds_totalmu'],
            mu_label=names['labels_totalmu'],
            
            homo=names['preds_totalhomo'],
            homo_label=names['labels_totalhomo'],
            
            lumo=names['preds_totallumo'],
            lumo_label=names['labels_totallumo'],
            
            zpve=names['preds_totalzpve'],
            zpve_label=names['labels_totalzpve'],
            
            r2=names['preds_totalr2'],
            r2_label=names['labels_totalr2'],
            
            alpha=names['preds_totalalpha'],
            alpha_label=names['labels_totalalpha'],
            
            U0=names['preds_totalU0'],
            U0_label=names['labels_totalU0'],
            
            # U=names['preds_totalU'],
            # U_label=names['labels_totalU'],

            H=names['preds_totalH'], 
            H_label=names['labels_totalH'],
            
            G=names['preds_totalG'],
            G_label=names['labels_totalG'],
            
            # Cv=names['preds_totalCv'],
            # Cv_label=names['labels_totalCv'],
            )

