##############################################
###### 20220315, ki,kd,ic50多任务模型的预测模块
###### acsf descriptor 维度 = 227
###### 运行命令示例
### python3 predict_mt_chembl.py --num_process=10 --bin_size=200 --batch_size=128 --sdfs_path=/data/wspy/data/metal_binding/data/top1-sdfs/ --protein_path=/data/wspy/data/metal_binding/data/protein_prep/ --temp_path=/data/wspy/data/metal_binding/data/temp/ --dock_engine=plants --csv_path=/data/wspy/data/metal_binding/data/csv_files/ --work_name=P9WKE1
##############################################
import rdkit
import argparse
from utils import set_random_seed
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from model_v2 import IGN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dgl.data.utils import Subset
from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot
from dgl.data.chem import BaseBondFeaturizer
from functools import partial
from prody import *
from pylab import *
import pandas as pd
import warnings
import os
import pickle
from dscribe.descriptors import ACSF
from ase import Atoms
from scipy import sparse
import warnings
from utils import *
import copy
import multiprocessing as mp
import gc
import joblib

warnings.filterwarnings("ignore")

# Setting up the ACSF descriptor
atoms_sybols = ['C', 'N', 'O', 'S', 'P', 'B', 'F', 'Cl', 'Br', 'I', 'Zn', 'Mg', 'Mn', 'Ca', 'Na', 'Fe',
                'X']  # pdb2020, other elments were encoded as 'X'
important_atoms_sybols = ['C', 'N', 'O', 'S', 'P', 'B', 'F', 'Cl', 'Br', 'I', 'Zn', 'Mg', 'Mn', 'Ca', 'Na', 'Fe']
acsf = ACSF(
    species=atoms_sybols,
    rcut=8.0,
    g2_params=[[4.0, 3.17]],
    g4_params=[[0.1, 3.14, 1]],
)


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=important_atoms_sybols,
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([partial(bond_type_one_hot,
                                                                         allowable_set=['SINGLE', 'DOUBLE', 'TRIPLE',
                                                                                        'AROMATIC', 'ZERO'],
                                                                         encode_unknown=True), bond_is_conjugated,
                                                                 bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


# AtomFeaturizer = MyAtomFeaturizer()  # 249  #pdb2016
# AtomFeaturizer = MyAtomFeaturizer()  # 396  pdb2020
AtomFeaturizer = MyAtomFeaturizer()  # 227
BondFeaturizer = MyBondFeaturizer()  # 23


def graphs_from_mol_ign(dir, key, label, affinity_type, dis_threshold=8.0):
    '''
    :param dir:
    :param key:
    :param label:
    :param affinity_type: Ki,Kd or IC50
    :param dis_threshold:
    :return:
    '''
    add_self_loop = False
    status = True
    # try:
    with open(dir, 'rb') as f:
        mol1, mol2 = pickle.load(f)
    # the distance threshold to determine the interaction between ligand atoms and protein atoms
    dis_threshold = dis_threshold

    # construct graphs1
    g = dgl.DGLGraph()
    # add nodes
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
    num_atoms = num_atoms_m1 + num_atoms_m2
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # add edges, ligand molecule
    num_bonds1 = mol1.GetNumBonds()
    src1 = []
    dst1 = []
    for i in range(num_bonds1):
        bond1 = mol1.GetBondWithIdx(i)
        u = bond1.GetBeginAtomIdx()
        v = bond1.GetEndAtomIdx()
        src1.append(u)
        dst1.append(v)
    src_ls1 = np.concatenate([src1, dst1])
    dst_ls1 = np.concatenate([dst1, src1])
    g.add_edges(src_ls1, dst_ls1)

    # add edges, pocket
    num_bonds2 = mol2.GetNumBonds()
    src2 = []
    dst2 = []
    for i in range(num_bonds2):
        bond2 = mol2.GetBondWithIdx(i)
        u = bond2.GetBeginAtomIdx()
        v = bond2.GetEndAtomIdx()
        src2.append(u + num_atoms_m1)
        dst2.append(v + num_atoms_m1)
    src_ls2 = np.concatenate([src2, dst2])
    dst_ls2 = np.concatenate([dst2, src2])
    g.add_edges(src_ls2, dst_ls2)

    # add interaction edges, only consider the euclidean distance within dis_threshold
    g3 = dgl.DGLGraph()
    g3.add_nodes(num_atoms)
    dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    node_idx = np.where(dis_matrix < dis_threshold)
    src_ls3 = np.concatenate([node_idx[0]])
    dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1])
    g3.add_edges(src_ls3, dst_ls3)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'), dtype=torch.float)  # init 'h'
    g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
    g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

    atom_ls = []
    atom_ls.extend([atom.GetSymbol() for atom in mol1.GetAtoms()])
    atom_ls.extend([atom.GetSymbol() for atom in mol2.GetAtoms()])
    atom_positions = np.concatenate([mol1.GetConformer().GetPositions(), mol2.GetConformer().GetPositions()],
                                    axis=0)
    try:
        mol_ase = Atoms(symbols=atom_ls, positions=atom_positions)
        mol_ase.set_chemical_symbols(
            list(map(lambda x: x if x in important_atoms_sybols else "X", mol_ase.get_chemical_symbols())))
        res = acsf.create(mol_ase)  # acsf len=187
        res_th = torch.tensor(res, dtype=torch.float)
        if torch.any(torch.isnan(res_th)):
            print('acsf error', key)
            status = False
        g.ndata['h'] = torch.cat([g.ndata['h'], res_th], dim=-1)
    except:
        status = False

    # assign edge features
    # 'd', distance between ligand atoms
    dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
    m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

    # 'd', distance between pocket atoms
    dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

    # 'd', distance between ligand atoms and pocket atoms
    inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]]])
    g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    # efeats1
    g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'), dtype=torch.float)  # init 'e'
    efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

    # efeats2
    efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

    # 'e'
    g1_d = torch.cat([m1_d, m2_d])
    g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
    g3.edata['e'] = g3_d * 0.1

    # init 'pos'
    g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3], dtype=torch.float)
    g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
    g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    g.ndata.pop('pos')
    # detect the nan values in the D3_info_th
    if torch.any(torch.isnan(D3_info_th)):
        status = False
    # except:
    #     print('error:', key)
    #     g = None
    #     g3 = None
    #     status = False
    if status:
        affinity_type_ls = ['Ki', 'Kd', 'IC50']
        label_np = np.zeros(3)
        label_np[affinity_type_ls.index(affinity_type)] = label
        mask_np = np.zeros(3)
        mask_np[affinity_type_ls.index(affinity_type)] = 1.0
        # # 返回参数，需要结合write_jobs函数
        # return {'g': g, 'g3': g3, 'key': key, 'affinity_type': affinity_type, 'label': list(label_np),
        #         'mask': list(mask_np)}
        # 直接写出数据
        with open(graph_file_path + path_marker + key, 'wb') as f:
            pickle.dump({'g': g, 'g3': g3, 'key': key, 'affinity_type': affinity_type, 'label': list(label_np),
                'mask': list(mask_np)}, f)


def write_jobs(jobs, graph_dic_path, path_marker):
    for job in jobs:
        dic = job.get()
        if dic is not None:
            with open(graph_dic_path + path_marker + dic['key'], 'wb') as f:
                pickle.dump(dic, f)


class GraphDatasetIGN(object):
    def __init__(self, bin_file_path):
        self.bin_file_path = bin_file_path
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.bin_file_path):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.bin_file_path, 'rb') as f:
                # data = pickle.load(f)
                data = joblib.load(f)
            self.graphs = data['g']
            self.graphs3 = data['g3']
            self.keys = data['keys']

    def __getitem__(self, indx):
        return self.graphs[indx], self.graphs3[indx], self.keys[indx]

    def __len__(self):
        return len(self.keys)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def collate_fn_ign(data_batch):
    graphs, graphs3, keys = map(list, zip(*data_batch))
    g = dgl.batch(graphs)
    g3 = dgl.batch(graphs3)
    return g, g3, keys


def run_a_eval_epoch(model, validation_dataloader, device):
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # print(i_batch)
            # DTIModel.zero_grad()
            bg, bg3, keys = batch
            bg, bg3 = bg.to(device), bg3.to(device)
            outputs, weights = model(bg, bg3)
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return pred, key


# 截取口袋
def generate_complex(protein_file, ligand_file):
    """
    protein_file: 绝对路径
    ligand_file: 绝对路径
    """
    # get the pocket of protein
    # protein = Chem.MolFromPDBFile(receptor)
    # the content of python file for Chimera
    pocket_file = chimera_pocket_path + ligand_file.split('/')[-1].replace('.sdf', '_pkt.pdb')
    filecontent = "from chimera import runCommand \n"
    filecontent += "runCommand('open 0 %s') \n" % ligand_file
    filecontent += "runCommand('open 1 %s') \n" % protein_file
    filecontent += "runCommand('select #1:.water') \n"
    filecontent += "runCommand('delete selected')\n"
    filecontent += "runCommand('select #1 & #0 z<8') \n"
    filecontent += "runCommand('write format pdb selected 1 %s') \n" % pocket_file
    filecontent += "runCommand('close 0') \n"
    filecontent += "runCommand('close 1')"
    filename = chimera_py_path + ligand_file.split('/')[-1].replace('.sdf', '.py')
    with open(filename, 'w') as f:
        f.write(filecontent)

    try:
        # cmdline = 'chimera --nogui --script select_residues.py'
        # cmdline = '/opt/UCSF/Chimera64-1.10.1/bin/chimera --nogui --silent --script %s' % filename  # the default and ligand pocket contains Hs
        cmdline = 'chimera --nogui --silent --script %s' % filename  # the default and ligand pocket contains Hs
        os.system(cmdline)
        ligand = Chem.SDMolSupplier(ligand_file, sanitize=False)[0]
        m_protein = Chem.MolFromPDBFile(pocket_file, sanitize=False)
        ligand = Chem.RemoveHs(ligand, sanitize=False)  # remove Hs for the ligand
        m_protein = Chem.RemoveHs(m_protein, sanitize=False)  # remove Hs for the ligand
        # write the ligand and pocket to pickle object
        if ligand and m_protein:
            filename = chimera_complex_path + ligand_file.split('/')[-1][:-4]  # 去掉后缀即为复合物的名字
            with open(filename, 'wb') as f:
                pickle.dump([ligand, m_protein], f)
    except:
        print('complex %s generation failed...' % ligand_file)


def write_jobs(jobs, graph_dic_path):
    for job in jobs:
        print(job)
        dic = job.get()
        if dic is not None:
            with open(graph_dic_path + dic['key'], 'wb') as f:
                pickle.dump(dic, f)

# 其他参数
path_marker = '/'
strategies = ['mixture', 'fine-tuning', 'pure-metal']
limit = None   #测试代码用, 正常运行代码时设置为 None
# 传入的参数
# 1. sdfs_path，sdf文件绝对路径; sdf文件命名格式：target_id.sdf, target和id种不要出现下划线，对接构象的存储路径，每个构象将根据target从protein_path去寻找其对应的蛋白。
# 2. protein_path，蛋白文件绝对路径; 文件命名格式：target.pdb, 蛋白文件存储路径。
# 3. temp_path，临时文件绝对路径; 用于保存chimera截取生成的复合物、chimera截取生成的py文件、chimera截取生成的蛋白口袋、dgl单个图对象、bin文件（一个bin文件包含多个dgl图对象）；
#    需要的硬盘空间较大，每10万个预测样本最好预留不少于60G的硬盘空间，预测完成后以上文件和文件夹将被清空。
# 4. dock_engine，构象产生所使用的对接软件，两个选项'glide' 和 'plants'。
# 5. num_process, 口袋截取和dgl单个图对象生成的线程数量。
# 6. csv_path，预测结果文件保存路径
# 7. bin_size, 一个bin文件包含多少个dgl图对象，可根据实际的机器内存大小设置，一般bin_size=10000时,内存峰值在25G左右。
# 8. batch_size, 模型预测时的batch_size，8G显存的GPU可使用的batch_size为512左后。
# 9. work_name
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--sdfs_path', type=str, default='/data/sdfs/', help="the absolute path of sdf files")
    argparser.add_argument('--protein_path', type=str, default='/data/proteins/', help="the absolute path of protein files")
    argparser.add_argument('--temp_path', type=str, default='/data/temp/', help="the absolute path of temp files")
    argparser.add_argument('--dock_engine', type=str, default='plants', help="the pose generation way, only support glide and plants")
    argparser.add_argument('--work_name', type=str, default='123', help="work name")
    argparser.add_argument('--csv_path', type=str, default='/data/prediction_results', help="the path to store the prediction results")
    argparser.add_argument('--num_process', type=int, default=10,
                           help='number of process for pocket truncation and generating graphs')
    argparser.add_argument('--bin_size', type=int, default=5000,
                           help='how many dgl graphs stored in one bin file')
    argparser.add_argument('--batch_size', type=int, default=512,
                           help='batch size for model prediction')
    args = argparser.parse_args()
    sdfs_path, protein_path, temp_path = args.sdfs_path, args.protein_path, args.temp_path
    dock_engine, num_process, csv_path = args.dock_engine, args.num_process, args.csv_path
    bin_size, batch_size, work_name = args.bin_size, args.batch_size, args.work_name
    assert dock_engine in ['glide', 'plants'], 'only support glide, vina or plants for dock_engine'
    # 模型训练参数
    args = argparse.Namespace(d_FC_layer=200, dropout=0.25, edge_feat_size_3d=23,
                              graph_feat_size=128, n_FC_layer=2, n_tasks=3,
                              node_feat_size=227, num_layers=5, outdim_g3=200,
                              repetitions=3, batch_size=512)
    node_feat_size, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_3d
    graph_feat_size, num_layers, batch_size = args.graph_feat_size, args.num_layers, args.batch_size
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks = args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks
    repetitions = args.repetitions

    # 模型权重文件
    #, Ki/Kd/IC50 muti-task training, 30A + ligand smiles preparation, glide
    if dock_engine == 'glide':
        # mixture model_train_ign_metal_mt_622_glide_1.log
        model_files_1 = [
            '../model_save/2022-04-22_10_17_08_319143.pth',
            '../model_save/2022-04-22_11_45_07_831223.pth',
            '../model_save/2022-04-22_13_07_08_503338.pth']
        # fine-tuning model_train_ign_metal_mt_622_glide_ft_1.log
        model_files_2 = ['../model_save/2022-04-25_15_44_42_726494.pth',
                         '../model_save/2022-04-25_16_05_46_345337.pth',
                         '../model_save/2022-04-25_16_45_02_354119.pth']
        # pure metal model_train_ign_metal_mt_622_glide_init_1.log
        model_files_3 = ['../model_save/2022-04-22_15_11_15_68188.pth',
                         '../model_save/2022-04-22_15_35_19_118667.pth',
                         '../model_save/2022-04-22_15_56_41_529420.pth']

    if dock_engine == 'plants':
        # mixture model_train_ign_metal_mt_622_plants_1.log
        model_files_1 = ['../model_save/2022-04-23_19_32_32_731010.pth',
                       '../model_save/2022-04-23_20_39_06_959178.pth',
                       '../model_save/2022-04-23_21_39_31_723573.pth']
        # fine-tuning model_train_ign_metal_mt_622_plants_1.log
        model_files_2 = [
            '../model_save/2022-04-25_15_32_14_310010.pth',
            '../model_save/2022-04-25_15_52_01_466028.pth',
            '../model_save/2022-04-25_16_34_27_998150.pth']

        # pure metal model_train_ign_metal_mt_622_plants_init_1.log
        model_files_3 = ['../model_save/2022-04-23_15_46_39_517472.pth',
                       '../model_save/2022-04-23_16_09_08_333285.pth',
                       '../model_save/2022-04-23_16_31_15_507915.pth']
    if dock_engine == 'crystal':
        # mixture model_train_ign_metal_mt_622_crystal_1.log
        model_files_1 = ['../model_save/2022-04-25_15_00_02_313019.pth',
                         '../model_save/2022-04-25_18_41_26_907070.pth',
                         '../model_save/2022-04-26_00_28_33_176638.pth']
        # Fine-tuning model_train_ign_metal_mt_622_crystal_ft_1.log
        model_files_2 = ['../model_save/2022-04-26_10_47_50_401553.pth',
                         '../model_save/2022-04-26_11_10_51_935372.pth',
                         '../model_save/2022-04-26_11_37_53_657582.pth']
        # pure metal model_train_ign_metal_mt_622_crystal_init_1.log
        model_files_3 = ['../model_save/2022-04-25_15_56_11_619148.pth',
                         '../model_save/2022-04-25_16_24_44_610947.pth',
                         '../model_save/2022-04-25_17_16_06_734181.pth']


    # 先清空临时文件夹
    if os.path.exists(temp_path):
        # 已存在, 先清空文件夹里的内容
        cmdline = 'rm -rf %s*' % temp_path
        os.system(cmdline)
    else:
        # 不存在, 则先建立临时文件夹
        cmdline = 'mkdir -p %s' % temp_path
        os.system(cmdline)

    # 创建临时文件夹中的子文件夹
    chimera_complex_path = temp_path + 'chimera_complex/'  # chimera截取生成的复合物路径
    cmdline = 'mkdir -p %s' % chimera_complex_path
    os.system(cmdline)
    chimera_py_path = temp_path + 'chimera_py/'  # chimera截取生成的py文件复合物路径
    cmdline = 'mkdir -p %s' % chimera_py_path
    os.system(cmdline)
    chimera_pocket_path = temp_path + 'chimera_pocket/'  # chimera截取生成的蛋白口袋路径
    cmdline = 'mkdir -p %s' % chimera_pocket_path
    os.system(cmdline)
    graph_file_path = temp_path + 'dgl_graphs/'  # dgl单个图对象存储路径
    cmdline = 'mkdir -p %s' % graph_file_path
    os.system(cmdline)
    bin_file_path = temp_path + 'bin_files/'  # bin文件存储路径，一个bin文件包含多个dgl图对象
    cmdline = 'mkdir -p %s' % bin_file_path
    os.system(cmdline)

    cmdline = 'mkdir -p %s' % csv_path  # make csv_file_path
    os.system(cmdline)

    # 获取配体
    ligand_files = os.listdir(sdfs_path)[:limit]
    ligand_files = [file for file in ligand_files if file.endswith('.sdf')]
    print('the number of sdf files is:', len(ligand_files), '\n')
    # 第一步，截取口袋和产生复合物
    st = time.time()
    print("step1 process start: pocket truncation using chimera")
    protein_files = [protein_path + file.split('_')[0] + '_protein.pdb' for file in ligand_files]
    ligand_files = [sdfs_path + file for file in ligand_files]
    pool = mp.Pool(num_process)
    pool.starmap_async(generate_complex, zip(protein_files, ligand_files))
    pool.close()
    pool.join()
    # 复合物已生产，可删除chimera_py_path和chimera_pocket_path中的文件
    cmdline = 'cd %s && ls | xargs rm -r' % chimera_py_path  # 删除文件
    os.system(cmdline)
    cmdline = 'rm -rf %s' % chimera_py_path  # 删除空文件夹
    os.system(cmdline)
    cmdline = 'cd %s && ls | xargs rm -r' % chimera_pocket_path  # 删除文件
    os.system(cmdline)
    cmdline = 'rm -rf %s' % chimera_pocket_path  # 删除空文件夹
    os.system(cmdline)
    print("step1 process end (time:%s S)\n" % (time.time() - st))

    # 第二步，生成dgl图对象
    print("step2 process start: generating dgl graphs")
    complex_files = os.listdir(chimera_complex_path)
    st = time.time()
    # # 采用参数回调的方式
    # pool = mp.Pool(num_process)
    # jobs = []
    # for complex_file in complex_files:
    #     complex_file_path = chimera_complex_path + complex_file
    #     p = pool.apply_async(partial(graphs_from_mol_ign, label=0, affinity_type='IC50', dis_threshold=8.0),
    #                          args=(complex_file_path, complex_file))
    #     jobs.append(p)
    #     if len(jobs) == 100:
    #         write_jobs(jobs, graph_dic_path=graph_file_path)
    #         jobs = []
    # write_jobs(jobs, graph_dic_path=graph_file_path)  #这条命令在某些机器上会有问题
    # pool.close()
    # pool.join()

    # 直接将每个样本生成的复合物写出到硬盘, 可能会比较吃内存？
    complex_file_paths = [chimera_complex_path + file for file in complex_files]
    pool = mp.Pool(num_process)
    pool.starmap_async(partial(graphs_from_mol_ign, label=0, affinity_type='IC50', dis_threshold=8.0), zip(complex_file_paths, complex_files))
    pool.close()
    pool.join()


    # 完成第二步
    # 删除复合物文件夹
    cmdline = 'cd %s && ls | xargs rm -r' % chimera_complex_path  # 删除文件
    os.system(cmdline)
    cmdline = 'rm -rf %s' % chimera_complex_path  # 删除空文件夹
    os.system(cmdline)
    print("step2 process end (time:%s S)\n" % (time.time() - st))

    # 第三步，将生成的图对象制作成bin文件，每个bin文件10000个图对象，1个图对象对应于1个复合物
    print("step3 process start: making bin files")
    num_suc_sanmples = len(os.listdir(graph_file_path))
    print('the number of successful processed samples is:', num_suc_sanmples)
    st = time.time()
    graph_files = os.listdir(graph_file_path)
    graphs = []
    graphs3 = []
    keys = []
    bin_file_th = 0
    for idx, graph_file in enumerate(graph_files):
        graph_file = graph_file_path + graph_file
        with open(graph_file, 'rb') as f:
            data = pickle.load(f)
        keys.append(data['key'])
        graphs.append(data['g'])
        graphs3.append(data['g3'])
        if (idx != 0) and (idx % bin_size == 0):
            bin_file = bin_file_path+str(bin_file_th) +'.bin'
            with open(bin_file, 'wb') as f:
                # pickle.dump({'g': graphs, 'g3': graphs3, 'keys': keys}, f)
                joblib.dump({'g': graphs, 'g3': graphs3, 'keys': keys}, f, compress=2)

            bin_file_th = bin_file_th + 1
            graphs, graphs3, keys = [], [], []
    # 最后一部分
    bin_file = bin_file_path+str(bin_file_th)+'.bin'
    with open(bin_file, 'wb') as f:
        # pickle.dump({'g': graphs, 'g3': graphs3, 'keys': keys}, f)
        joblib.dump({'g': graphs, 'g3': graphs3, 'keys': keys}, f, compress=2)
    graphs, graphs3, keys = [], [], []

    # 完成第三步，可删除相应的dgl graphs
    cmdline = 'cd %s && ls | xargs rm -r' % graph_file_path  # 删除文件
    os.system(cmdline)
    cmdline = 'rm -rf %s' % graph_file_path  # 删除空文件夹
    os.system(cmdline)
    print("step3 process end (time:%s S)\n" % (time.time() - st))

    # 加载第3步的bin文件，用训练好的ign预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTIModel = IGN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d,
                   num_layers=num_layers,
                   graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                   d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout,
                   n_tasks=n_tasks, task_type='regression')

    print("step4 process start: making predictions")
    # 不同训练策略下的预测
    for strategy in strategies:
        if strategy == 'mixture':
            model_files = model_files_1
        elif strategy == 'fine-tuning':
            model_files = model_files_2
        else:
            model_files = model_files_3
        st = time.time()
        bin_files = os.listdir(bin_file_path)
        total_pred1_ki, total_pred2_ki, total_pred3_ki = [], [], []
        total_pred1_kd, total_pred2_kd, total_pred3_kd = [], [], []
        total_pred1_ic50, total_pred2_ic50, total_pred3_ic50 = [], [], []
        total_keys = []
        for bin_file in bin_files[:]:
            bin_file = bin_file_path + bin_file
            test_dataset = GraphDatasetIGN(bin_file_path=bin_file)
            total_keys.extend(test_dataset.keys)
            print('bin_file:', bin_file, 'data_points:', len(test_dataset))
            for repetition_th in range(repetitions):
                test_dataset_cp = copy.deepcopy(test_dataset)
                DTIModel = copy.deepcopy(DTIModel)
                DTIModel.load_state_dict(torch.load(
                    f=model_files[repetition_th], map_location='cpu')['model_state_dict'])
                DTIModel.to(device)
                test_dataloader = DataLoaderX(test_dataset_cp, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn_ign)
                test_pred, test_keys = run_a_eval_epoch(DTIModel, test_dataloader, device)

                # test_pred = np.concatenate(np.array(test_pred), 0).flatten()
                test_pred = np.concatenate(np.array(test_pred), 0)
                # test_keys = np.concatenate(np.array(test_keys), 0).flatten()

                if repetition_th == 0:
                    total_pred1_ki.extend(test_pred[:, 0])
                    total_pred1_kd.extend(test_pred[:, 1])
                    total_pred1_ic50.extend(test_pred[:, 2])
                elif repetition_th == 1:
                    total_pred2_ki.extend(test_pred[:, 0])
                    total_pred2_kd.extend(test_pred[:, 1])
                    total_pred2_ic50.extend(test_pred[:, 2])
                else:
                    total_pred3_ki.extend(test_pred[:, 0])
                    total_pred3_kd.extend(test_pred[:, 1])
                    total_pred3_ic50.extend(test_pred[:, 2])
                del test_dataset_cp
                del test_dataloader
                gc.collect()
            del test_dataset
            gc.collect()

        total_average_ki = (np.array(total_pred1_ki) + np.array(total_pred2_ki) + np.array(total_pred3_ki)) / repetitions
        total_average_kd = (np.array(total_pred1_kd) + np.array(total_pred2_kd) + np.array(total_pred3_kd)) / repetitions
        total_average_ic50 = (np.array(total_pred1_ic50) + np.array(total_pred2_ic50) + np.array(
            total_pred3_ic50)) / repetitions

        final_res_ki = pd.DataFrame(
            {'keys': total_keys, 'pred1': total_pred1_ki, 'pred2': total_pred2_ki, 'pred3': total_pred3_ki,
             'average': total_average_ki})
        final_res_ki.to_csv(csv_path + 'predict_%s_%s_ki_%s.csv' % (dock_engine, work_name, strategy), index=False)

        final_res_kd = pd.DataFrame(
            {'keys': total_keys, 'pred1': total_pred1_kd, 'pred2': total_pred2_kd, 'pred3': total_pred3_kd,
             'average': total_average_kd})
        final_res_kd.to_csv(csv_path + 'predict_%s_%s_kd_%s.csv' % (dock_engine, work_name, strategy), index=False)

        final_res_ic50 = pd.DataFrame(
            {'keys': total_keys, 'pred1': total_pred1_ic50, 'pred2': total_pred2_ic50, 'pred3': total_pred3_ic50,
             'average': total_average_ic50})
        final_res_ic50.to_csv(csv_path + 'predict_%s_%s_ic50_%s.csv' % (dock_engine, work_name, strategy), index=False)

    # 完成第四步，可删除相应的bin file
    cmdline = 'cd %s && ls | xargs rm -r' % bin_file_path  # 删除文件
    os.system(cmdline)
    cmdline = 'rm -rf %s' % temp_path  # 删除临时文件夹
    os.system(cmdline)
    print("step4 process end (time:%s S)\n" % (time.time() - st))
