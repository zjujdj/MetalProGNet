# 利用plants进行对接
import pandas as pd
import numpy as np
import os
import sys
import multiprocessing as mp
import argparse


def get_center_from_mol2(ligand_mol2):
    # 根据共晶配体确定坐标
    x = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}' | awk '{x+=$1} END {print x/(NR-2)}'" % ligand_mol2).read()
    y = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}' | awk '{y+=$1} END {print y/(NR-2)}'" % ligand_mol2).read()
    z = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}' | awk '{z+=$1} END {print z/(NR-2)}'" % ligand_mol2).read()
    xyz = [float(x.strip()), float(y.strip()), float(z.strip())]
    return xyz


def write_config_file(protein_file, ligand_file, config_file, output_dir, x, y, z):
        content = f'''# scoring function and search settings
# plp|plp95|chemplp
scoring_function chemplp
search_speed speed1


# input
protein_file {protein_file}
ligand_file {ligand_file}

# output
output_dir {output_dir}

# write single mol2 files (e.g. for RMSD calculation)
write_multi_mol2 0
write_per_atom_scores 0
write_protein_conformations 0
write_protein_bindingsite 0

# binding site definition
bindingsite_center {x} {y} {z}
bindingsite_radius 20


# cluster algorithm
cluster_structures 10
cluster_rmsd 2.0
'''
        with open(config_file, 'w') as f:
            f.write(content)


def plants_docking(config_file):
    cmd = f'timeout 5m %sPLANTS1.2_64bit --mode screen %s' % (plants_path, config_file)
    os.system(cmd)


def docking_results_collect(dir, pdb_id, ligand_id):
    score_csv = '%sranking.csv' % dir
    top1_pose = '%s*****_entry_00001_conf_01.mol2' % dir
    if os.path.exists(score_csv) and os.path.exists(top1_pose):
        docking_score = pd.read_csv(score_csv).TOTAL_SCORE.values[0]
        cmdline = 'mv %s %s%s_%s_%.4f.mol2' % (top1_pose, top1_pose_path, pdb_id, ligand_id, docking_score)
        os.system(cmdline)
        cmdline = 'rm -rf %s' % dir
        os.system(cmdline)


def plants_docking_pipeline(pdb_id, ligand_id):
    protein_mol2_file = receptor_path + '%s_protein_sp.mol2' % pdb_id
    ligand_mol2_file = dst_path + '%s_sp.mol2' % ligand_id
    crystal_mol2_file = receptor_path + '%s_ligand.mol2' % pdb_id
    output_dir = '%s_%s' % (pdb_id, ligand_id)
    xyz = get_center_from_mol2(crystal_mol2_file)
    config_file = '%s%s_%s.conf' % (config_file_path, pdb_id, ligand_id)
    # 写出对接的config文件
    write_config_file(protein_mol2_file, ligand_mol2_file, config_file, output_dir, xyz[0], xyz[1], xyz[2])
    plants_docking(config_file)

    # the docking results absoute path
    dir = '%s%s_%s/' % (docking_running_path, pdb_id, ligand_id)
    docking_results_collect(dir, pdb_id, ligand_id)

def convert(mol2_file, sdf_file):
    cmdline = 'obabel %s -O %s' % (mol2_file, sdf_file)
    os.system(cmdline)

# 传入的参数
# 1. top1_pose_path, 用于存储plants对接程序为每个分子所产生的top-1构象(mol2格式). 程序运行完后该文件夹会被删除; 存在则会自动清空，不存在将自动创建
# 2. top1_pose_sdf_path, 用于存储plants对接程序为每个分子所产生的top-1构象(sdf格式). 存在则会自动清空，不存在将自动创建
# 3. receptor_path, 准备好的蛋白文件(格式: pdbid_protein_sp.mol2),以及蛋白口袋定位的小分子文件(格式:pdbid_ligand.mol2). 与plants_protein_prep.py文件共享此参数
# 4. plants_path, plants程序的绝对路径.
# 5. config_file_path, 用于存储plants对接程序为每个分子所产生的config file. 程序运行完后该文件夹会被删除; 存在则会自动清空，不存在将自动创建
# 6. docking_running_path, 对接过程运行的实际路径，每个分子对接过程产生的临时文件将在次文件夹中。需要手动创建, 程序运行完后该文件夹会被删除；plants_docking.py将在此文件夹下运行
# 7. dst_path, 目标文件路径，用于存储每个分子准备好的文件，文件命令格式 name_sp.mol2。与plants_ligand_prep.py文件共享此参数
# 8. num_process, 线程数量
# 运行示例：
# 1. cd /data/wspy/data/ign_metal_prediction_pipeline && rm -rf docking_runing && mkdir -p docking_running
# 2. cp /data/wspy/data/ign_metal_prediction_pipeline/scripts/plants_docking.py docking_running
# 3. cd docking_running
# 4. python3  plants_docking.py --num_process=10 --top1_pose_path=/data/wspy/data/ign_metal_prediction_pipeline/top1_pose/ --top1_pose_sdf_path=/data/wspy/data/ign_metal_prediction_pipeline/top1_pose_sdf/  --receptor_path=/data/wspy/data/ign_metal_prediction_pipeline/receptors/  --plants_path=/data/wspy/data/ign_metal_prediction_pipeline/plants/  --dst_path=/data/wspy/data/ign_metal_prediction_pipeline/prepared_ligands/  --config_file_path=/data/wspy/data/ign_metal_prediction_pipeline/config_file/  --docking_running_path=/data/wspy/data/ign_metal_prediction_pipeline/docking_running/
# 5. cd /data/wspy/data/ign_metal_prediction_pipeline && rm -rf docking_runing

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_process', type=int, default=4,
                           help='number of process for protein preparation using SPORES_64bit')
    argparser.add_argument('--top1_pose_path', type=str, default='/data/top1_pose/', help="the absolute path for storing the top-1 pose(mol2)")
    argparser.add_argument('--top1_pose_sdf_path', type=str, default='/data/top1_pose_sdf/', help="the absolute path for storing the top-1 pose(sdf)")
    argparser.add_argument('--receptor_path', type=str, default='/data/proteins/', help="the absolute path of receptor files")
    argparser.add_argument('--plants_path', type=str, default='/data/plants/', help="the absolute path of plants docking program")
    argparser.add_argument('--dst_path', type=str, default='/data/ligands/', help="the absolute path for saving the prepared ligands")
    argparser.add_argument('--config_file_path', type=str, default='/data/config_file/', help="the absolute path for saving the config file used for docking")
    argparser.add_argument('--docking_running_path', type=str, default='/data/config_file/', help="the absolute path for saving the temp file used for docking")
    args = argparser.parse_args()
    top1_pose_path, top1_pose_sdf_path, receptor_path, plants_path, num_process = args.top1_pose_path, args.top1_pose_sdf_path, args.receptor_path, args.plants_path, args.num_process
    dst_path, config_file_path, docking_running_path  = args.dst_path, args.config_file_path, args.docking_running_path

    # 先清空top1_pose_path文件夹
    if os.path.exists(top1_pose_path):
        # 已存在, 先清空文件夹里的内容
        cmdline = 'rm -rf %s*' % top1_pose_path
        os.system(cmdline)
    else:
        # 不存在, 则先建立临时文件夹
        cmdline = 'mkdir -p %s' % top1_pose_path
        os.system(cmdline)

    # 先清空top1_pose_sdf_path文件夹
    if os.path.exists(top1_pose_sdf_path):
        # 已存在, 先清空文件夹里的内容
        cmdline = 'rm -rf %s*' % top1_pose_sdf_path
        os.system(cmdline)
    else:
        # 不存在, 则先建立临时文件夹
        cmdline = 'mkdir -p %s' % top1_pose_sdf_path
        os.system(cmdline)

    # 先清空config_file_path文件夹
    if os.path.exists(config_file_path):
        # 已存在, 先清空文件夹里的内容
        cmdline = 'rm -rf %s*' % config_file_path
        os.system(cmdline)
    else:
        # 不存在, 则先建立临时文件夹
        cmdline = 'mkdir -p %s' % config_file_path
        os.system(cmdline)

    # 先清空docking_running_path文件夹
    if os.path.exists(docking_running_path):
        pass
    else:
        print('please make the docking running path: %s' % docking_running_path)
        sys.exit(1)

    files = os.listdir(receptor_path)
    proteins = [_ for _ in files if _.endswith('protein_sp.mol2')]  # 获取plants准备好的蛋白文件
    pdb_ids = [_.split('_')[0] for _ in proteins]


    ligands = os.listdir(dst_path)
    ligand_ids = [_.split('_')[0] for _ in ligands]

    # pdb_id, ligand_id
    for i, pdb_id in enumerate(pdb_ids):
        pdbs = [pdb_id for _ in ligand_ids]
        pool = mp.Pool(32)
        pool.starmap_async(plants_docking_pipeline, zip(pdbs, ligand_ids))
        pool.close()
        pool.join()

    mol2_files = os.listdir(top1_pose_path)
    mol2_paths = [top1_pose_path + _ for _ in mol2_files]
    sdf_paths = [top1_pose_sdf_path + _.replace('.mol2', '.sdf') for _ in mol2_files]
    pool = mp.Pool(32)
    pool.starmap_async(convert, zip(mol2_paths, sdf_paths))
    pool.close()
    pool.join()

    # 最后删除无用文件夹
    cmdline = 'rm -rf %s' % top1_pose_path
    os.system(cmdline)

    cmdline = 'rm -rf %s' % config_file_path
    os.system(cmdline)

