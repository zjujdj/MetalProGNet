# 配体准备
import multiprocessing as mp
import pandas as pd
import os
import argparse


def ligand_prep(name, smi):
    #1. 写出smi文件
    with open('%s%s.smi' % (temp_path, name), 'w') as f:
        f.write(smi)

    #2. 用obabel产生smi的3d构象,
    # obgen输出的分子格式为sdf格式，obabel将sdf转化为mol2格式时会自动添加原子的部分电荷信息
    cmdline = 'obgen -ff MMFF94 %s%s.smi > %s%s.sdf && obabel %s%s.sdf -O %s%s.mol2' % (temp_path, name, temp_path, name, temp_path, name, temp_path, name)
    os.system(cmdline)

    #3. 利用SPORES_64bit准备配体，但是准备之后的配体电荷信息好像没了。估计在对接过程中计算
    cmdline = '%sSPORES_64bit --mode complete %s%s.mol2 %s%s_sp.mol2' % (plants_path, temp_path, name, dst_path, name)
    os.system(cmdline)


# 传入的参数
# 1. ligand_file_path，配体文件存储的绝对路径，csv文件，需包含两个字段smiles和name(不要有特殊字符，唯一标记每个分子)
# 2. plants_path, plants程序的绝对路径
# 3. num_process, 线程数量
# 4. temp_path, 临时文件路径，用于存储每个分子的smi文件，程序运行完后该文件夹会被删除; 存在则会自动清空，不存在将自动创建
# 5. dst_path, 目标文件路径，用于存储每个分子准备好的文件，文件命令格式 name_sp.mol2, 该文件夹下的内容将用于下一步的对接；存在则会自动清空，不存在将自动创建。
# 运行示例：python3  plants_ligand_prep.py --num_process=10 --ligand_file_path=/data/wspy/data/ign_metal_prediction_pipeline/ligand_file/ --plants_path=/data/wspy/data/ign_metal_prediction_pipeline/plants/ --temp_path=/data/wspy/data/ign_metal_prediction_pipeline/temp/ --dst_path=/data/wspy/data/ign_metal_prediction_pipeline/prepared_ligands/


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ligand_file_path', type=str, default='/data/ligand_csv_file/', help="the absolute path of ligand files")
    argparser.add_argument('--plants_path', type=str, default='/data/plants/', help="the absolute path of plants docking program")
    argparser.add_argument('--num_process', type=int, default=4,
                           help='number of process for protein preparation using SPORES_64bit')
    argparser.add_argument('--temp_path', type=str, default='/data/temp_path/',
                           help="the absolute path of temporary files")
    argparser.add_argument('--dst_path', type=str, default='/data/ligands/',
                           help="the absolute path for saving the prepared ligands")
    args = argparser.parse_args()
    ligand_file_path, plants_path, num_process, temp_path, dst_path  = args.ligand_file_path, args.plants_path, args.num_process, args.temp_path, args.dst_path
    files = os.listdir(ligand_file_path)
    for file in files:
        if file.endswith('.csv'):
            break
    ligand_csv_file = ligand_file_path + file

    # 先清空临时文件夹
    if os.path.exists(temp_path):
        # 已存在, 先清空文件夹里的内容
        cmdline = 'rm -rf %s*' % temp_path
        os.system(cmdline)
    else:
        # 不存在, 则先建立临时文件夹
        cmdline = 'mkdir -p %s' % temp_path
        os.system(cmdline)

    # 先清空准备好配体的存储文件夹
    if os.path.exists(dst_path):
        # 已存在, 先清空文件夹里的内容
        cmdline = 'rm -rf %s*' % dst_path
        os.system(cmdline)
    else:
        # 不存在, 则先建立文件夹
        cmdline = 'mkdir -p %s' % dst_path
        os.system(cmdline)

    all_ligands = pd.read_csv(ligand_csv_file)
    names = list(all_ligands['name'].values)
    smis = list(all_ligands['smiles'].values)
    pool = mp.Pool(num_process)
    pool.starmap_async(ligand_prep, zip(names, smis))
    pool.close()
    pool.join()

    # 删除临时文件夹
    cmdline = 'rm -rf %s' % temp_path  # 删除临时文件夹
    os.system(cmdline)
