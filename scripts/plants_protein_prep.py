# 蛋白准备。
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import argparse

# pdb_file文件格式：/%s/%s/pdbid_protein.pdb
def process(pdb_file):
    cmdline = 'babel -ipdb %s -omol2 %s' % (pdb_file, pdb_file.replace('.pdb', '.mol2'))
    os.system(cmdline)

    # 准备蛋白结构 (利用plants自带工具, SPORES_64bit)
    # 利用plants准备好的蛋白文件格式, /%s/%s/pdbid_protein_sp.mol2
    cmdline = '%sSPORES_64bit --mode complete %s %s' % (plants_path, pdb_file.replace('.pdb', '.mol2'), pdb_file.replace('.pdb', '_sp.mol2'))
    os.system(cmdline)


# 传入的参数
# 1. receptor_path, 需要进行准备的蛋白文件绝对路径, 蛋白文件需用薛定谔（protein preparation wizard）等软件提前处理(蛋白文件命名格式: pdbid_protein.pdb), 且此处最好保留住金属离子跟蛋白原子的螯合键信息，pdb文件中的record记录
# 2. plants_path, plants程序的绝对路径
# 3. num_process, 线程数量
# 最终准备好的蛋白文件存储于receptor_path中，文件格式 pdbid_protein_sp.pdb，但是金属离子跟蛋白原子的螯合键信息会被SPORES_64bit程序清理掉.
# 运行示例：python3  plants_protein_prep.py --num_process=10 --receptor_path=/data/wspy/data/ign_metal_prediction_pipeline/receptors/ --plants_path=/data/wspy/data/ign_metal_prediction_pipeline/plants/
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--receptor_path', type=str, default='/data/proteins/', help="the absolute path of receptor files")
    argparser.add_argument('--plants_path', type=str, default='/data/plants/', help="the absolute path of plants docking program")
    argparser.add_argument('--num_process', type=int, default=4,
                           help='number of process for protein preparation using SPORES_64bit')
    args = argparser.parse_args()
    receptor_path, plants_path, num_process = args.receptor_path, args.plants_path, args.num_process
    files = os.listdir(receptor_path)
    pdb_files = [file for file in files if file.endswith('protein.pdb')]
    pdb_files = [receptor_path + file for file in pdb_files]
    pool = mp.Pool(num_process)
    pool.starmap_async(process, zip(pdb_files))
    pool.close()
    pool.join()
