### 第一步，运行plants_protein_prep.py进行蛋白准备
# 传入的参数
# 1. receptor_path, 需要进行准备的蛋白文件绝对路径, 蛋白文件需用薛定谔（protein preparation wizard）等软件提前处理(蛋白文件命名格式: pdbid_protein.pdb), 且此处最好保留住金属离子跟蛋白原子的螯合键信息，pdb文件中的record记录
# 2. plants_path, plants程序的绝对路径
# 3. num_process, 线程数量
# 最终准备好的蛋白文件存储于receptor_path中，文件格式 pdbid_protein_sp.pdb，但是金属离子跟蛋白原子的螯合键信息会被SPORES_64bit程序清理掉.
# 运行示例： python3  plants_protein_prep.py --num_process=4 --receptor_path=/data/wspy/data/ign_metal_prediction_pipeline/receptors/ --plants_path=/data/wspy/data/ign_metal_prediction_pipeline/plants/

### 第二步，运行plants_ligand_prep.py进行配体准备
# 传入的参数
# 1. ligand_file_path，配体文件存储的绝对路径，csv文件，需包含两个字段smiles和name(不要有特殊字符，唯一标记每个分子)；程序自动在ligand_file_path路径下匹配csv文件
# 2. plants_path, plants程序的绝对路径
# 3. num_process, 线程数量
# 4. temp_path, 临时文件路径，用于存储每个分子的smi文件，程序运行完后该文件夹会被删除; 存在则会自动清空，不存在将自动创建
# 5. dst_path, 目标文件路径，用于存储每个分子准备好的文件，文件命令格式 name_sp.mol2, 该文件夹下的内容将用于下一步的对接；存在则会自动清空，不存在将自动创建。
# 运行示例：python3  plants_ligand_prep.py --num_process=10 --ligand_file_path=/data/wspy/data/ign_metal_prediction_pipeline/ligand_file/ --plants_path=/data/wspy/data/ign_metal_prediction_pipeline/plants/ --temp_path=/data/wspy/data/ign_metal_prediction_pipeline/temp/ --dst_path=/data/wspy/data/ign_metal_prediction_pipeline/prepared_ligands/


#### 第三步， 运行plants_docking.py进行分子对接
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
# 5. cd /data/wspy/data/ign_metal_prediction_pipeline && rm -rf docking_running

### 第四步，运行predict_mt_chembl.py
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
# 运行示例：
# python3  predict_mt_chembl.py --num_process=10 --bin_size=5 --batch_size=512 --sdfs_path=/data/wspy/data/ign_metal_prediction_pipeline/top1_pose_sdf/ --protein_path=/data/wspy/data/ign_metal_prediction_pipeline/receptors/ --temp_path=/data/wspy/data/ign_metal_prediction_pipeline/temp/ --dock_engine=plants --csv_path=/data/wspy/data/ign_metal_prediction_pipeline/csv_files/ --work_name=test