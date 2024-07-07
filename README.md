# MetalProGNet
MetalProGNet: A Structure-based Deep Graph Model Specific For Metalloprotein-Ligand Interaction Predictions; MetalProGNet was developed based on our previous [IGN](https://github.com/zjujdj/InteractionGraphNet/tree/master) framework.
![Image text](https://github.com/zjujdj/MetalProGNet/blob/master/fig/workflow.jpg)

- **Step1: Clone the Repository**
```python
git clone https://github.com/zjujdj/MetalProGNet.git
```
- **Step2: Download the trained models and conda-packed env**

download url: [dgl430_py37_gpu.tar.gz](https://drive.google.com/file/d/10k32qTk80a7kfgu2MDR4bwYp8lx_s-74/view); [model_save.zip](https://drive.google.com/file/d/16WqXOJs0bVxatpgZHkgSZOKdch_Q2sdP/view?usp=sharing)
```python
cd MetalProGNet && unzip model_save.zip && tar -xzvf dgl430_py37_gpu.tar.gz -C /home/conda_env/dgl430_py37_gpu
source activate /home/conda_env/ dgl430_py37_gpu
conda-unpack
```
- **Step3: Give executable privileges to the PLANTS program**
```python
chmod +x ../plants/PLANTS1.2_64bit ../plants/SPORES_64bit
```

- **Step4: Using plants_protein_prep.py for protein preparation**
```python
python3  plants_protein_prep.py --num_process=4 --receptor_path=../receptors/ --plants_path=../plants/
```

- **Step5: Using plants_ligand_prep.py for ligand preparation**
```python
python3 plants_ligand_prep.py --num_process=10 --ligand_file_path=../ligand_file/ --plants_path=../plants/ --temp_path=../temp/ --dst_path=../prepared_ligands/
```

- **Step6: Using plants_docking.py for molecular docking**
```python
cd .. && rm -rf docking_runing && mkdir -p docking_running
cp ./scripts/plants_docking.py docking_running
cd docking_running
python3 plants_docking.py --num_process=10 --top1_pose_path=../top1_pose/ --top1_pose_sdf_path=../top1_pose_sdf/ --receptor_path=../receptors/  --plants_path=../plants/  --dst_path=../prepared_ligands/  --config_file_path=../config_file/  --docking_running_path=../docking_running/
cd .. && rm -rf docking_running
```

- **Step7: Using predict_mt_chembl.py for binding affinity prediction**
```python
cd scripts
python3 predict_mt_chembl.py --num_process=10 --bin_size=5 --batch_size=512 --sdfs_path=../top1_pose_sdf/ --protein_path=../receptors/ --temp_path=../temp/ --dock_engine=plants --csv_path=../csv_files/ --work_name=test
```

## More Info Is Available In ./scripts/readme.txt
