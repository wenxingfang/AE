#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=ae
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/C14Mix/AE/log_train.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/C14Mix/AE/log_train.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000
#
# Specify how many GPU cards to us:
#SBATCH --gres=gpu:v100:1
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
hostname
df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup_conda.sh
conda activate pytorch1.71 
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/C14Mix/AE/
python $workpath/train_ae.py  --smooth 0 --epochs 200 --lr 3e-4 --loss 'mae' --batch 128 --clip_grad 2 --scheduler 'Plateau' --train_file $workpath/dataset/train_ep.txt --valid_file $workpath/dataset/valid_ep.txt --test_file $workpath/dataset/test_ep.txt --channel 0 --npe_scale 5 --out_name $workpath/model/AE_npe_0MeV_mae.pth
#python $workpath/train_ae.py  --smooth 0 --epochs 200 --lr 3e-4 --batch 128 --clip_grad 2 --scheduler 'Plateau' --train_file $workpath/dataset/train_ep.txt --valid_file $workpath/dataset/valid_ep.txt --test_file $workpath/dataset/test_ep.txt --channel 0 --npe_scale 5 --out_name $workpath/model/AE_npe_0MeV_ClipGrad.pth
#python $workpath/train_ae.py  --smooth 0 --epochs 200 --lr 4e-4 --batch 128 --train_file $workpath/dataset/train_ep.txt --valid_file $workpath/dataset/valid_ep.txt --test_file $workpath/dataset/test_ep.txt --channel 0 --npe_scale 5 --out_name $workpath/model/AE_npe_0MeV_lr.pth
#python $workpath/train_ae.py  --smooth 0 --epochs 200 --batch 128 --train_file $workpath/dataset/train_ep.txt --valid_file $workpath/dataset/valid_ep.txt --test_file $workpath/dataset/test_ep.txt --channel 0 --npe_scale 5 --out_name $workpath/model/AE_npe_0MeV.pth
#python $workpath/train_ae.py  --smooth 0 --epochs 200 --batch 128 --train_file $workpath/dataset/train_ep.txt --valid_file $workpath/dataset/valid_ep.txt --test_file $workpath/dataset/test_ep.txt --channel 0 --npe_scale 1 --out_name $workpath/model/AE_npe.pth
