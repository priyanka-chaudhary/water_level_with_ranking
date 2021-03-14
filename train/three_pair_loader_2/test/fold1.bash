#BSUB -W 4:00
#BSUB -o /cluster/work/igp_psr/pchaudha/flood/rank/exp/three_pair_loader/three_pair_loader_7/test/test_result_fold1.txt
#BSUB -e /cluster/work/igp_psr/pchaudha/flood/rank/exp/three_pair_loader/three_pair_loader_7/test/error_fold1.txt
#BSUB -n 1
#BSUB -R "rusage[mem=32768,ngpus_excl_p=1]"
#### BEGIN #####
module load python_gpu/3.7.1
module load hdf5/1.10.1
module load eth_proxy

pip install --user tensorboardX
pip install -U --user protobuf
python3 -c 'import keras; print(keras.__version__)'
python3 fold1.py

#### END #####
