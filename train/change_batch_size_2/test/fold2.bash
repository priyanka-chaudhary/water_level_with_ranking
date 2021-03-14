#BSUB -W 4:00
#BSUB -o /cluster/work/igp_psr/pchaudha/flood/rank/exp/change_batch_size/change_batch_size_6/test/test_result_fold2.txt
#BSUB -e /cluster/work/igp_psr/pchaudha/flood/rank/exp/change_batch_size/change_batch_size_6/test/error_fold2.txt
#BSUB -n 1
#BSUB -R "rusage[mem=32768,ngpus_excl_p=1]"
#### BEGIN #####
module load python_gpu/3.7.1
module load hdf5/1.10.1
module load eth_proxy

pip install --user tensorboardX
pip install -U --user protobuf
python3 -c 'import keras; print(keras.__version__)'
python3 fold2.py

#### END #####
