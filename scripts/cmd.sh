# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"
#export mkgraph_cmd="queue.pl -l arch=*64,ram_free=4G,mem_free=4G"
#export big_memory_cmd="queue.pl -l arch=*64,ram_free=8G,mem_free=8G"
#export cuda_cmd="queue.pl -l gpu=1"

#c) run it locally... works for CMU rocks cluster
#export train_cmd="slurm.pl -p 4gpuq -N 3 --gpu 4 --mem 122880"
#export train_cmd=slurm.pl
export train_cmd="slurm.pl"
export cpu_cmd="slurm.pl --qos qm15 --mem 20G --num-threads 1"
export decode_cmd="slurm.pl -p gpu,2080ti --gpu 1 --mem 120G --num-threads 10"
export genlat_cmd="slurm.pl -p gpu,2080ti --gpu 1 --mem 10G --num-threads 10"
export cuda_cmd="slurm.pl -p gpu,2080ti --gpu 1 --mem 20G --num-threads 3"
export local_cmd="run.pl"
