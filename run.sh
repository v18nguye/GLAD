#!/bin/bash 
# nohup sh run_job.sh -d com -t bridge -e com_fix > exp_com_fix.out 2>&1  &

export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR=${PWD}'/data/' # absoluate path to the data directotory.

while getopts d:t:e:n: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        t) task=${OPTARG};;
        e) experiment=${OPTARG};;
        n) name=${OPTARG};;
    esac
done

echo "run dataset: $dataset, task: $task, experiment: $experiment, name: $name";

if [ "$task" == "base" ]
    then
        python ./run_base.py data=$dataset +$experiment=$name

elif [ "$task" == "bridge" ]
    then
        python ./run_bridge.py data=$dataset +$experiment=$name

elif [ "$task" == "sample" ]
    then 
        python ./run_sampler.py data=$dataset +$experiment=$name

else 
    echo "no task to run !!!"
fi