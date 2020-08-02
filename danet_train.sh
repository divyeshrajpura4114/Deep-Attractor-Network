#!/usr/bin/env bash

stage=-1
num_of_sources=2

echo "$0 $@"  # Print the command line for logging

. local/parse_options.sh || exit 1;

if [ $# != 0 ]; then
    echo "Usage: $0"
    echo " e.g.: $0"
    echo "main options (for others, see top of script file)"
    echo "  --nj <n|10>                                     # Number of jobs"
    echo "  --stage <stage|0>                               # To control partial reruns"
    echo "  --num-of-sources                                # Number of sources"
    exit 1;
fi

# Path to dataset
musdb18_dir=/home/divraj/divyesh/dataset/MUSDB18-7-STEMS


# # Name of directories
# train_dir=xvector_vox1_vox2
# test_dir=voxceleb1_test
# mfcc_dir=mfcc
# model_dir=models
# test_vec_dir=test_vec

if [ $stage -eq 0 ]; then
    local/decode.sh ${musdb18_dir} train data/musdb18_train/data
    local/decode.sh ${musdb18_dir} test data/musdb18_test/data
    
    local/make_musdb18.pl data/musdb18_train/data data/musdb18_train
    local/make_musdb18.pl data/musdb18_test/data data/musdb18_test

    local/get_durations.py --in-data-dir data/musdb18_train
    local/get_segments.py --in-data-dir data/musdb18_train --conf-path conf/conf.yaml
fi

if [ $stage -eq 1 ]; then
    local/extract_mfcc.py --in-data-dir data/musdb18_train --conf-path conf/conf.yaml
fi

if [ $stage -eq 2 ]; then
    local/extract_mfcc.py --in-data-dir data/musdb18_train --conf-path conf/conf.yaml
fi

    # mkdir -p data/musdb18
    # if [ ${num_of_sources} -eq 2 ]; then
    #     echo -e "vocal\ninstrument" > data/musdb18/sources
    # elif [ ${num_of_sources} -eq 4 ]; then
    #     echo -e "vocal\ndrum\nbass\nother" > data/musdb18/sources
    # fi