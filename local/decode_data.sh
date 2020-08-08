#!/usr/bin/env bash

echo "$0 $@"  # Print the command line for logging

. local/parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "Usage: $0 <dataset-path> <data-type> <out-dir>"
    echo " e.g.: $0 /export/corpota/musdb18 train data/musdb18/train"
    echo "main options (for others, see top of script file)"
    exit 1;
fi

dataset_path=$1
data_type=$2
out_dir=$3

for stem in ${dataset_path}/${data_type}/*.stem.mp4;
do 
    name=`echo $stem | cut -d'.' -f1`
    name=`basename "${name}"`
    name=`echo ${name} | sed 's/-/ /g' | tr -s ' ' | sed 's/ /-/g'`
    mkdir -p "${out_dir}/${name}"
   
	ffmpeg -i "${stem}" -map 0:0 -ac 1 -ar 16000 "${out_dir}/${name}/mixture.wav"
    ffmpeg -i "${stem}" -map 0:1 -ac 1 -ar 16000 "${out_dir}/${name}/drums.wav"
    ffmpeg -i "${stem}" -map 0:2 -ac 1 -ar 16000 "${out_dir}/${name}/bass.wav"
    ffmpeg -i "${stem}" -map 0:3 -ac 1 -ar 16000 "${out_dir}/${name}/other.wav"
    ffmpeg -i "${stem}" -map 0:4 -ac 1 -ar 16000 "${out_dir}/${name}/vocals.wav"

    ffmpeg -i "${out_dir}/${name}/drums.wav" -i "${out_dir}/${name}/bass.wav" -i "${out_dir}/${name}/other.wav" -filter_complex amix=inputs=3 -ac 1 -ar 16000 "${out_dir}/${name}/instrument_mixture.wav"
done
