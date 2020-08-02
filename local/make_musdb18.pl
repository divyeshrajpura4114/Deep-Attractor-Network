#!/usr/bin/perl
#
# Copyright 2020 Divyesh Rajpura
#
# Usage: make_musdb18.pl /export/musdb18 data/musdb18

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-radmri> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/musdb18 data/musdb18\n";
  exit(1);
}

(${data_base}, ${out_dir}) = @ARGV;

if (system("mkdir -p ${out_dir}") != 0) {
  die "Error making directory ${out_dir}";
}

open(WAV_TEST, ">", "${out_dir}/wav.scp") or die "Could not open the output file ${out_dir}/wav.scp";
# open(UTT2SPK, ">", "${out_dir}/utt2spk") or die "Could not open the output file ${out_dir}/utt2spk";
# open(SPK2UTT, ">", "${out_dir}/spk2utt") or die "Could not open the output file ${out_dir}/spk2utt";

opendir my ${dh}, "${data_base}" or die "Cannot open directory: $!";
my @music_dirs = grep {-d "${data_base}/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

foreach (@music_dirs) {
    my ${music_id} = $_;
    my ${wav_path} = "${data_base}/${music_id}";
    print WAV_TEST "${music_id}", " ${wav_path}", "\n";
}

close(WAV_TEST) or die;
# close(UTT2SPK) or die;
# close(SPK2UTT) or die;
