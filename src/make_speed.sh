#!/bin/bash

rm -rf data_speed_16k
cp -rf data_16k data_speed_16k

# speed perturbation
for scene in home residential_area; 
do
  mkdir data_speed_16k/TUT-sound-events-2016-development/audio/${scene}_sp
  rm -rf data_speed_16k/TUT-sound-events-2016-development/audio/$scene/*_S*.wav
  ls data_speed_16k/TUT-sound-events-2016-development/audio/$scene/*.wav > /tmp/flist
  for wav in `grep $scene /tmp/flist`; do
    fbase=`basename $wav .wav`
    for speed in 0.8 0.9 1.1 1.2;
    do
       sox --norm -t wavpcm $wav -t wavpcm data_speed_16k/TUT-sound-events-2016-development/audio/${scene}_sp/${fbase}_S${speed}.wav speed $speed &
       sleep 0.5
       done
done
done 
for file in `ls data_16k/TUT-sound-events-2016-development/evaluation_setup/*_train.txt`;
do
fbase=`basename $file`
cat $file | python -c "
import os,sys
for line in sys.stdin.readlines():
  line = line.strip()
  if len(line) == 0:
    continue
  tok = line.split('\t')
  folder = os.path.dirname(tok[0])+'_sp/'
  fname = ((tok[0]).split('/')[-1]).split('.')[0]
  print line
  for speed in [0.8, 0.9, 1.1, 1.2]:
    print folder+fname+'_S'+str(speed)+'.wav'+'\t'+tok[1]+'\t'+str(float(tok[2])/speed)+'\t'+str(float(tok[3])/speed)+'\t'+tok[-1]
" > data_speed_16k/TUT-sound-events-2016-development/evaluation_setup/$fbase
done

