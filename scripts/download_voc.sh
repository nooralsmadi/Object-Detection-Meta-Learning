#!/bin/bash

DEST="datasets/VOCdevkit"

mkdir -p $DEST
cd $DEST || exit

echo ">>> Downloading VOC 2007 trainval..."
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
echo ">>> Downloading VOC 2007 test..."
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
echo ">>> Downloading VOC 2012 trainval..."
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

echo ">>> Extracting..."
for f in *.tar; do
  tar -xf $f
done

echo ">>> Done downloading and extracting VOC datasets."

cd ../../

echo ">>> Generating 1-shot splits (seed 1)..."
python datasets/prepare_voc_few_shot.py --seeds 1

echo ">>> Copying 1-shot files to vocsplit/..."
cp datasets/vocsplit/seed1/*1shot*.txt datasets/vocsplit/

echo ">>> Dataset setup complete!"
