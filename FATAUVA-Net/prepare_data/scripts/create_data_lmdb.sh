#!/usr/bin/env bash
#LMDB=F:/FATAUVA-Net/prepare_data/CelebA/all_data       # output LMDB 存储位置
#DATA=F:/FATAUVA-Net/prepare_data/CelebA/all_data       # path of data.txt
#DATA_ROOT=E:/CelebA/Img/img_align_celeba/
#
#RESIZE_HEIGHT=218
#RESIZE_WIDTH=178

#LMDB=F:/FATAUVA-Net/prepare_data/AFEW-VA/crop       # output LMDB 存储位置
#DATA=F:/FATAUVA-Net/prepare_data/AFEW-VA/crop/       # path of data.txt
#DATA_ROOT=E:/crop_data/AFEW-VA/
#
#RESIZE_HEIGHT=170
#RESIZE_WIDTH=170

LMDB=F:/bolin_lmdb/03       # output LMDB 存储位置
DATA=F:/txts/03       # path of data.txt
DATA_ROOT=F:/Bolin_Speech/

RESIZE_HEIGHT=227
RESIZE_WIDTH=227

# Checks for DATA_ROOT Path
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
    echo "Set the DATA_ROOT variable to the path where the data instances are stored."
    exit 1
fi

# Creating LMDB
echo "Creating data lmdb..."
GLOG_logtostderr=1 convert_imageset \
  --resize_height=$RESIZE_HEIGHT \
  --resize_width=$RESIZE_WIDTH \
  --shuffle \
  $DATA_ROOT \
  $DATA/train.txt \
  $LMDB/train_lmdb

echo "Done."
