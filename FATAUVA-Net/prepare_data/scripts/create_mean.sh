#!/usr/bin/env bash
LMDB=F:/FATAUVA-Net/prepare_data/AFEW-VA/crop      # input lmdb
DATA=F:/FATAUVA-Net/prepare_data/AFEW-VA/crop       # output mean

compute_image_mean $LMDB/test_data_lmdb \
  $DATA/test_data.binaryproto