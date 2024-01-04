#!/bin/bash
set -ex

#data in 10 seconds
wget -O /resource_1h.tar.gz \
     https://tetris-icloud.s3.amazonaws.com/resource_10s.tar.gz

tar -xzvf resource_10s.tar.gz
