#!/bin/bash
set -ex

#data in 1 hour
wget -O /resource_1h.tar.gz \
     https://tetris-icloud.s3.amazonaws.com/resource_1h.tar.gz

tar -xzvf resource_1h.tar.gz