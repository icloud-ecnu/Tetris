"""
测试os popen 
"""
import os
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--podname",type=str)
    parse.add_argument("--nodenameId",type=int)
    args = parse.parse_args()
    node_prefix = "k8s-node"
    nodename = node_prefix + str(args.nodenameId)
    podname = args.podname
    
    print(f"nodename = {nodename} podname = {podname}")


