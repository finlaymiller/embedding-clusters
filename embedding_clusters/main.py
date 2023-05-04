# imports
import os
import torch
import argparse
import datetime
import numpy as np
import hearbaseline
import matplotlib as mpl
import matplotlib.cm as cm
from GURA import fusion_cat_xwc
from scipy import signal as sps
from sklearn.manifold import TSNE
from data_loader import DataLoader
from scipy.io import wavfile as wav
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Clustering by embedding using GURA")

    parser.add_argument("-i", "--input", action="store", default="data", help="Path to audio library/libraries")
    parser.add_argument("-o", "--output", action="store", default="embeddings", help="Path to write embeddings to")

    args = parser.parse_args(raw_args)
    print(f"input parameters {vars(args)}")
    return args

def main(raw_args=None):
  # process args
  args = get_args(raw_args)
  loader = DataLoader(args)

  loader.load()

  # load model
  # model = fusion_cat_xwc.load_model()

  # os.chdir(args.input)
  # all_embeddings = {}

  # # save embeddings too
  # t = datetime.datetime.now()
  # newfolder = args.output + f"embeddings{t.year}{t.month}{t.day}{t.hour}{t.minute}{t.second}"


  # for dirname in os.listdir():
  #   os.chdir(dirname)
  #   os.mkdir(newfolder)

  #   print(f"\nProcessing {dirname} folder")

  #   new_embeddings = []
  #   audio = []
  #   for filename in os.listdir():
  #     sr, d = wav.read(filename)

  #     print(f"Loaded file {filename} with sample rate {sr}", d.shape)

  #     # stereo to mono
  #     if d.shape[-1] == 2:
  #       d = d.sum(axis=1) / 2

  #     # if sr != 16000:
  #     #   print(f"Resampling to 16000, new length is {round(len(d) * float(16000) / sr)}")
  #     #   d = sps.resample(d, round(len(d) * float(16000) / sr))
  #     #   sr = 16000

  #     # trim to 30s due to memory
  #     # if d.shape[0] >= 30 * sr:
  #     #   d = d[:30 * 16000]

  #     embedding = fusion_cat_xwc.get_scene_embeddings(torch.as_tensor(d, dtype=torch.float32)[None, :], model)

  #     new_embeddings.append({'filename': filename, 'embedding': embedding})
  #     torch.save(embedding, f"{newfolder}/{filename}.pt")

  #   all_embeddings[dirname] = new_embeddings

  #   os.chdir("..")

  return

if __name__=="__main__":
  main()