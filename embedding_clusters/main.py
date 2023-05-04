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

    parser.add_argument("-s", "--save", action="store_true", help="Save embeddings to output folder")
    parser.add_argument("-d", "--debug", action="store_true", help="Use debug settings, maximum verbosity")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more prints")
    parser.add_argument("-i", "--input", action="store", default="data", help="Path to audio library/libraries")
    parser.add_argument("-o", "--output", action="store", default="output", help="Path to write outputs to")

    args = parser.parse_args(raw_args)
    print(f"input parameters {vars(args)}")
    return args

def main(raw_args=None):
  t = datetime.datetime.now()
  tf = f"{t.year}{t.month}{t.day}{t.hour}{t.minute}{t.second}"

  # process args
  args = get_args(raw_args)
  if args.debug:
    args.input = "/media/nova/Datasets/DCASE2016/eval/audio"
    args.ouput = "test"
    args.verbose = True
    args.save = False
  loader = DataLoader(args)

  # load data and model
  loader.collect(3)
  model = fusion_cat_xwc.load_model()

  # save embeddings too
  all_embeddings = {"dcase": []}
  newfolder = os.path.join(args.output, tf)
  os.makedirs(newfolder)

  # generate embeddings
  for infile in loader.files:
    filename = os.path.basename(infile)
    sr, d = wav.read(infile)

    if args.verbose:
      print(f"Loaded file {filename} with sample rate {sr} and shape {d.shape}")

    # stereo to mono
    if d.shape[-1] == 2:
      d = d.sum(axis=1) / 2

    # if sr != 16000:
    #   print(f"Resampling to 16000, new length is {round(len(d) * float(16000) / sr)}")
    #   d = sps.resample(d, round(len(d) * float(16000) / sr))
    #   sr = 16000

    # trim to 10s due to memory
    # if d.shape[0] >= 10 * sr:
    #   d = d[:10 * 16000]

    # save embedding
    embedding = fusion_cat_xwc.get_scene_embeddings(torch.as_tensor(d, dtype=torch.float32)[None, :], model)
    # new_embeddings.append({'filename': filename, 'embedding': embedding})
    # all_embeddings[dirname] = new_embeddings
    all_embeddings["dcase"].append({'filename': filename, 'embedding': embedding})
    if args.save:
      torch.save(embedding, f"{newfolder}/{filename}.pt")

  if args.verbose:
    print("Created embeddings:")
    print(all_embeddings)

  # get all embeddings as flat list
  z = np.asarray([np.asarray(item) for sublist in list(all_embeddings.values()) for item in sublist]).squeeze()

  # get labels as spread array
  labels = []
  for k, v in all_embeddings.items():
    for e in range(len(v)):
      labels.append(k)
  print(labels)

  # Create a two dimensional t-SNE projection of the embeddings
  tsne = TSNE(3, verbose=1)
  tsne_proj = tsne.fit_transform(z)

  cmap = cm.get_cmap('tab20')

  datasets = list(all_embeddings.keys())

  fig, ax = plt.subplots(figsize=(15,15))

  i = 0
  dots = []
  for key, embeddings in all_embeddings.items():
    embedding_len = len(embeddings)
    kp = datasets.index(key) / len(datasets)
    ax.plot(tsne_proj[i:i+embedding_len, 0], tsne_proj[i:i+embedding_len, 1], color=cmap(kp), label=key)
    i = i + embedding_len

  # browser = PointBrowser()
  # fig.canvas.mpl_connect('pick_event', browser.on_pick)
  # fig.canvas.mpl_connect('key_press_event', browser.on_press)

  plt.xticks([])
  plt.yticks([])
  plt.legend()
  plt.show()
  
  if args.debug:
    plt.savefig(os.path.join(args.output, "plots", f"{tf}.png"))

  return

if __name__=="__main__":
  main()