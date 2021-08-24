import os
import numpy as np
import matplotlib.pyplot as plt
import ast
import cv2
import argparse
from joblib import load
from matplotlib import rcParams
rcParams['font.family'] = ['serif']
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 16

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default=11,
                    help="Index of the video")
parser.add_argument("-m", "--metric", default='euclidean',
                    help="metric for the tracking backend")
args = parser.parse_args()


video = f"source_material/streetVideos/seattle{args.video}.mp4"
videoDir = f"results/plots_tables/{args.metric}/video{args.video}"

csvDirSamples = f"results/plots_tables/{args.metric}/samples.txt"
csvDirSpeeds = f"results/plots_tables/{args.metric}/speeds.txt"
csvDirLineSpeeds = f"results/plots_tables/{args.metric}/lineSpeeds.txt"

# Create containing folder for plots
if not os.path.exists(videoDir):
    os.mkdir(videoDir)

# Create file for numeric results
f_samples = open(csvDirSamples, 'a+')
f_speeds = open(csvDirSpeeds, 'a+')
f_lineSpeeds = open(csvDirLineSpeeds, 'a+')

# Load video to get FPS
cap = cv2.VideoCapture(video)

if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))

# Load all result data from video execution
bbDists = load(f"results/bbox_distances/{args.metric}/bbox_distances{args.video}")
bbDists = ast.literal_eval(bbDists)
data = load(f"results/velocities/{args.metric}/velocities{args.video}")
spotVelocities = load(f"results/spot_velocities/{args.metric}/spot_velocities{args.video}")


all_lengths = list(map(lambda arr: len(arr), data))


print("saving sampling data")
registered_objects = len(all_lengths)
mean_samples = np.mean(all_lengths)
median_samples = np.median(all_lengths)
std_samples = np.std(all_lengths)

f_samples.write(f"{registered_objects},{mean_samples},{median_samples},{std_samples}\n")



print("saving sampling plot")
_ = plt.hist(all_lengths, bins=50, color='g')
plt.yscale('log')
_ = plt.xlabel("n_samples")
_ = plt.ylabel("Frequency")
plt.savefig(f"{videoDir}/samplesPerObject{args.video}.jpg")



print("saving speed data")
all_samples_concat = np.hstack(list(data))
len_all_samples = len(all_samples_concat)
mean_speeds = np.mean(all_samples_concat)
median_speeds = np.median(all_samples_concat)
std_speeds = np.std(all_samples_concat)
portion_70 = len(all_samples_concat[all_samples_concat > 70])/len(all_samples_concat)
portion_100 = len(all_samples_concat[all_samples_concat > 100])/len(all_samples_concat)

f_speeds.write(f"{len_all_samples},{mean_speeds},{median_speeds},{std_speeds},{portion_70},{portion_100}\n")


print("saving speed plots")
plt.clf()
_ = plt.hist(all_samples_concat, bins=100, density=False)
_ = plt.axvline(x=np.mean(all_samples_concat), color='red')
plt.yscale("log")
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")
plt.savefig(f"{videoDir}/SpeedDistribution{args.video}.jpg")



sorted_samples = sorted(data, reverse=True, key=lambda arr: len(arr))[:10]
sorted_lens = list(map(lambda arr: len(arr), sorted_samples))
median_lens, std_lens = np.median(sorted_lens), np.std(sorted_lens)

inds = [list(map(lambda t: t*25/fps,range(len(sorted_sample)))) for sorted_sample in sorted_samples]

sorted_bbs = sorted(bbDists, reverse=True, key=lambda arr: len(arr))[:10]
sorted_lens_bb = list(map(lambda arr: len(arr), sorted_bbs))
median_lens_bb, std_lens_bb = np.median(sorted_lens_bb), np.std(sorted_lens_bb)
inds_bbs = [range(len(sorted_bb)) for sorted_bb in sorted_bbs]

print("saving speed time series")
plt.clf()
for i in range(len(inds)):
    _ = plt.plot(inds[i], sorted_samples[i], )
    _ = plt.ylim(0,100)
_ = plt.xlabel("Time (s)")
_ = plt.ylabel("Speed (km/h)")
plt.savefig(f"{videoDir}/SpeedTimeSeries.jpg")

print("saving separated speed time series")
plt.clf()
f, ax = plt.subplots(len(inds), 1, sharex=False, sharey=False)
for i in range(len(inds)):
    ax[i].plot(inds[i], sorted_samples[i])
f.text(0.5, 0.04, 'Time (s)', ha='center')
f.text(0.01, 0.5, 'Speed (km/h)', va='center', rotation='vertical')
plt.subplots_adjust(hspace=0.5)
f.set_size_inches(6.5, 9.5)
plt.savefig(f"{videoDir}/multiPlotTimeSeries{args.video}.jpg")

print("saving bounding box time series")
plt.clf()
for i in range(len(inds)):
    _ = plt.plot(inds_bbs[i], sorted_bbs[i])

_ = plt.xlabel("Frames")
_ = plt.ylabel("Diagonal size")
plt.savefig(f"{videoDir}/bbDistTimeSeries.jpg")

print("saving bounding box separated time series")
plt.clf()
f, ax = plt.subplots(len(inds_bbs), 1, sharex=False, sharey=False)
for i in range(len(inds_bbs)):
    ax[i].plot(inds_bbs[i], sorted_bbs[i])
f.text(0.5, 0.04, 'Time (s)', ha='center')
f.text(0.04, 0.5, 'Speed (km/h)', va='center', rotation='vertical')
plt.subplots_adjust(hspace=0.5)
f.set_size_inches(6.5, 9.5)
plt.savefig(f"{videoDir}/multiPlotBBsTimeSeries.jpg")



print("saving speed boxplots")
plt.clf()
fig, ax = plt.subplots()
_ = ax.boxplot(np.array(sorted_samples, dtype=object))
_ = ax.set_xlabel("object ID")
_ = ax.set_ylabel("Speed (km/h)")
plt.savefig(f"{videoDir}/speedBoxPlots{args.video}.jpg")

print("saving line speeds data")
registered_line_speeds = len(spotVelocities)
mean_line_speed = np.mean(spotVelocities)
median_line_speed = np.median(spotVelocities)
std_line_speed = np.std(spotVelocities)

f_lineSpeeds.write(f"{registered_line_speeds},{mean_line_speed},{median_line_speed},{std_line_speed}\n")


print("saving line speed distribution")
plt.clf()
_ = plt.hist(spotVelocities, bins=10, color="#FFA500")
_ = plt.xlabel("Speed (km/h)")
_ = plt.ylabel("Frequency")
plt.savefig(f"{videoDir}/SpeedLineDistribution{args.video}.jpg")



f_samples.close()
f_speeds.close()
f_lineSpeeds.close()

print(f"All done for video {args.video} with {args.metric}!")