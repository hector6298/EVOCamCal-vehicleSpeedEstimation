# Vehicle speed estimation from roadside camera
This repo holds my graduation project. The goal is to harness computer vision and projective geometry to successfully transform a simple street camera into a vehicle speed estimator.

## TO-DO list
- Change the targets from 2D image points to world points (long, lat)
- Output metrics to file, in order to compute metrics of:
  - RMSE means and stds for distance and reprojection error of the population vs. generations
  - Pictures of virtual plane accross generations.
  - speed evolution of a sample of cars to evaluate congruence. (check top speeds and also vehicles parked)
- Implement speed smoothing (Maybe exponential moving average)

## Getting started

Clone the repo:

```
git clone https://github.com/hector6298/titulacion_vehice_speed_estimation.git

```

Install all dependencies for python3:

```
cd titulacion_vehicle_speed_estimation
sudo pip3 install -r requirements.txt

```

Note that you also have to install OpenCV for C++ in order to use the camera calibrator. Please see : LINK
Now compile the calibration code:

```
cd manual_calib
g++ -I/usr/local/include/ -L/usr/local/lib/ -g -o bin ./src/main.cpp ./src/Cfg.cpp ./src/CamCal.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -lm

```
Run:

```
./bin
```

## Configuring object detectors
Thanks to @hunglc007 for the implementation of yolov4. Download yolov4 weights from: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Convert them to tensorflow usable weights:

```
python3 workflowUtils/yolov4/save_model.py --weights workflowUtils/yolov4/data/yolov4.weights --output workflowUtils/yolov4/data/checkpoints/yolov4 --input_size 512 --model yolov4
```

