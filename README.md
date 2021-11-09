# Vehicle speed estimation from roadside camera
This repo holds my graduation project. The goal is to harness computer vision and projective geometry to successfully transform a simple street camera into a vehicle speed estimator.

![vehicleestimation](https://user-images.githubusercontent.com/41920808/136614193-12f8896f-c8c3-46b0-8251-a90ce3963fcf.gif)


## Set up

Clone the repo:

```
git clone https://github.com/hector6298/titulacion_vehice_speed_estimation.git

```

### Python Dependencies

Install all dependencies for python3:

```
cd titulacion_vehicle_speed_estimation
sudo pip3 install -r requirements.txt

```

Note that you also have to install OpenCV for C++ in order to use the camera calibrator. I used [this](https://vitux.com/opencv_ubuntu/) article to Install openCV 4.5 on my local machine running Ubuntu 20.04
Now compile the calibration code:

```
cd manual_calib
g++ -I/usr/local/include/ -L/usr/local/lib/ -g -o bin ./src/main.cpp ./src/Cfg.cpp ./src/CamCal.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -lm

```
Run:

```
./bin
```

### Configuring object detectors

Thanks to @hunglc007 for the implementation of yolov4. Download yolov4 weights from [here](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT). Then, convert the weights to tensorflow usable weights:

```
cd ..

python3 workflowUtils/yolov4/save_model.py --weights workflowUtils/yolov4/data/yolov4.weights --output workflowUtils/yolov4/data/checkpoints/yolov4 --input_size 512 --model yolov4
```

### Download all my dataset

This [dataset](https://drive.google.com/drive/folders/1sqsVGd72b57PlKgSkkSwdlUeAB5Y4rcF?usp=sharing) contains footage from 10 videos taken from the Department of Transportation of Seattle. These videos are in .mp4 format, and each video contains the following annotations:

- Date when the videos were recorded.
- 2D image coordinates from different points in the scene
- longitude-latitude coordinates from a view from above, each representing the same spot as the pixel coordinates, forming point correspondences.
- The address of the scene.

All these annotations are contained on a single json file called points.json.

We are all set!


## Running the Calibrations

There is a simple bash script already implemented to set up the configuration file and running the calibration for all the videos. Type:

```
sh bash_scripts/executeCalibrations.sh

```
After all the calibrations were computed. The results should be stored in manual_calib/data. There is one folder for each calibration method: base, RANSAC, and EDA. The directory should look like:

```
tesis/
    manual_calib/
        data/
            base/
                calibration1.jpg
                calibration1.txt
                      ...
                calibration14.jpg
                caliration14.txt
            RANSAC/
                calibration1.jpg
                calibration1.txt
                      ...
                calibration14.jpg
                caliration14.txt
            EDA/
                calibration1.jpg
                calibration1.txt
                      ...
                calibration14.jpg
                caliration14.txt
```

The .jpg files contain the virtual model of the street in each scene. Furthermore, the .txt files follow this pattern:

```
<H_1> <H_2> <H_3> <H_4> <H_5> <H_6> <H_7> <H_8> 1
<Projection_error>
<Backprojection_error>
<Distance_error>
```
If all of these files look ok, then we are ready to move on to the next components.

## Pre-computing and saving the vehicle detections

This step was carried out, due to hardware limitations to speed up the experimentation. It will pre-compute all bounding boxes for every video and store them on pickle files. Type:

```
sh bash_scripts/obtainDets.sh
```

On my computer, it took almost 120 hours, so it will take quite some time. After everything is done, pickles should be stored in results/detections, like so:

```
tesis/
    results/
        detections/
            video1
            video2
              ...
            video14
```
## Executing the tracker and computing all raw measures

This part will use the previous detections, and will compute all speed and tracking measures using the object tracker. Run:

```
sh bash_scripts/executeTracking.sh
```
It should generate two folders inside results/ folder: velocities and spot_velocities. The results directory should now look like:

```
tesis/
    results/
        detections/
        velocities/
            iou/
              velocities1
                ...
              velocities14
        spot_velocities/
            iou/
              spot_velocities1
                ...
              spot_velocities14
```

## Computing all the final Plots

After all the raw material is computed, it is time to get the plots to diagnose the system. There is also a script for this. Type:

```
sh bash_scripts/plotExtractor.sh
```

We should know have one additional folder inside results called plots_tables. As you can guess, it contains plots and tables about the distributions of speed and tracks for every video.
