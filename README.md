# CNN based DCASE 2016 sound event detection system 

Sound event detection system submitted to [DCASE 2016](http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-real-life-audio) (detection and classification of acoustic scenes and events) challenge. 

Convolutional neural network is used for detecting and classifying polyphonic events in a long temporal context of filter bank acoustic features. Training data are augmented vi sox speed perturbation.

On development data set the system achieves 0.84% segment error rate (7.7% relative imporment compared to baseline) 36.3% F-measure (55.1 relative better than baseline system). 

Technical details are descibed in the [challenge report](http://www.cs.tut.fi/sgn/arg/dcase2016/documents/challenge_technical_reports/Task3/Gorin_2016_task3.pdf). Detailed results summary on development and evaluation audios are also [available](http://www.cs.tut.fi/sgn/arg/dcase2016/task-results-sound-event-detection-in-real-life-audio):

## Basic usage

*run-cnn-pipeline.sh* - complete self-documented script for reproducing all the experiments including the following:

  * *task3_gmm_baseline.py* - baseline GMM system [provided](https://github.com/TUT-ARG/DCASE2016-baseline-system-python) by organizers.

  * *src/make_downsample.sh* - basic data preparation (down sampling)

  * *task3_cnn.py* - run CNN based system training and testing

  * *src/make_speed.sh* - speed perturbation
