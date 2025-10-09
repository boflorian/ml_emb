Repository for Project in *Machine Learning for Embedded Systems* Course at **Taltech** 


To run IMU Data Collection on Raspberry Pi Pico compile imu_data_logger/main.cpp

```
mkdir build && cd build && cmake .. && make 
```

Settings for Sampling Rate and Duration are at the top. 

Currently outputs: 
- imu_log.csv
- imu_spectra.csv
- imu_statistics.txt
