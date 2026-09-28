#!/usr/bin/env bash
# KITTI raw 2011_09_26_drive_0001 (Velodyne HDL-64E, 108 스캔). 라이선스 CC BY-NC-SA 3.0 — 저장소에 커밋하지 않는다.
set -e
mkdir -p ~/lidar-data && cd ~/lidar-data
[ -f drive_0001.zip ] || wget -q -O drive_0001.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
unzip -q -o drive_0001.zip '*/velodyne_points/*'
ls 2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data | wc -l
