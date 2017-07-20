#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9
do
   python pt_model_tiff.py runs/debug --device-ids 0,1 --batch-size 80 --fold $i --model densenet161
done