# ASPDAC Tutorial 8
## Introduction

This repository includes:
1) CircuitOps hands-on example
  1) Demo of CircuitOps LPG query script.
2) OpenROAD Python APIs example
  1) Image-absed Static IR Drop prediction using OpenROAD Python APIs (similar to 2023 ICCAD Contest Problem C)
  2) RL-based sizing using OpenROAD Python APIs

##Get Started
Build the Docker image using the Dockerfile provided:
```
docker build -t <image_name> .
docker run -it --name <container_name> <image_name>
```
CircuitOps example:
```
#clone [CircuitOps](https://github.com/NVlabs/CircuitOps/tree/main) and change designs in <CircuitOps DIR>/src/tcl/set_design.tcl
#build IR Tables
./OpenROAD/build/src/openroad <CircuitOps DIR>/src/tcl/generate_tables.tcl
#Extract buffer tree
python3 <CircuitOps DIR>/src/python/BT_sampling_OpenROAD.py <CircuitOps DIR>/IRs/nangate45/<design> <Buffer tree output DIR> 
python3 <CircuitOps DIR>/src/python/LPG_query.py <Buffer tree output DIR> <CircuitOps DIR>/IRs/nangate45/<design>/
```
OpenROAD Python APIs example:
###Image-absed Static IR Drop prediction using OpenROAD Python APIs
```
#clone this repository and [CircuitOps](https://github.com/NVlabs/CircuitOps/tree/main) and change designs in <ASPDAC2024-Tutorial DIR>/ASPDAC_helpers.py on line 199 then run the example scripts using python-enabled OpenROAD:
./OpenROAD/biuld/src/openroad -python IR_prediction.py --path <CircuitOps DIR>
```
###RL-based sizing using OpenROAD Python APIs
```
#clone this repository and run the example scripts using python-enabled OpenROAD:
./OpenROAD/biuld/src/openroad -python gate_sizing_example.py --path <ASPDAC2024-Tutorial DIR>
```



