# ASP-DAC24 Tutorial-8: CircuitOps and OpenROAD: Unleashing ML EDA for Research and Education
This is a GitHub repository that has the scripts being demoed at the tutorial. The scripts highlight example use cases of ML EDA infrastructure. 

## Background
Over the last decade, there has been a significant rise in machine learning (ML)--based electronic design automation (ML-EDA) research. However, a notable challenge lies in the interaction between existing EDA tools and ML frameworks. Researchers often use TCL scripts for interaction with EDA tools and Python scripts for ML frameworks. They rely on file I/O for interaction between the EDA world and ML world or reinvent the wheel and implement the complete EDA tools in Python. Both these approaches have challenges. The first is extremely slow and it's impossible to iteratively send data/information between EDA tools and is a barrier to entry for non-chip designers. The second slows down research and makes it challenging to make apples-to-apples comparisons between various ML EDA algorithms. As a remedy, there is a pressing need for an ML-friendly EDA tool that seamlessly integrates with ML methodologies, reducing reliance on TCL programming and file-based information extraction.

This tutorial introduces the CircuitOps and OpenROAD Python APIs, representing a significant milestone in ML-EDA research. CircuitOps is an ML-friendly data infrastructure utilizing labeled property graphs (LPGs) backed by relational tables (IR tables) to generate datasets for ML-EDA applications. The Python-compatible LPG minimizes the developmental effort required for ML-EDA research. The IR tables are generated using OpenROAD Python APIs, which offer numerous advantages over TCL APIs. One key advantage is that OpenROAD can now be directly imported into a Python environment, this not only means we can incorporate OpenROAD with other Python libraries, but we can also interact with OpenROAD in real-time, providing unprecedented flexibility in ML-EDA application development. Additionally, we can directly get the information via the Python API without file IO, which increases ML-EDA performance.

## Tutorial Contents

This tutorial is composed of two hands-on sessions, each with two demos:
- Session 1 :
  - Demo1: Introduce the OpenROAD Python APIs for an EDA flow o
  - Demo2: Two ML-EDA examples powered by the Python APIs.
    - Image-based static IR Drop prediction using OpenROAD Python APIs (similar to 2023 ICCAD Contest Problem C)
    - RL-based sizing using OpenROAD Python APIs
- Session 2 :
  - Demo3: Introduce CircuitOps's LPG generation and query and interaction with OpenROAD
  - Demo4: Use of CircuitOps to generate data for stage delay prediction

## Getting Started


### Live at ASP-DAC 24

If attending the 2024 ASP-DAC tutorial live and have emailed your public ssh key to us for Google Cloud Computing resources, you would have received an email on how to login, please login using the instructions. We have already cloned the repoository in your user directory and installed the required software dependencies and built the OpenROAD application. You can directly jump to the running the scripts as described in session 1 and sesion 2. 


### Everywhere else (Not attending the ASP-DAC tutorial)

If you are not attending the tutorial, you require clone this repository, install the required software dependencies, and build OpenROAD and CircuitOps.

#### Clone the repository

```
git clone --recursive https://github.com/ASU-VDA-Lab/ASP-DAC24-Tutorial
```

#### Build OpenROAD and CircuitOps

#####  Option 1: Build using Docker 
The following technique assumes you have docker installed on your machine. If you do not have then install docker from [here](https://docs.docker.com/engine/install/). Build the docker image and run using the following commands:
```
docker build -t <image_name> .
docker run -it --name <container_name> <image_name>
```

##### Option 2: Build locally
The following technique assumes you have a machine with the required Ubuntu OS prerequisite of OpenROAD an CircuitOps.

Install dependendencies for OpenROAD:
```
sudo ./OpenROAD/etc/DependencyInstaller.sh
```

Install dependencies for CircuitOps and ML EDA applications:
```
sudo apt-get install -y python3-matplotlib
sudo apt-get install -y nvidia-cuda-toolkit
sudo apt-get update
sudo apt-get install -y python3-graph-tool
sudo apt-get update && apt-get install -y gnupg2 ca-certificates
sudo apt-get install -y python3-pip
pip3 install torch
pip3 install dgl
pip3 install pycairo
pip3 install pandas
pip3 install scikit-learn
```

Once pacakges have been installed, build OpenROAD:

```
cd ./OpenROAD/
mkdir build
cd build
cmake ..
make -j
```

## Running the Demo Scripts

### Session 1

The following script asssume that we work from the ASPDAC2024-Tutorial directory.
Prior to testing the script please select the Tutorial directory.

```
cd <Path to ASP-DAC-24-Tutorial Directoy>
```

### Demo 1
OpenROAD Python APIs EDA flow example:

```
./OpenROAD/build/src/openroad -python session1/demo1_flow.py
```

OpenROAD Python APIs circuit properties query example:

```
./OpenROAD/build/src/openroad -python session1/demo1_query.py
```

### Demo 2 

Image-based static IR Drop prediction using OpenROAD Python APIs:

```
./OpenROAD/build/src/openroad -python session1/demo2_IR.py 
```

RL-based sizing using OpenROAD Python APIs:

```
./OpenROAD/build/src/openroad -python session1/demo2_gate_sizing.py 
```

## Session 2

### Demo 3

CircuitOps LPG generation and query example:

- IR Tables generation 

```
cd CircuitOps
../OpenROAD/build/src/openroad -python ./src/python/generate_tables.py
cd ../
```

- LPG generation & query example:
  - create LPG via IR Tables
  ```
  python3 session2/demo3_LPG_query_example.py 
  ```
  
  - create LPG via OpenROAD Python API
  ```
  ./OpenROAD/build/src/openroad -python session2/demo3_LPG_query_example.py --use_pd
  ``` 

### Demo 4

CircuitOps application example

```
./OpenROAD/build/src/openroad -python session2/demo4_preroute_net_delay_prediction.py
```
