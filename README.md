# ASPDAC Tutorial-8: CircuitOps and OpenROAD: Unleashing ML EDA for Research and Education
## Introduction
Over the last decade, there has been a significant rise in Machine Learning (ML)-based Electronic Design Automation (EDA) research. Various ML methodologies, including Boosting trees, SVM, CNN, GNN, RL, and the latest LLM, have been applied to address both front-end design, back-end optimization, and even cross-stage challenges.

However, a notable challenge lies in treating EDA tools as black boxes, particularly in RL applications. Researchers often use EDA tools as black-box environments, relying on TCL script programming and file I/O for interaction and information extraction. This approach increases the complexity of ML-EDA research, as researchers must become proficient in TCL programming. Notably, not all EDA researchers are well-versed in TCL, which even poses a barrier for ML researchers from other fields seeking to contribute.

Moreover, relying on files for information extraction hampers ML application performance on larger designs. As a remedy, there is a pressing need for an ML-friendly EDA tool that seamlessly integrates with ML methodologies, reducing reliance on TCL programming and file-based information extraction.

This tutorial introduces the CircuitOps and OpenROAD Python APIs, representing a significant milestone in ML-EDA research. CircuitOps is an ML-friendly data infrastructure utilizing labeled property graphs (LPGs) backed by relational tables (IR tables) to generate datasets for ML-EDA applications. The Python-compatible LPG minimizes the developmental effort required for ML-EDA research. The IR tables are generated using OpenROAD Python APIs, which offer numerous advantages over TCL APIs. One key advantage is that OpenROAD can now be directly imported into a Python environment, this not only means we can incorporate OpenROAD with other Python libraries, but we can also interact with OpenROD in real-time, providing unprecedented flexibility in ML-EDA application development. Additionally, we can directly get the information via the Python API without file IO, which increases ML-EDA performance.

This tutorial composes of two hands-on sessions, each with two demos:
- Session 1 :
  - Demo1 : Introduce the OpenROAD Python APIs.
  - Demo2 : Two ML-EDA examples powered by the Python APIs.
    - Image-absed Static IR Drop prediction using OpenROAD Python APIs (similar to 2023 ICCAD Contest Problem C)
    - RL-based sizing using OpenROAD Python APIs
- Session 2 :
  - Demo3 : Introduce CircuitOps's LPG grneration and query.
  - Demo4 : One ML-EDA example uses LPG.

## Get Started

(If attending the 2024 ASP-DAC tutorial, please jump to Session 1 directly.)

### Build the Docker image using the Dockerfile provided and run the docker container:

```
docker build -t <image_name> .
docker run -it --name <container_name> <image_name>
```

## Session 1

### Demo 1

OpenROAD Python APIs example:

```
```

### Demo 2 

Image-absed Static IR Drop prediction using OpenROAD Python APIs:

```
./OpenROAD/build/src/openroad -python ./ASPDAC2024-Tutorial/demo2_IR_prediction_example.py --path ./CirccuitOps/
```

RL-based sizing using OpenROAD Python APIs:

```
./OpenROAD/build/src/openroad -python ./ASPDAC2024-Tutorial/demo2_gate_sizing_example.py --path ./ASPDAC2024-Tutorial/
```

## Session 2

### Demo 3

CircuitOps LPG generation and query example:

- IR Table generation 

```
cd CircuitOps
../OpenROAD/build/src/openroad -python ./src/python/generate_tables.py -w
cd ../
```

- LPG generation & query example

```
cd CircuitOps
python3 ./src/python/BT_sampling_OpenROAD.py ./IRs/nangate45/gcd/ ./PK/
cd ../ASPDAC2024-Tutorial
python3 demo3_LPG_query_example.py --path_BT ../CircuitOps/PK/ --path_IR ../CircuitOps/IRs/nangate45/gcd/ --path_LPG_gen_fun ../CircuitOps/src/python/generate_LPG_from_tables.py
```

### Demo 4

CircuitOps application example

```
```

