# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import numpy as np
from graph_tool.all import *
from numpy.random import *
import time
import graph_tool as gt
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

#### helper functions ####
def read_tables_OpenROAD(data_root, design=None):
    cell_cell_path = data_root + "cell_cell_edge.csv"
    cell_pin_path = data_root  + "cell_pin_edge.csv"
    cell_path = data_root  + "cell_properties.csv"
    net_pin_path = data_root  +  "net_pin_edge.csv"
    net_path = data_root  + "net_properties.csv"
    pin_pin_path = data_root  + "pin_pin_edge.csv"
    pin_path = data_root  + "pin_properties.csv"
    net_cell_path = data_root  + "cell_net_edge.csv"
    all_fo4_delay_path = data_root + "libcell_properties.csv"

    ### load tables
    fo4_df = pd.read_csv(all_fo4_delay_path)
    pin_df = pd.read_csv(pin_path)
    cell_df = pd.read_csv(cell_path)
    net_df = pd.read_csv(net_path)
    cell_cell_df = pd.read_csv(cell_cell_path)
    pin_pin_df = pd.read_csv(pin_pin_path)
    cell_pin_df = pd.read_csv(cell_pin_path)
    net_pin_df = pd.read_csv(net_pin_path)
    net_cell_df = pd.read_csv(net_cell_path)
    return pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, fo4_df

### rename cells with cell0, cell1, ... and update the cell names in pin_df
def rename_cells(cell_df, pin_df):
    ### rename cells ###
    cell_name = cell_df[["name"]]
    cell_name.loc[:, ["new_cellname"]] = ["cell" + str(i) for i in range(cell_name.shape[0])]
    pin_df = pin_df.merge(cell_name.rename(columns={"name":"cellname"}), on="cellname", how="left")
    idx = pin_df[pd.isna(pin_df.new_cellname)].index

    port_names = ["port" + str(i) for i in range(len(idx))]
    pin_df.loc[idx, "new_cellname"] = port_names
    cell_df["new_cellname"] = cell_name.new_cellname.values
    return cell_df, pin_df

### rename nets with net0, net1, ... and update the net names in pin_df
def rename_nets(net_df, pin_df):
    ### rename nets ###
    net_name = net_df[["name"]]
    net_name.loc[:, ["new_netname"]] = ["net" + str(i) for i in range(net_name.shape[0])]
    pin_df = pin_df.merge(net_name.rename(columns={"name":"netname"}), on="netname", how="left")
    return net_df, pin_df

### 1) get edge src and tar ids and 2) generate edge_df by merging all edges
def generate_edge_df_OpenROAD(pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df):
    edge_id = pd.concat([pin_df.loc[:,["id", "name"]], cell_df.loc[:,["id", "name"]], net_df.loc[:,["id", "name"]]], ignore_index=True)
    src = edge_id.copy()
    src = src.rename(columns={"id":"src_id", "name":"src"})
    tar = edge_id.copy()
    tar = tar.rename(columns={"id":"tar_id", "name":"tar"})

    pin_pin_df = pin_pin_df.merge(src, on="src", how="left")
    pin_pin_df = pin_pin_df.merge(tar, on="tar", how="left")

    cell_pin_df = cell_pin_df.merge(src, on="src", how="left")
    cell_pin_df = cell_pin_df.merge(tar, on="tar", how="left")

    net_pin_df = net_pin_df.merge(src, on="src", how="left")
    net_pin_df = net_pin_df.merge(tar, on="tar", how="left")

    net_cell_df = net_cell_df.merge(src, on="src", how="left")
    net_cell_df = net_cell_df.merge(tar, on="tar", how="left")

    cell_cell_df = cell_cell_df.merge(src, on="src", how="left")
    cell_cell_df = cell_cell_df.merge(tar, on="tar", how="left")

    # drop illegal edges
    idx = pin_pin_df[pd.isna(pin_pin_df.src_id)].index
    pin_pin_df = pin_pin_df.drop(idx)
    idx = pin_pin_df[pd.isna(pin_pin_df.tar_id)].index
    pin_pin_df = pin_pin_df.drop(idx)

    idx = cell_pin_df[pd.isna(cell_pin_df.src_id)].index
    cell_pin_df = cell_pin_df.drop(idx)
    idx = cell_pin_df[pd.isna(cell_pin_df.tar_id)].index
    cell_pin_df = cell_pin_df.drop(idx)

    idx = net_pin_df[pd.isna(net_pin_df.src_id)].index
    net_pin_df = net_pin_df.drop(idx)
    idx = net_pin_df[pd.isna(net_pin_df.tar_id)].index
    net_pin_df = net_pin_df.drop(idx)

    idx = net_cell_df[pd.isna(net_cell_df.src_id)].index
    net_cell_df = net_cell_df.drop(idx)
    idx = net_cell_df[pd.isna(net_cell_df.tar_id)].index
    net_cell_df = net_cell_df.drop(idx)

    idx = cell_cell_df[pd.isna(cell_cell_df.src_id)].index
    cell_cell_df = cell_cell_df.drop(idx)
    idx = cell_cell_df[pd.isna(cell_cell_df.tar_id)].index
    cell_cell_df = cell_cell_df.drop(idx)

    edge_df = pd.concat([pin_pin_df.loc[:,["src_id", "tar_id"]], cell_pin_df.loc[:,["src_id", "tar_id"]], \
                      net_pin_df.loc[:,["src_id", "tar_id"]], net_cell_df.loc[:,["src_id", "tar_id"]], \
                      cell_cell_df.loc[:,["src_id", "tar_id"]]], ignore_index=True)

    return pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, edge_df

def get_large_components(hist, th=2000):
    labels = []
    for i in range(len(hist)):
        if hist[i] > th:
            labels.append(i)
    return labels

### generate subgraph
def get_subgraph(g_old, v_mask, e_mask):
    u = GraphView(g_old, vfilt=v_mask, efilt=e_mask)
    print("connected component graph: num of edge; num of nodes", u.num_vertices(), u.num_edges())
    ### check whether subgraph is connected and is DAG
    _, hist2 = label_components(u, directed=False)
    return u

### generate cell graph from cell ids
def get_cell_graph_from_cells(u_cells, g, e_type, e_id):
    u_cells = np.unique(u_cells).astype(int)

    # add cell2cell edge
    v_mask_cell = g.new_vp("bool")
    e_mask_cell = g.new_ep("bool")
    v_mask_cell.a[u_cells] = True

    e_ar = g.get_edges(eprops=[e_type, e_id])
    mask = e_ar[:,2]==4 # edge type == 4: cell2cell
    e_ar = e_ar[mask]
    e_src = e_ar[:,0]
    e_tar = e_ar[:,1]
    e_mask = (v_mask_cell.a[e_src] == True) & (v_mask_cell.a[e_tar] == True)
    e_mask_cell.a[e_ar[:,-1][e_mask]] = True
    print("num of edges to add", e_mask.sum())
    print("num of edges", e_mask_cell.a.sum())

    ### construct and check u_cell_g
    u_cell_g = get_subgraph(g, v_mask_cell, e_mask_cell)
    return u_cell_g


