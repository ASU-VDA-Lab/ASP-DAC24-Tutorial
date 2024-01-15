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
from demo4_preroute_net_delay_prediction_helpers import *

##################
# read IR tables #
##################
pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, fo4_df = \
  read_tables_OpenROAD("../CircuitOps/IRs/nangate45/gcd/")

######################
# rename dfs columns #
######################
pin_df = pin_df.rename(columns={ \
  "pin_name":"name", "cell_name":"cellname", "net_name":"netname", \
  "pin_tran":"tran", "pin_slack":"slack", "pin_rise_arr":"risearr", \
  "pin_fall_arr":"fallarr", "input_pin_cap":"cap", "is_startpoint":"is_start", \
  "is_endpoint":"is_end"})
cell_df = cell_df.rename(columns={ \
  "cell_name":"name", "libcell_name":"ref", "cell_static_power":"staticpower", \
  "cell_dynamic_power":"dynamicpower"})
net_df = net_df.rename(columns={"net_name":"name"})
fo4_df = fo4_df.rename(columns={"libcell_name":"ref"})

##########################################################
# add is_macro, is_seq to pin_df, change pin_dir to bool #
##########################################################
cell_type_df = cell_df.loc[:,["name", "is_macro", "is_seq"]]
cell_type_df = cell_type_df.rename(columns={"name":"cellname"})
pin_df = pin_df.merge(cell_type_df, on="cellname", how="left")
pin_df["is_macro"] = pin_df["is_macro"].fillna(False)
pin_df["is_seq"] = pin_df["is_seq"].fillna(False)
pin_df["dir"] = (pin_df["dir"] == 0)
fo4_df["libcell_id"] = range(fo4_df.shape[0])

### get cell center loc
cell_df["x"] = 0.5*(cell_df.x0 + cell_df.x1)
cell_df["y"] = 0.5*(cell_df.y0 + cell_df.y1)

### add is_buf is_inv to pin_df
cell_type_df = cell_df.loc[:,["name", "is_buf", "is_inv"]]
cell_type_df = cell_type_df.rename(columns={"name":"cellname"})
pin_df = pin_df.merge(cell_type_df, on="cellname", how="left")
pin_df["is_buf"] = pin_df["is_buf"].fillna(False)
pin_df["is_inv"] = pin_df["is_inv"].fillna(False)

### rename cells and nets
cell_df, pin_df = rename_cells(cell_df, pin_df)
net_df, pin_df = rename_nets(net_df, pin_df)

### get dimensions
N_pin, _ = pin_df.shape
N_cell, _ = cell_df.shape
N_net, _ = net_df.shape
total_v_cnt = N_pin+N_cell+N_net
pin_df['id'] = range(N_pin)
cell_df['id'] = range(N_pin, N_pin+N_cell)
net_df['id'] = range(N_pin+N_cell, total_v_cnt)

### generate edge_df
pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, edge_df = \
    generate_edge_df_OpenROAD(pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df)

### get edge dimensions
N_pin_pin, _ = pin_pin_df.shape
N_cell_pin, _ = cell_pin_df.shape
N_net_pin, _ = net_pin_df.shape
N_net_cell, _ = net_cell_df.shape
N_cell_cell, _ = cell_cell_df.shape
total_e_cnt = N_pin_pin + N_cell_pin + N_net_pin + N_net_cell + N_cell_cell

edge_df["e_type"] = 0 # pin_pin
# edge_df.loc[0:N_pin_edge,["is_net"]] = pin_edge_df.loc[:, "is_net"]
edge_df.loc[N_pin_pin : N_pin_pin+N_cell_pin, ["e_type"]] = 1 # cell_pin
edge_df.loc[N_pin_pin+N_cell_pin : N_pin_pin+N_cell_pin+N_net_pin, ["e_type"]] = 2 # net_pin
edge_df.loc[N_pin_pin+N_cell_pin+N_net_pin : N_pin_pin+N_cell_pin+N_net_pin+N_net_cell, ["e_type"]] = 3 # net_cell
edge_df.loc[N_pin_pin+N_cell_pin+N_net_pin+N_net_cell : N_pin_pin+N_cell_pin+N_net_pin+N_net_cell+N_cell_cell, ["e_type"]] = 4 # cell_cell

############
#create LPG#
############
### generate graph
g = Graph()
g.add_vertex(total_v_cnt)
v_type = g.new_vp("int")
v_type.a[0:N_pin] = 0 # pin
v_type.a[N_pin:N_pin+N_cell] = 1 # cell
v_type.a[N_pin+N_cell:total_v_cnt] = 2 # net

### add edge to graph
e_type = g.new_ep("int")

print("num of nodes, num of edges: ", g.num_vertices(), g.num_edges())
g.add_edge_list(edge_df.values.tolist(), eprops=[e_type])
print("num of nodes, num of edges: ", g.num_vertices(), g.num_edges())

### processing fo4 table
fo4_df["group_id"] = pd.factorize(fo4_df.func_id)[0] + 1
fo4_df["libcell_id"] = range(fo4_df.shape[0])
libcell_np = fo4_df.to_numpy()

### assign cell size class
fo4_df["size_class"] = 0
fo4_df["size_class2"] = 0
fo4_df["size_cnt"] = 0
class_cnt = 50
for i in range(fo4_df.group_id.min(), fo4_df.group_id.max()+1):
    temp = fo4_df.loc[fo4_df.group_id==i, ["group_id", "fix_load_delay"]]
    temp = temp.sort_values(by=['fix_load_delay'], ascending=False)
    fo4_df.loc[temp.index, ["size_class"]] = range(len(temp))
    fo4_df.loc[temp.index, ["size_cnt"]] = len(temp)

    temp["size_cnt"] = 0
    MIN = temp.fix_load_delay.min()
    MAX = temp.fix_load_delay.max()
    interval = (MAX-MIN)/class_cnt
    for j in range(1, class_cnt):
        delay_h = MAX - j*interval
        delay_l = MAX - (j+1)*interval
        if j == (class_cnt-1):
            delay_l = MIN
        temp.loc[(temp.fix_load_delay < delay_h) & (temp.fix_load_delay >= delay_l), ["size_cnt"]] = j
    fo4_df.loc[temp.index, ["size_class2"]] = temp["size_cnt"]

cell_fo4 = fo4_df.loc[:,["ref", "fo4_delay", "fix_load_delay",  "group_id", "libcell_id", "size_class", "size_class2", "size_cnt"]]
cell_df = cell_df.merge(cell_fo4, on="ref", how="left")
cell_df["libcell_id"] = cell_df["libcell_id"].fillna(-1)

### add node and edge ids
v_id = g.new_ep("int")
v_id.a = range(v_id.a.shape[0])

e_id = g.new_ep("int")
e_id.a = range(e_id.a.shape[0])

### add pin properties to LPG ###
v_x = g.new_vp("float")
v_y = g.new_vp("float")
v_is_in_clk = g.new_vp("bool")
v_is_port = g.new_vp("bool")
v_is_start = g.new_vp("bool")
v_is_end = g.new_vp("bool")
v_dir = g.new_vp("bool")
v_maxcap = g.new_vp("float")
v_maxtran = g.new_vp("float")
v_num_reachable_endpoint = g.new_vp("int")
v_tran = g.new_vp("float")
v_slack = g.new_vp("float")
v_risearr = g.new_vp("float")
v_fallarr = g.new_vp("float")
v_cap = g.new_vp("float")
v_is_macro = g.new_vp("bool")
v_is_seq = g.new_vp("bool")
v_is_buf = g.new_vp("bool")
v_is_inv = g.new_vp("bool")


v_x.a[0:N_pin] = pin_df["x"].to_numpy()
v_y.a[0:N_pin] = pin_df["y"].to_numpy()
v_is_in_clk.a[0:N_pin] = pin_df["is_in_clk"].to_numpy()
v_is_port.a[0:N_pin] = pin_df["is_port"].to_numpy()
v_is_start.a[0:N_pin] = pin_df["is_start"].to_numpy()
v_is_end.a[0:N_pin] = pin_df["is_end"].to_numpy()
v_dir.a[0:N_pin] = pin_df["dir"].to_numpy()
v_maxcap.a[0:N_pin] = pin_df["maxcap"].to_numpy()
v_maxtran.a[0:N_pin] = pin_df["maxtran"].to_numpy()
v_num_reachable_endpoint.a[0:N_pin] = pin_df["num_reachable_endpoint"].to_numpy()
v_tran.a[0:N_pin] = pin_df["tran"].to_numpy()
v_slack.a[0:N_pin] = pin_df["slack"].to_numpy()
v_risearr.a[0:N_pin] = pin_df["risearr"].to_numpy()
v_fallarr.a[0:N_pin] = pin_df["fallarr"].to_numpy()
v_cap.a[0:N_pin] = pin_df["cap"].to_numpy()
v_is_macro.a[0:N_pin] = pin_df["is_macro"].to_numpy()
v_is_seq.a[0:N_pin] = pin_df["is_seq"].to_numpy()
v_is_buf.a[0:N_pin] = pin_df["is_buf"].to_numpy()
v_is_inv.a[0:N_pin] = pin_df["is_inv"].to_numpy()

### add cell properties to LPG ###
v_x0 = g.new_vp("float")
v_y0 = g.new_vp("float")
v_x1 = g.new_vp("float")
v_y1 = g.new_vp("float")
v_staticpower = g.new_vp("float")
v_dynamicpower = g.new_vp("float")

v_fo4_delay = g.new_vp("float")
v_fix_load_delay = g.new_vp("float")
v_group_id = g.new_ep("int")
v_libcell_id = g.new_ep("int")
v_size_class = g.new_ep("int")
v_size_class2 = g.new_ep("int")
v_size_cnt = g.new_ep("int")

v_is_seq.a[N_pin:N_pin+N_cell] = cell_df["is_seq"].to_numpy()
v_is_macro.a[N_pin:N_pin+N_cell] = cell_df["is_macro"].to_numpy()
v_is_in_clk.a[N_pin:N_pin+N_cell] = cell_df["is_in_clk"].to_numpy()
v_x0.a[N_pin:N_pin+N_cell] = cell_df["x0"].to_numpy()
v_y0.a[N_pin:N_pin+N_cell] = cell_df["y0"].to_numpy()
v_x1.a[N_pin:N_pin+N_cell] = cell_df["x1"].to_numpy()
v_y1.a[N_pin:N_pin+N_cell] = cell_df["y1"].to_numpy()
v_is_buf.a[N_pin:N_pin+N_cell] = cell_df["is_buf"].to_numpy()
v_is_inv.a[N_pin:N_pin+N_cell] = cell_df["is_inv"].to_numpy()
v_staticpower.a[N_pin:N_pin+N_cell] = cell_df["staticpower"].to_numpy()
v_dynamicpower.a[N_pin:N_pin+N_cell] = cell_df["dynamicpower"].to_numpy()
v_x.a[N_pin:N_pin+N_cell] = cell_df["x"].to_numpy()
v_y.a[N_pin:N_pin+N_cell] = cell_df["y"].to_numpy()

v_fo4_delay.a[N_pin:N_pin+N_cell] = cell_df["fo4_delay"].to_numpy()
v_fix_load_delay.a[N_pin:N_pin+N_cell] = cell_df["fix_load_delay"].to_numpy()
v_group_id.a[N_pin:N_pin+N_cell] = cell_df["group_id"].to_numpy()
v_libcell_id.a[N_pin:N_pin+N_cell] = cell_df["libcell_id"].to_numpy()
v_size_class.a[N_pin:N_pin+N_cell] = cell_df["size_class"].to_numpy()
v_size_class2.a[N_pin:N_pin+N_cell] = cell_df["size_class2"].to_numpy()
v_size_cnt.a[N_pin:N_pin+N_cell] = cell_df["size_cnt"].to_numpy()

### add net properties to LPG ###
v_net_route_length = g.new_vp("float")
v_net_steiner_length = g.new_vp("float")
v_fanout = g.new_vp("int")
v_total_cap = g.new_vp("float")
v_net_cap = g.new_vp("float")
v_net_coupling = g.new_vp("float")
v_net_res = g.new_vp("float")

v_net_route_length.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["net_route_length"].to_numpy()
v_net_steiner_length.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["net_steiner_length"].to_numpy()
v_fanout.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["fanout"].to_numpy()
v_total_cap.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["total_cap"].to_numpy()
v_net_cap.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["net_cap"].to_numpy()
v_net_coupling.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["net_coupling"].to_numpy()
v_net_res.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["net_res"].to_numpy()

### add cell id to pin_df
cell_temp = cell_df.loc[:, ["name", "id"]]
cell_temp = cell_temp.rename(columns={"name":"cellname", "id":"cell_id"})
pin_df = pin_df.merge(cell_temp, on="cellname", how="left")
idx = pin_df[pd.isna(pin_df.cell_id)].index
pin_df.loc[idx, ["cell_id"]] = pin_df.loc[idx, ["id"]].to_numpy()

pin_cellid = pin_df.cell_id.to_numpy()
# pin_isseq = v_is_seq.a[0:N_pin]
pin_ismacro = v_is_macro.a[0:N_pin]
# mask = (pin_isseq==True)| (pin_ismacro==True)
mask = pin_ismacro==True
pin_cellid[mask] = pin_df[mask].id ### for pins in macro and seq, pin_cellid = pin id

### add net id to pin_df
net_temp = net_df.loc[:, ["name", "id"]]
net_temp = net_temp.rename(columns={"name":"netname", "id":"net_id"})
pin_df = pin_df.merge(net_temp, on="netname", how="left")

### generate pin-pin graph ###
g_pin = GraphView(g, vfilt=(v_type.a==0), efilt=e_type.a==0)
print("pin graph: num of nodes, num of edges: ", g_pin.num_vertices(), g_pin.num_edges())

### threshold to remove small components in the netlist
cell_cnt_th = 200

### get the large components
comp, hist = label_components(g_pin, directed=False)
comp.a[N_pin:] = -1
labels = get_large_components(hist, th=cell_cnt_th)
v_valid_pins = g_pin.new_vp("bool")
for l in labels:
    v_valid_pins.a[comp.a==l] = True
print(v_valid_pins.a.sum())

### get subgraphs
e_label = g_pin.new_ep("bool")
e_label.a = False
e_ar = g_pin.get_edges(eprops=[e_id])
v_ar = g.get_vertices(vprops=[v_is_buf, v_is_inv, v_valid_pins])
src = e_ar[:,0]
tar = e_ar[:,1]
idx = e_ar[:,2]
mask = (v_ar[src, -1] == True) & (v_ar[tar, -1] == True)
e_label.a[idx[mask]] = True
u = get_subgraph(g_pin, v_valid_pins, e_label)

### mark selected pins ###
pin_df["selected"] = v_valid_pins.a[0:N_pin]
###

### get buffer tree start and end points
v_bt_s = g.new_vp("bool")
v_bt_e = g.new_vp("bool")
v_bt_s.a = False
v_bt_e.a = False

e_ar = u.get_edges()
v_ar = g.get_vertices(vprops=[v_is_buf, v_is_inv])
src = e_ar[:,0]
tar = e_ar[:,1]
src_isbuf = v_ar[src,1]
src_isinv = v_ar[src,2]
tar_isbuf = v_ar[tar,1]
tar_isinv = v_ar[tar,2]
is_s = (tar_isbuf | tar_isinv ) & np.logical_not(src_isbuf) & np.logical_not(src_isinv)
v_bt_s.a[src[is_s==1]] = True

src_iss = v_bt_s.a[src]==True
is_e = (src_isbuf | src_isinv | src_iss) & np.logical_not(tar_isbuf) & np.logical_not(tar_isinv)
v_bt_e.a[tar[is_e==1]] = True
print("buf tree start cnt: ", v_bt_s.a.sum(), "buf tree end cnt: ", v_bt_e.a.sum())

### get buf tree start pin id ###
v_net_id = g.new_vp("int")
v_net_id.a[0:N_pin] = pin_df.net_id.to_numpy()
mask = v_bt_s.a < 1
v_net_id.a[mask] = 0

### mark buffer trees
v_tree_id = g.new_vp("int")
v_tree_id.a = 0
v_polarity = g.new_vp("bool")
v_polarity.a = True
e_tree_id = g.new_ep("int")
e_tree_id.a = 0

tree_end_list = []
buf_list = []

v_all = g.get_vertices()
l = np.array(list(range(1, int(v_bt_s.a.sum())+1)))
v_tree_id.a[v_bt_s.a>0] = l
loc = v_all[v_bt_s.a>0]
out_v_list = []
for i in loc:
    out_e = u.get_out_edges(i, eprops=[e_id])
    out_v = out_e[:,1]
    v_tree_cnt = v_tree_id[i]
    net_id = v_net_id[i]
    e_tree_id.a[out_e[:,-1]] = v_tree_cnt
    v_tree_id.a[out_v] = v_tree_cnt
    v_net_id.a[out_v] = net_id
    tree_end_list.append(out_v[(v_is_buf.a[out_v]==False) & (v_is_inv.a[out_v]==False)])
    out_v = out_v[(v_is_buf.a[out_v]==True) | (v_is_inv.a[out_v]==True)]
    buf_list.append(out_v)
    out_v_list.append(out_v)
new_v = np.concatenate(out_v_list, axis=0)
N,  = new_v.shape
print("num of buffer tree out pins: ", N)
while N > 0:
    out_v_list = []
    for i in new_v:
        if v_is_buf[i]:
            out_e = u.get_out_edges(i, eprops=[e_id])
            out_v = out_e[:,1]
            v_tree_cnt = v_tree_id[i]
            net_id = v_net_id[i]
            v_p = v_polarity.a[i]
            e_tree_id.a[out_e[:,-1]] = v_tree_cnt
            v_tree_id.a[out_v] = v_tree_cnt
            v_net_id.a[out_v] = net_id
            v_polarity.a[out_v] = v_p
            tree_end_list.append(out_v[(v_is_buf.a[out_v]==False) & (v_is_inv.a[out_v]==False)])
            out_v = out_v[(v_is_buf.a[out_v]==True) | (v_is_inv.a[out_v]==True)]
            buf_list.append(out_v)
            out_v_list.append(out_v)
        else:
            out_e = u.get_out_edges(i, eprops=[e_id])
            out_v = out_e[:,1]
            v_tree_cnt = v_tree_id[i]
            net_id = v_net_id[i]
            v_p = v_polarity.a[i]
            e_tree_id.a[out_e[:,-1]] = v_tree_cnt
            v_tree_id.a[out_v] = v_tree_cnt
            v_net_id.a[out_v] = net_id
            if v_dir[i]:
                v_polarity.a[out_v] = not v_p
            else:
                v_polarity.a[out_v] = v_p
            ###
            tree_end_list.append(out_v[(v_is_buf.a[out_v]==False) & (v_is_inv.a[out_v]==False)])
            ###
            out_v = out_v[(v_is_buf.a[out_v]==True) | (v_is_inv.a[out_v]==True)]
            ###
            buf_list.append(out_v)
            ###
            out_v_list.append(out_v)
    new_v = np.concatenate(out_v_list, axis=0)
    N, = new_v.shape
    print("num of buffer tree out pins: ", N)

### get actual number of BT end pin cnt
tree_end_list_new = np.concatenate(tree_end_list, axis=0)
print(tree_end_list_new.shape[0], v_bt_e.a.sum())
N_bt_e = tree_end_list_new.shape[0]
v_bt_e = g.new_vp("bool")
v_bt_e.a = False
v_bt_e.a[tree_end_list_new] = True
print(v_bt_e.a.sum())

pin_df["net_id_rm_bt"] = pin_df["net_id"]
pin_df.loc[tree_end_list_new, ["net_id_rm_bt"]] = v_net_id.a[tree_end_list_new]

############################################
#Gathering dataset for training and testing#
############################################

### get selected pins ###
selected_pin_df = pin_df[(pin_df.selected == True) & (pin_df.is_buf == False) & (pin_df.is_inv == False)]

### get driver pins and related properties ###
driver_pin = selected_pin_df[selected_pin_df.dir==0]
driver_pin_info = driver_pin.loc[:, ["id", "net_id", "x", "y", "cell_id", "risearr", "fallarr"]]
driver_pin_info = driver_pin_info.rename(columns={"id":"driver_pin_id", "x":"driver_x", "y":"driver_y", "cell_id":"driver_id", "risearr":"driver_risearr", "fallarr":"driver_fallarr"})
cell_info = cell_df.loc[:, ["id", "libcell_id", "fo4_delay", "fix_load_delay"]]
cell_info = cell_info.rename(columns={"id":"driver_id", "y":"driver_y"})
driver_pin_info = driver_pin_info.merge(cell_info, on="driver_id", how="left")

### get sink pins and related properties ###
sink_pin = selected_pin_df[selected_pin_df.dir==1]
sink_pin_info = sink_pin.loc[:, ["id", "x", "y", "cap", "net_id", "cell_id", "risearr", "fallarr"]]
sink_pin_info = sink_pin_info.merge(driver_pin_info, on="net_id", how="left")

sink_pin_info.x = sink_pin_info.x - sink_pin_info.driver_x
sink_pin_info.y = sink_pin_info.y - sink_pin_info.driver_y
idx = sink_pin_info[pd.isna(sink_pin_info.driver_x)].index
sink_pin_info = sink_pin_info.drop(idx)

### get context sink locations ###
sink_loc = sink_pin_info.groupby('net_id', as_index=False).agg({'x': ['mean', 'min', 'max', 'std'], 'y': ['mean', 'min', 'max', 'std'], 'cap': ['sum']})
sink_loc.columns = ['_'.join(col).rstrip('_') for col in sink_loc.columns.values]
sink_loc['x_std'] = sink_loc['x_std'].fillna(0)
sink_loc['y_std'] = sink_loc['y_std'].fillna(0)

### merge information and rename ###
sink_pin_info = sink_pin_info.merge(sink_loc, on="net_id", how="left")
sink_pin_info = sink_pin_info.rename(columns={"libcell_id":"driver_libcell_id", "fo4_delay":"driver_fo4_delay", "fix_load_delay":"driver_fix_load_delay", \
                                              "x_mean": "context_x_mean", "x_min": "context_x_min", "x_max": "context_x_max", "x_std": "context_x_std", \
                                             "y_mean": "context_y_mean", "y_min": "context_y_min", "y_max": "context_y_max", "y_std": "context_y_std", \
                                             "risearr":"sink_risearr", "fallarr":"sink_fallarr"})
sink_pin_info["sink_arr"] = sink_pin_info[["sink_risearr", "sink_fallarr"]].min(axis=1)
sink_pin_info["driver_arr"] = sink_pin_info[["driver_risearr", "driver_fallarr"]].min(axis=1)

### get cell arc delays ###
cell_arc = pin_pin_df.groupby('tar_id', as_index=False).agg({'arc_delay': ['mean', 'min', 'max']})
cell_arc.columns = ['_'.join(col).rstrip('_') for col in cell_arc.columns.values]
cell_arc = cell_arc.rename(columns={"tar_id":"driver_pin_id"})
sink_pin_info = sink_pin_info.astype({"driver_pin_id":"int"})
sink_pin_info = sink_pin_info.merge(cell_arc, on="driver_pin_id", how="left")
idx = sink_pin_info[pd.isna(sink_pin_info.arc_delay_mean)].index
sink_pin_info = sink_pin_info.drop(idx)

### get net delay ###
cell_arc = cell_arc.rename(columns={"driver_pin_id":"id", "arc_delay_mean":"net_delay_mean", "arc_delay_min":"net_delay_min", "arc_delay_max":"net_delay_max"})
sink_pin_info = sink_pin_info.merge(cell_arc, on="id", how="left")

### stage delay = driver cell arc delay + net delay ###
sink_pin_info["stage_delay"] = sink_pin_info.arc_delay_max + sink_pin_info.net_delay_max

print(sink_pin_info)

# x, y: distance between driver and the target sink
# cap, cap_sum: sink capacitance
# driver_fo4_delay driver_fix_load_delay: driving strength of the driver cell
# context_x_mean", context_x_min, context_x_max, context_x_std, context_y_mean, context_y_min, context_y_max, context_y_std: Context sink locations
features = sink_pin_info.loc[:, ["x", "y", "cap", "cap_sum", "driver_fo4_delay", "driver_fix_load_delay", \
                                 "context_x_mean", "context_x_min", "context_x_max", "context_x_std", \
                                "context_y_mean", "context_y_min", "context_y_max", "context_y_std"]].to_numpy().astype(float)
labels = sink_pin_info.loc[:, ["stage_delay"]].to_numpy().astype(float)

features = preprocessing.normalize(features, axis=0)
labels = preprocessing.normalize(labels, axis=0)
labels = labels.reshape([-1,])

nb_samples = features.shape[0]
nb_feat = features.shape[1]

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.05)

nb_train_samples = train_x.shape[0]
nb_test_samples = train_y.shape[0]

print("Training Machine Learning Model")

nb_estim = 500
max_feat = 0.5
model = RandomForestRegressor(n_estimators=nb_estim, max_features=max_feat)
model.fit(train_x, train_y)

pred = model.predict(train_x)

plt.plot(pred, train_y, "*")

pred = model.predict(test_x)

plt.plot(pred, test_y, "*")


