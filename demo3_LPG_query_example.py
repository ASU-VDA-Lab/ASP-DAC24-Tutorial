import graph_tool as gt
import pickle
import numpy as np
import pandas as pd
import sys

#from generate_LPG_from_tables import generate_LPG_from_tables
#import importlib.util
import argparse

#############################################################################################
#LPG dataframe properties:                                                                  #
#('id' maps to the node id in LPG)                                                          #
#pin_df: 'name', 'x', 'y', 'is_in_clk', 'is_port', 'is_start', 'is_end', 'dir',             #
#        'maxcap', 'maxtran', 'num_reachable_endpoint', 'cellname', 'netname',              #
#        'tran', 'slack', 'risearr', 'fallarr', 'cap', 'is_macro', 'is_seq',                #
#        'is_buf', 'is_inv', 'new_cellname', 'new_netname', 'id'                            #
#                                                                                           #
#cell_df: 'name', 'is_seq', 'is_macro', 'is_in_clk', 'x0', 'y0', 'x1', 'y1',                #
#         'is_buf', 'is_inv', 'ref', 'staticpower', 'dynamicpower', 'x', 'y',               #
#         'new_cellname', 'id'                                                              #
#                                                                                           #
#net_df: 'name', 'net_route_length', 'net_steiner_length', 'fanout', 'total_cap',           #
#        'net_cap', 'net_coupling', 'net_res', 'id'                                         #
#                                                                                           #
#fo4_df: 'ref', 'func_id', 'libcell_area', 'worst_input_cap', 'libcell_leakage',            #
#        'fo4_delay', 'libcell_delay_fixed_load', 'libcell_id'                              #
#                                                                                           #
#pin_pin_df: 'src', 'tar', 'src_type', 'tar_type', 'is_net', 'arc_delay', 'src_id', 'tar_id'#
#cell_pin_df: 'src', 'tar', 'src_type', 'tar_type', 'src_id', 'tar_id'                      #
#net_pin_df: 'src', 'tar', 'src_type', 'tar_type', 'src_id', 'tar_id'                       #
#net_cell_df: 'src', 'tar', 'src_type', 'tar_type', 'src_id', 'tar_id'                      #
#cell_cell_df: 'src', 'tar', 'src_type', 'tar_type', 'src_id', 'tar_id'                     #
#edge_df: 'src_id', 'tar_id', 'e_type'                                                      #
#                             (etype: 0-p_p, 1-c_p, 2-n_p, 3-n_c, 4-c_c)                    #
#############################################################################################

######################################################################################
#buffer tree node & edge properties:                                                 #
#                                                                                    #
#node: index, v_tree_id, v_BT_height, v_bt_s, v_x, v_y, v_risearr, v_fallarr, v_tran,#
#      v_polarity, v_libcell_id                                                      #
#                                                                                    #
#edge: src_index, tar_index, e_tree_id                                               #
######################################################################################

def load_BT(pickle_path):
  with open(pickle_path + 'BT_nodes.pkl', 'rb') as fn, open(pickle_path + 'BT_edges.pkl', 'rb') as fe:
    nodes = pickle.load(fn)
    edges = pickle.load(fe)
    return nodes, edges

def get_node_index_property_list(bt, prop_index):
  prop = []
  for i in bt:
    for j in i:
      prop.append([int(j[0]), j[prop_index]])
  return prop

if __name__ == "__main__":
  #################################################
  #parse args and import functions from CircuitOps#
  #################################################
  parser = argparse.ArgumentParser(description='path of your CircuitOps clone (must include /CircuitO')
  parser.add_argument('--path_BT', type = str, default='./', action = 'store')
  parser.add_argument('--path_IR', type = str, default='./', action = 'store')
  parser.add_argument('--path_CircuitOps_py_scripts', type = str, default='./', action = 'store')
  pyargs = parser.parse_args()
  
  sys.path.append(pyargs.path_CircuitOps_py_scripts)  
  from generate_LPG_from_tables import generate_LPG_from_tables
  ##############################
  #get feature from buffer tree#
  ##############################
  nodes, edges = load_BT(pyargs.path_BT)

  BT_v_x = get_node_index_property_list(nodes, 4)
  BT_v_y = get_node_index_property_list(nodes, 5)
  BT_v_risearr = get_node_index_property_list(nodes, 6)
  BT_v_fallarr = get_node_index_property_list(nodes, 7)
  BT_v_tran = get_node_index_property_list(nodes, 8)
  BT_v_polarity = get_node_index_property_list(nodes, 9)
  BT_v_libcell_id = get_node_index_property_list(nodes, 10)

  ######################
  #get feature from LPG#
  ######################
  LPG, pin_df, cell_df, net_df, fo4_df, pin_pin_df, cell_pin_df, \
    net_pin_df, net_cell_df, cell_cell_df, edge_df, v_type, e_type \
    = generate_LPG_from_tables(pyargs.path_IR)
  
  sys.path.remove(pyargs.path_CircuitOps_py_scripts)
  ### get dimensions
  N_pin, _ = pin_df.shape
  N_cell, _ = cell_df.shape
  N_net, _ = net_df.shape
  total_v_cnt = N_pin+N_cell+N_net

  N_pin_pin, _ = pin_pin_df.shape
  N_cell_pin, _ = cell_pin_df.shape
  N_net_pin, _ = net_pin_df.shape
  N_net_cell, _ = net_cell_df.shape
  N_cell_cell, _ = cell_cell_df.shape
  total_e_cnt = N_pin_pin + N_cell_pin + N_net_pin + N_net_cell + N_cell_cell
  
  #string type properties not supported
  LPG_pin_id = LPG.new_vp("int")
  LPG_pin_id.a[0:N_pin] = pin_df["id"].to_numpy()

  LPG_pin_x = LPG.new_vp("float")
  LPG_pin_x.a[0:N_pin] = pin_df["x"].to_numpy()

  LPG_pin_y = LPG.new_vp("float")
  LPG_pin_y.a[0:N_pin] = pin_df["y"].to_numpy()
  
  LPG_cell_is_seq = LPG.new_vp("bool")
  LPG_cell_is_seq.a[N_pin:N_pin+N_cell] = cell_df["is_seq"].to_numpy()

  LPG_net_net_steiner_length = LPG.new_vp("float")
  LPG_net_net_steiner_length.a[N_pin+N_cell:N_pin+N_cell+N_net] = net_df["net_steiner_length"].to_numpy()

  v_props = LPG.get_vertices(vprops = [LPG_pin_id, LPG_pin_x, LPG_pin_y, LPG_cell_is_seq, LPG_net_net_steiner_length])
  #v_props will contain the node index, pin_id, pin_x, pin_y, cell_is_seq, net_steiner_length



