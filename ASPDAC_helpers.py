import openroad as ord
import pdn, odb, utl
from openroad import Tech, Design
from collections import defaultdict, namedtuple
import argparse
import numpy as np
import math, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, RelGraphConv
import matplotlib
import matplotlib.pyplot as plt
import random
from itertools import count
import dgl
from tqdm import tqdm
from time import time
import copy
from pathlib import Path

def OpenROAD_map_creation(map_type, tech_design, design, corner, congestion_layer):
  block = ord.get_db_block()
  insts = block.getInsts()
  nets = block.getNets()
  #################################
  #get unit for lef unit transform#
  #################################
  unit = block.getDbUnitsPerMicron()
  ###############
  #get core bbox#
  ###############
  core_x0 = block.getCoreArea().xMin()
  core_y0 = block.getCoreArea().yMin()
  core_x1 = block.getCoreArea().xMax()
  core_y1 = block.getCoreArea().yMax()
  ##############
  #get die bbox#
  ##############
  die_x0 = block.getDieArea().xMin()
  die_y0 = block.getDieArea().yMin()
  die_x1 = block.getDieArea().xMax()
  die_y1 = block.getDieArea().yMax()
  ################
  #get row height#
  ################
  row_height = block.getRows()[1].getOrigin()[1] - block.getRows()[0].getOrigin()[1]
  #################
  #get track width#
  #################
  track_width = block.getTrackGrids()[0].getGridX()[1] - block.getTrackGrids()[0].getGridX()[0]
  ################
  #get gcell grid#
  ################
  gcell_grid_x = block.getGCellGrid().getGridX()
  gcell_grid_x_delta = gcell_grid_x[1] - gcell_grid_x[0]
  gcell_grid_y = block.getGCellGrid().getGridY()
  gcell_grid_y_delta = gcell_grid_y[1] - gcell_grid_y[0]
  #########################
  #generate feature map(s)#
  #########################
  image_width = (core_x1 - core_x0) // track_width
  if (core_x1 - core_x0)%track_width > 0:
    image_width += 1
  image_height = (core_y1 - core_y0) // row_height
  if (core_y1 - core_y0)%row_height > 0:
    image_height += 1
  feature_map = np.zeros((image_width, image_height))
  ###############################
  #assign congestion and IR drop#
  ###############################
  if map_type == "static_IR" or map_type == "congestion":
    db_tech = tech_design.getDB().getTech()
    if map_type == "static_IR":
      #run pdn analysis#
      psm_obj = design.getPDNSim()
      psm_obj.setNet(ord.Tech().getDB().getChip().getBlock().findNet("VDD"))
      design.evalTclString(f"psm::set_corner [sta::cmd_corner]")
      psm_obj.analyzePowerGrid('', False, '', '')
      #extract value#
      layers = db_tech.getLayers()
      drops = psm_obj.getIRDropForLayer(layers[2])
      for pt,v in drops.items():
        if pt.x() < core_x0 or pt.x() > core_x1 or pt.y() < core_y0 or pt.y() > core_y1:
          continue
  
        anchor_x = core_x1 - 1 if pt.x() == core_x1 else pt.x()
        anchor_y = core_y1 - 1 if pt.y() == core_y1 else pt.y()
  
        if v > feature_map[(anchor_x - core_x0)//track_width][(anchor_y - core_y0)//row_height]:
          feature_map[(anchor_x - core_x0)//track_width][(anchor_y - core_y0)//row_height] = v
    else:
      layers = db_tech.getLayers()
      layer = layers[congestion_layer]
      for x in range(len(gcell_grid_x)):
        for y in range(len(gcell_grid_y)):
          capacity = block.getGCellGrid().getHorizontalCapacity(layer, x, y)
          usage = block.getGCellGrid().getHorizontalUsage(layer, x, y)
          congestion = usage - capacity
          
          if gcell_grid_x[x] < core_x0:
            if gcell_grid_x[x] - core_x0 + gcell_grid_x_delta < core_x0:
              continue
          if gcell_grid_y[y] < core_y0:
            if gcell_grid_y[y] + gcell_grid_y_delta < core_y0:
              continue
          if gcell_grid_x[x] >= core_x1 or gcell_grid_y[y] >= core_y1:
            continue

          anchor_x = (gcell_grid_x[x] - core_x0)//track_width if gcell_grid_x[x] - core_x0 >= 0 else 0
          anchor_y = (gcell_grid_y[y] - core_y0)//row_height if gcell_grid_y[y] - core_y0 >= 0 else 0
          
          for delta_x in range(math.ceil(gcell_grid_x_delta/track_width)):
            for delta_y in range(math.ceil(gcell_grid_y_delta/row_height)):
              if anchor_x + delta_x >= feature_map.shape[0] or anchor_y + delta_y >= feature_map.shape[1]:
                continue
              
              if congestion > feature_map[int(anchor_x + delta_x)][int(anchor_y + delta_y)]:
                feature_map[int(anchor_x + delta_x)][int(anchor_y + delta_y)] = congestion
  ################################################################################
  #assign static_power and dynamic_power value by iterating through each instance#
  ################################################################################
  if map_type == "static_power" or map_type == "dynamic_power":
    for inst in insts:
      ###############
      #get cell bbox#
      ###############
      inst_x0 = inst.getBBox().xMin()
      inst_y0 = inst.getBBox().yMin()
      inst_x1 = inst.getBBox().xMax()
      inst_y1 = inst.getBBox().yMax()

      anchor_index_x = (inst_x0 - core_x0) // track_width
      anchor_index_y = (inst_y0 - core_y0) // row_height
      #############
      #get feature#
      #############
      if map_type == "static_power":
        feature = design.staticPower(inst, corner)
        feature /= ((inst_x1 - inst_x0) * (inst_y1 - inst_y0))
      elif map_type == "dynamic_power":
        feature = design.dynamicPower(inst, corner)
        feature /= ((inst_x1 - inst_x0) * (inst_y1 - inst_y0))
      ###################################################
      #compute the amount of pixels covered by this cell#
      #in case there are non-interger-track-width cells #
      #or off track DRC                                 #
      ###################################################
      covered_horizon_pixel_cnt = (inst_x1 - inst_x0) // track_width
      if (inst_x0 - core_x0) % track_width > 0:
        covered_horizon_pixel_cnt += 1
      if (inst_x1 - core_x0) % track_width > 0:
        covered_horizon_pixel_cnt += 1
      #############################################
      #in case there are non-interger-height cells#
      #############################################
      covered_vertical_pixel_cnt = (inst_y1 - inst_y0) // row_height
      if (inst_y0 - core_y0) % row_height > 0:
        covered_vertical_pixel_cnt += 1
      if (inst_y0 - core_y0) % row_height > 0:
        covered_vertical_pixel_cnt += 1
      ##############
      #assign value#
      ##############
      for y in range(covered_vertical_pixel_cnt):
        for x in range(covered_horizon_pixel_cnt):
          
          if y == 0 and y == covered_vertical_pixel_cnt -1:
            tmp_height = row_height
          elif y == 0 and y != covered_vertical_pixel_cnt -1:
            tmp_height = row_height - (inst_y0 % row_height)
          elif y != 0 and y == covered_vertical_pixel_cnt -1:
            tmp_height = inst_y1 % row_height
            if tmp_height == 0:
              tmp_height = row_height
          else:
            tmp_height = row_height

          if x == 0 and x == covered_horizon_pixel_cnt -1:
            tmp_width = inst_x1 - inst_x0
          elif x == 0 and x != covered_horizon_pixel_cnt -1:
            tmp_width = track_width - (inst_x0 % track_width)
          elif x != 0 and x == covered_horizon_pixel_cnt -1:
            tmp_width = inst_x1 % track_width
            if tmp_width == 0:
              tmp_width = track_width
          else:
            tmp_width = track_width

          cover_area = tmp_height * tmp_width
          feature_map[anchor_index_x + x][anchor_index_y + y] += feature * cover_area
  return row_height, track_width, feature_map


class CircuitOps_File_DIR:
  def __init__(self):
    ### SET DESIGN ###
    self.DESIGN_NAME = "gcd"
    #self.DESIGN_NAME = "aes"
    #self.DESIGN_NAME = "bp_fe"
    #self.DESIGN_NAME = "bp_be"

    ### SET PLATFORM ###
    self.PLATFORM = "nangate45"

    ### SET OUTPUT DIRECTORY ###
    self.OUTPUT_DIR = "./IRs/" + self.PLATFORM + "/" + self.DESIGN_NAME
    self.create_path()

    ### INTERNAL DEFINTIONS: DO NOT MODIFY BELOW ####
    self.CIRCUIT_OPS_DIR = "./"
    self.DESIGN_DIR = self.CIRCUIT_OPS_DIR + "/designs/" + self.PLATFORM + "/" + self.DESIGN_NAME
    self.PLATFORM_DIR = self.CIRCUIT_OPS_DIR + "/platforms/" + self.PLATFORM

    self.DEF_FILE = self.DESIGN_DIR + "/6_final.def.gz"
    self.TECH_LEF_FILE = [os.path.join(root, file) for root, _, files in os.walk(self.PLATFORM_DIR + "/lef/") for file in files if file.endswith("tech.lef")]
    self.LEF_FILES = [os.path.join(root, file) for root, _, files in os.walk(self.PLATFORM_DIR + "/lef/") for file in files if file.endswith(".lef")]
    self.LIB_FILES = [os.path.join(root, file) for root, _, files in os.walk(self.PLATFORM_DIR + "/lib/") for file in files if file.endswith(".lib")]
    self.SDC_FILE = self.DESIGN_DIR + "/6_final.sdc.gz"
    self.NETLIST_FILE = self.DESIGN_DIR + "/6_final.v"
    self.SPEF_FILE = self.DESIGN_DIR + "/6_final.spef.gz"

    self.cell_file = self.OUTPUT_DIR + "/cell_properties.csv"
    self.libcell_file = self.OUTPUT_DIR + "/libcell_properties.csv"
    self.pin_file = self.OUTPUT_DIR + "/pin_properties.csv"
    self.net_file = self.OUTPUT_DIR + "/net_properties.csv"
    self.cell_pin_file = self.OUTPUT_DIR + "/cell_pin_edge.csv"
    self.net_pin_file = self.OUTPUT_DIR + "/net_pin_edge.csv"
    self.pin_pin_file = self.OUTPUT_DIR + "/pin_pin_edge.csv"
    self.cell_net_file = self.OUTPUT_DIR + "/cell_net_edge.csv"
    self.cell_cell_file = self.OUTPUT_DIR + "/cell_cell_edge.csv"

  def create_path(self):
    if not(os.path.exists(self.OUTPUT_DIR)):
      os.mkdir(self.OUTPUT_DIR)


def add_global_connection(design, *,
                          net_name=None,
                          inst_pattern=None,
                          pin_pattern=None,
                          power=False,
                          ground=False,
                          region=None):
  if net_name is None:
    utl.error(utl.PDN, 1501, "The net option for the " +
                  "add_global_connection command is required.")

  if inst_pattern is None:
    inst_pattern = ".*"

  if pin_pattern is None:
    utl.error(utl.PDN, 1502, "The pin_pattern option for the " +
                  "add_global_connection command is required.")

  net = design.getBlock().findNet(net_name)
  if net is None:
    net = odb.dbNet_create(design.getBlock(), net_name)

  if power and ground:
    utl.error(utl.PDN, 1551, "Only power or ground can be specified")
  elif power:
    net.setSpecial()
    net.setSigType("POWER")
  elif ground:
    net.setSpecial()
    net.setSigType("GROUND")

  # region = None
  if region is not None:
    region = design.getBlock().findRegion(region)
    if region is None:
      utl.error(utl.PDN, 1504, f"Region {region} not defined")

  design.getBlock().addGlobalConnect(region, inst_pattern, pin_pattern, net, True)


def load_design(_CircuitOps_File_DIR):
  tech = Tech()
  tech_design = ord.Tech()
  for libFile in _CircuitOps_File_DIR.LIB_FILES:
    tech.readLiberty(libFile)
  for techFile in _CircuitOps_File_DIR.TECH_LEF_FILE:
    tech.readLef(techFile)
  for lefFile in _CircuitOps_File_DIR.LEF_FILES:
    tech.readLef(lefFile)

  design = Design(tech)
  design.readDef(_CircuitOps_File_DIR.DEF_FILE)
  design.evalTclString("read_sdc " + _CircuitOps_File_DIR.SDC_FILE)
  design.evalTclString("read_spef " + _CircuitOps_File_DIR.SPEF_FILE)
  design.evalTclString("set_propagated_clock [all_clocks]")
  add_global_connection(design, net_name="VDD", pin_pattern="VDD", power=True)
  add_global_connection(design, net_name="VSS", pin_pattern="VSS", ground=True)
  odb.dbBlock.globalConnect(ord.get_db_block())
  return tech_design, design



###########

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()

    self.enc_conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
    self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

    self.dec_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
    self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
    self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.dec_conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

  def forward(self, x0):
    x0 = torch.cat(x0, dim=1)
    x1 = nn.functional.relu(self.enc_conv1(x0))
    x2 = nn.functional.relu(self.enc_conv2(self.pool(x1)))
    x3 = nn.functional.relu(self.enc_conv3(self.pool(x2)))
    x4 = nn.functional.relu(self.enc_conv4(self.pool(x3)))

    x = nn.functional.relu(self.upconv1(x4))
    x = torch.cat([x, x3], dim=1)
    x = nn.functional.relu(self.dec_conv1(x))

    x = nn.functional.relu(self.upconv2(x))
    x = torch.cat([x, x2], dim=1)
    x = nn.functional.relu(self.dec_conv2(x))

    x = nn.functional.relu(self.upconv3(x))
    x = torch.cat([x, x1], dim=1)
    x = nn.functional.relu(self.dec_conv3(x))
  
    x = self.dec_conv4(x)

    return x

def handle_size(input_array, target_size):
  processed = list()
  if input_array.shape[0] >= target_size[0]:
    if input_array.shape[1] > target_size[1]:
      for x in range(0, int(np.ceil(input_array.shape[0]/target_size[0]))):
        if (x + 1) * target_size[0] > input_array.shape[0]:
          for y in range(0, int(np.ceil(input_array.shape[1]/target_size[1]))):
            if (y + 1) * target_size[1] > input_array.shape[1]:
              processed.append(input_array[input_array.shape[0] - target_size[0] : input_array.shape[0], input_array.shape[1] - target_size[1] : input_array.shape[1]])
            else:
              processed.append(input_array[input_array.shape[0] - target_size[0] : input_array.shape[0], y * target_size[1] : (y + 1) * target_size[1]])
        else:
          for y in range(0, int(np.ceil(input_array.shape[1]/target_size[1]))):
            if (y + 1) * target_size[1] > input_array.shape[1]:
              processed.append(input_array[x * target_size[0] : (x + 1) * target_size[0], input_array.shape[1] - target_size[1] : input_array.shape[1]])
            else:
              processed.append(input_array[x * target_size[0] : (x + 1) * target_size[0], y * target_size[1] : (y + 1) * target_size[1]])
    elif input_array.shape[1] == target_size[1]:
      for x in range(0, int(np.ceil(input_array.shape[0]/target_size[0]))):
        if (x + 1) * target_size[0] > input_array.shape[0]:
          processed.append(input_array[input_array.shape[0] - target_size[0] : input_array.shape[0]])
        else:
          processed.append(input_array[x * target_size[0] : (x + 1) * target_size[0]])
    else:
      for x in range(0, int(np.ceil(input_array.shape[0]/target_size[0]))):
        if (x + 1) * target_size[0] > input_array.shape[0]:
          processed.append(np.pad(input_array[input_array.shape[0] - target_size[0] : input_array.shape[0]], ((0, 0), (0, target_size[1] - input_array.shape[1])), mode = 'constant', constant_values = 0))
        else:
          processed.append(np.pad(input_array[x * target_size[0] : (x + 1) * target_size[0]], ((0, 0), (0, target_size[1] - input_array.shape[1])), mode = 'constant', constant_values = 0))
  else:
    if input_array.shape[1] > target_size[1]:
      for y in range(0, int(np.ceil(input_array.shape[1]/target_size[1]))):
        if (y + 1) * target_size[1] > input_array.shape[1]:
          processed.append(np.pad(input_array[:, input_array.shape[1] - target_size[1] : input_array.shape[1]], ((0, target_size[0] - input_array.shape[0]), (0, 0)), mode = 'constant', constant_values = 0))
        else:
          processed.append(np.pad(input_array[:, y * target_size[1] : (y + 1) * target_size[1]], ((0, target_size[0] - input_array.shape[0]), (0, 0)), mode = 'constant', constant_values = 0))
    elif input_array.shape[1] == target_size[1]:
      processed.append(np.pad(input_array, ((0, target_size[0] - input_array.shape[0]), (0, 0)), mode = 'constant', constant_values = 0))
    else:
      processed.append(np.pad(input_array, ((0, target_size[0] - input_array.shape[0]), (0, target_size[1] - input_array.shape[1])), mode = 'constant', constant_values = 0))
  return processed

################

# replay memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('graph', 'action', 'next_state', 'reward'))
        
    # insert if not yet filled else trea it like a circular buffer and add.
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # random sampling for the training step
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

  def __init__(self, n_state, n_cells, n_features, device):
    super(DQN, self).__init__()
    self.conv1 = RelGraphConv(n_state,64,2)
    self.conv2 = RelGraphConv(64,64,2)
    self.conv3 = RelGraphConv(64,2,2)
    self.device = device
    self.n_state = n_state
    self.n_cells = n_cells
    self.n_features = n_features
  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def  forward(self, graph, x=None):
    if x is None:
      x = get_state(graph, self.n_state, self.n_cells, self.n_features).to(self.device)
    e = graph.edata['types']
    x = F.relu(self.conv1(graph, x, e))
    x = F.relu(self.conv2(graph, x, e))
    x = self.conv3(graph, x, e)
    # Get a list of actions that are not valid and ensure they cant be selected.
    mask = generate_masked_actions(graph)
    x = x*(~mask) + (mask)*(x.min() - 1)
    #x = F.log_softmax(x.view(-1),dim=0)
    return x

def get_type(cell_type, cell_dict, cell_name_dict):
  cell, drive = cell_type.split("_")
  drive = "_"+drive
  if cell in cell_name_dict:
    cell_values = cell_dict[cell_name_dict[cell]]
    if drive in cell_values['sizes']:
      idx = cell_values['sizes'].index(drive)
      return int(cell_name_dict[cell]), idx
    else:
      print("Drive strength "+drive+" not found in cell :"+cell)
      print("Possible sizes"+cell_values['sizes'])
      return None,None
  else:
    print("cell: "+cell+" not in dictionary")
    return None,None

def pin_properties(pin_name, CLKset, ord_design, timing):
  pin = ord_design.getStaPin(pin_name)
  dbpin = timing.staToDBPin_ITerm(pin)
  ITerms = dbpin.getNet().getITerms()
  #slack
  slack = min(timing.getPinMaxFallSlack(pin), timing.getPinMaxRiseSlack(pin))
  if slack < -0.5*CLKset[0]:
    slack = 0
  #slew
  slew = timing.getPinSlew(dbpin)  
  #load
  Corners = timing.getCorners()
  load = 0
  for ITerm in ITerms:
    if ITerm.isInputSignal():
      tmp_pin_name = ord_design.getITermName(ITerm)
      tmp_sta_pin = ord_design.getStaPin(tmp_pin_name)
      tmp_pin_port = ord_design.Pin_liberty_port(tmp_sta_pin)
      tmp_load = 0
      for Corner in Corners:
        tmp_capacitance = ord_design.LibertyPort_capacitance(tmp_pin_port,\
                                                        Corner,\
                                                        ord_design.getStaMinMax("max"))
        if tmp_capacitance > tmp_load:
          tmp_load = tmp_capacitance
      load += tmp_load

  return slack, slew, load

def min_slack(inst_name, cell_dict, inst_dict, ord_design, timing):
  pin_name = inst_name + cell_dict[str(inst_dict[inst_name]['cell_type'][0])]['out_pin']
  pin = ord_design.getStaPin(pin_name)
  slack = min(timing.getPinMaxFallSlack(pin), timing.getPinMaxRiseSlack(pin))  
  return slack

def generate_masked_actions(graph):
  # max size keep track of the index of the maximum size.
  # If the current size is maximum size we mask it out as an action
  upper_mask = graph.ndata['cell_types'][:,1] >= graph.ndata['max_size']-1
  lower_mask = graph.ndata['cell_types'][:,1] == 0
  ###########################
  #upper_mask = upper_mask | (~slack_mask | ~slew_mask| ~load_mask | ~DFF_mask)
  #         if len(delay_mask)!=0:
  #             upper_mask[torch.tensor(delay_mask)] = True
  ##########################
  # if the criteria for the mask is met we replace it with the minimum
  # to make sure that that action is never chosen
  mask = torch.cat((upper_mask.view(-1,1), lower_mask.view(-1,1)),1)
  return mask

def update_lambda(initial_lambda, slacks, K):
    Slack_Lambda = initial_lambda * ((1-slacks)**K)
    return Slack_Lambda

def optimize_model(memory, BATCH_SIZE, device, GAMMA, policy_net,\
                  target_net, optimizer, loss_history):
  st=time()
  if len(memory) < BATCH_SIZE:
    return optimizer, loss_history
  transitions = memory.sample(BATCH_SIZE)
  Transition = namedtuple('Transition', ('graph', 'action', 'next_state', 'reward'))
  batch = Transition(*zip(*transitions))
  #     print("optim 0.1 %5.4f"%(time() -st))
  st = time()
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  #     print("optim 0.2 %5.4f"%(time() -st))
  st = time()
  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = torch.zeros(BATCH_SIZE, device=device,dtype=torch.float32)
  #     bg = dgl.batch(batch.graph)
  #     actions = policy_net(bg)
  #     print(actions.shape)
  #     tot = 0
  #     for n,nb in enumerate(bg.batch_num_nodes()):
  #         state_action_values[n] = actions[tot:tot+nb,:].view(-1)[action_batch[n,0]]
  #         tot+=nb
  #     print(state_action_values.shape)
  for n_state, graph in enumerate(batch.graph):
    #         state = get_state(graph)
    #         state_g = state.to(device)
    #         actions = policy_net(graph, state_g.view(graph.num_nodes(),-1),graph.edata['types'])
    actions = policy_net(graph)
    state_action_values[n_state] = actions.view(-1)[action_batch[n_state,0]]
  #     print("optim 0.3 %5.4f"%(time() -st))
  #     print("Training Nodes:", bg.num_nodes())
  st = time()
  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(BATCH_SIZE, device=device,dtype=torch.float32)
  #     x= torch.cat(batch.next_state)
  #     bg = dgl.batch(batch.graph)
  #     print("Training Nodes:", bg.num_nodes())
  #     print("Training Nodes:", bg.batch_num_nodes())
  #     actions = target_net(bg,x)
  #     print(actions.shape)
  #     tot = 0
  #     for n,nb in enumerate(bg.batch_num_nodes()):
  #         next_state_values[n] = actions[tot:tot+nb,:].view(-1)[action_batch[n,0]]
  #         tot+=nb
  #     next_state_values =  actions[action_batch[:,0]]
  #     print(next_state_values.shape)

  for n_state, state in enumerate(batch.next_state):
    if state is not None:
      graph = batch.graph[n_state]
      state_g = state.to(device)
      next_state_values[n_state] = target_net(graph, state_g.view(graph.num_nodes(),-1)).max().detach()
  #             next_state_values[n_state] = policy_net(graph, state_g.view(graph.num_nodes(),-1)).max().detach()
  # Compute the expected Q values
  #     print("optim 0.4 %5.4f"%(time() -st))
  st = time()
  #     print(BATCH_SIZE, state_action_values.shape, next_state_values.shape)
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))

  # Optimize the model
  optimizer.zero_grad()
  #     print("optim 0.5 %5.4f"%(time() -st))
  st = time()
  loss.backward()
  loss_history.append(loss.item())
  #     print("optim 0.6 %5.4f"%(time() -st))
  st = time()
  optimizer.step()
  #     print("optim %5.4f"%(time() -st))
  st = time()
  return optimizer, loss_history

def select_action(graph, inference = False, total_taken = False,\
                  steps_done = False, random_taken = False, policy_net = False,\
                  EPS_END = False, EPS_START = False, EPS_DECAY = False, device = False):
  #global steps_done
  #global total_taken, random_taken
  total_taken +=1
  if inference:
    with torch.no_grad():
      #             mask = torch.cat((torch.ones((G.num_nodes(),1)),torch.zeros((G.num_nodes(),1))),dim=1).long()
                  # mask of 0,1,0,1,0, ... to mask downsizes
      #             mask = mask.to(device)
      #             state_g = state.to(device)
      #             action = policy_net(graph, state_g,graph.edata['types'])
      action = policy_net(graph)
      #             action = (action-action.min()+1)*(mask)
      return torch.argmax(action.view(-1)).view(1,1), total_taken, steps_done, random_taken

  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * 0.95**(steps_done / EPS_DECAY)
  steps_done += 1
  # get the mask
  mask = generate_masked_actions(graph)

  if int(sum(~mask.view(-1)))==0 :
    return -1, total_taken, steps_done, random_taken
  # Threshold keeps decreasing, so over time it takes more from the policy net.
  if sample > eps_threshold:
    with torch.no_grad():
      #state_g = state.to(device)
      #action = policy_net(graph, state_g,graph.edata['types'])
      action = policy_net(graph)
      return torch.argmax(action.view(-1)).view(1,1), total_taken, steps_done, random_taken
  else:
    action = torch.randn_like(mask,dtype=torch.float32)
    action = (action-action.min()+1)*(~mask)
    random_taken+=1
    return torch.tensor([[torch.argmax(action.view(-1))]], device=device, dtype=torch.long),\
            total_taken, steps_done, random_taken

def get_subgraph(graph, nodes):
  node_set = {x.item() for x in nodes}
  #level 1
  in_nodes, _ = graph.in_edges(list(node_set))
  _, out_nodes = graph.out_edges(list(node_set))
  node_set.update(in_nodes.tolist())
  node_set.update(out_nodes.tolist())
  # level 2
  in_nodes, _ = graph.in_edges(list(node_set))
  _, out_nodes = graph.out_edges(list(node_set))
  node_set.update(in_nodes.tolist())
  node_set.update(out_nodes.tolist())
  #     # level 3
  #     in_nodes, _ = graph.in_edges(list(node_set))
  #     _, out_nodes = graph.out_edges(list(node_set))
  #     node_set.update(in_nodes.tolist())
  #     node_set.update(out_nodes.tolist())

  subgraph = dgl.node_subgraph(graph, list(node_set))

  return subgraph

def get_critical_path_nodes(graph, ep_num, TOP_N_NODES, n_cells):
  #     critical_path = (graph.ndata['slack'] <= graph.ndata['slack'].min() +1e-5)
  topk = min(len(graph.ndata['slack'])-1 , int(TOP_N_NODES*(1+0.01*ep_num)))
  min_slacks, critical_path = torch.topk(graph.ndata['slack'], topk, largest=False)
  #min_slacks, critical_path = torch.topk(graph.ndata['slack'], int(TOP_N_NODES*(1+0.02*ep_num)), largest=False)
  critical_path = critical_path[min_slacks<0]

  #     critical_path = (graph.ndata['slack'] < 0)
  #     critical_path = (critical_path.nonzero()).view(-1)
  if critical_path.numel() <=0:
    critical_path = torch.arange(0,graph.num_nodes())

  return critical_path

def get_state(graph, n_state, n_cells, n_features):
  state = torch.zeros(graph.num_nodes(), n_state)
  state[:,-1] = graph.ndata['area']
  state[:,-2] = graph.ndata['slack']
  state[:,-3] = graph.ndata['slew']
  state[:,-4] = graph.ndata['load']
  #     print(F.one_hot(graph.ndata['cell_types'][:,0],n_cells).shape, graph.ndata['cell_types'][:,1].shape,n_features)
  state[:,:-n_features] =F.one_hot(graph.ndata['cell_types'][:,0],n_cells)*graph.ndata['cell_types'][:,1:2]
  return state

def env_step(episode_G, graph, state, action, CLKset, ord_design, timing,\
            cell_dict, norm_data, inst_names, episode_inst_dict, inst_dict,\
            n_cells, n_features, block, device, Slack_Lambda, eps):
  #global load_mask,slack_mask,delay_mask
  #next_state = copy.deepcopy(state)
  next_state = state.clone()
  reward = 0
  done =0
  # based on the selected action you choose the approriate cell and upsize it or downsize
  cell_sub = int(action/2)
  cell = graph.ndata['_ID'][cell_sub].item()
  inst_name = inst_names[cell]
  cell_size = episode_inst_dict[inst_name]['cell_type'][1]
  cell_idx = episode_inst_dict[inst_name]['cell_type'][0]
  old_slack = min_slack(inst_name, cell_dict, inst_dict, ord_design, timing)
  o_master_name = cell_dict[str(cell_idx)]['name']+\
                  cell_dict[str(cell_idx)]['sizes'][cell_size]
  #     print("Master name old:", o_master_name, inst_name)

  #     print('cell %d %s',cell,inst_name)
  #     print('m sz ',G.ndata['max_size'][cell])
  #     print('sz 1 ',cell_size)
  if(action%2 == 0):
      cell_size +=1
  else:
      cell_size -=1
  #     print('sz 2 ',cell_size)
  if(cell_size>=cell_dict[str(cell_idx)]['n_sizes']):
    print("Above max")
    print(action,cell_dict[str(cell_idx)]['n_sizes'], cell_idx, cell_size)
  if(cell_size<0):
    print("below min")
    print(action,cell_dict[str(cell_idx)]['n_sizes'], cell_idx, cell_size)
  episode_inst_dict[inst_name]['cell_type'] = (cell_idx,cell_size)
  size = cell_dict[str(cell_idx)]['sizesi'][cell_size] #actual size

  # one hot encode the relavant feature with the magnitude of size.
  next_state[cell_sub,:-n_features] = F.one_hot(torch.tensor([cell_idx]),n_cells)*size
  episode_G.ndata['cell_types'][cell] = torch.tensor((cell_idx,cell_size))

  # replace the master node in the code and find the new slack,
  inst = block.findInst(inst_name)
  n_master_name = cell_dict[str(cell_idx)]['name']+\
                  cell_dict[str(cell_idx)]['sizes'][cell_size]
  #     print("Master name new:", n_master_name, inst_name )
  db = ord.get_db()
  n_master = db.findMaster(n_master_name)
  inst.swapMaster(n_master)
  new_slack = min_slack(inst_name, cell_dict, inst_dict, ord_design, timing)

  old_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
  episode_inst_dict[inst_name]['area']= n_master.getWidth() * n_master.getHeight()
  new_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
  episode_G.ndata['area'][cell] = new_area

  # update_area
  next_state[cell_sub,-1] = new_area
  #reward += AREA_COEFF * torch.tensor(old_area-new_area)
  reward += torch.tensor(old_area-new_area)
  old_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slews = torch.zeros(len(episode_inst_dict.keys()))
  new_loads = torch.zeros(len(episode_inst_dict.keys()))

  #     for n,inst in enumerate(episode_inst_dict.keys()):
  for n, inst in inst_names.items():
    old_slacks[n] = episode_inst_dict[inst]['slack']
    
    tmp_sta_pin = ord_design.getStaPin(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    tmp_db_pin = timing.staToDBPin_ITerm(tmp_sta_pin)
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
      episode_inst_dict[inst]['slew'],
      episode_inst_dict[inst]['load']) = pin_properties(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'], CLKset, ord_design, timing)
      new_slacks[n] = episode_inst_dict[inst]['slack']
      new_slews[n] = episode_inst_dict[inst]['slew']
      new_loads[n] = episode_inst_dict[inst]['load']
  episode_G.ndata['slack'] = new_slacks.to(device)/norm_data['clk_period']
  for i in range(len(episode_G.ndata['slack'])):
    if episode_G.ndata['slack'][i] > 1:
      episode_G.ndata['slack'][i] = 1
  episode_G.ndata['slack'][torch.isinf(episode_G.ndata['slack'])] = 1
  episode_G.ndata['slew'] = new_slews.to(device)/ norm_data['max_slew']
  episode_G.ndata['load'] = new_loads.to(device)/ norm_data['max_load']

  next_state[:,-2] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['slack']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['clk_period']
  next_state[:,-3] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['slew']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['max_slew']
  next_state[:,-4] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['load']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['max_load']
  next_state[torch.isinf(next_state[:,-2]),-2] = 1 # remove infinity from primary outputs. hadle it better.

  #Check TNS
  new_TNS = torch.min(new_slacks,torch.zeros_like(new_slacks))
  new_TNS = Slack_Lambda*new_TNS

  old_TNS = torch.min(old_slacks,torch.zeros_like(old_slacks))
  old_TNS = Slack_Lambda*old_TNS

  factor = torch.max(torch.abs(0.1*old_TNS), eps*torch.ones_like(old_TNS))
  factor = torch.max(torch.ones_like(old_TNS), 1/factor)
  reward += (torch.sum((new_TNS - old_TNS) * factor)).item()

  #     reward += (torch.sum((new_TNS - old_TNS).to(device) * Slack_Lambda)).item()

  return reward, done, next_state, episode_inst_dict, episode_G

def env_reset(reset_state = None, episode_num = None, cell_name_dict = None,\
              CLKset = None, ord_design = None, timing = None, G = None,\
              inst_dict = None, CLK_DECAY = None, CLK_DECAY_STRT = None,\
              clk_init = None, clk_range = None, clk_final = None, inst_names = None,\
              block = None, cell_dict = None, norm_data = None, device = None):
  episode_G = copy.deepcopy(G)
  episode_inst_dict = copy.deepcopy(inst_dict)

  if episode_num is not None:

    if episode_num<CLK_DECAY_STRT:
      clk = clk_init
    elif episode_num<CLK_DECAY+CLK_DECAY_STRT:
      clk = clk_init - clk_range*(episode_num -CLK_DECAY_STRT) /CLK_DECAY
    else:
      clk = clk_final
    ord_design.evalTclString("create_clock [get_ports i_clk] -name core_clock -period " + str(clk*1e-9))
    #ord.create_clock("core_clk", ["clk"], clk*1e-9)

  for i in range(len(inst_names)):
    inst_name = inst_names[i]
    inst = block.findInst(inst_name)
    if reset_state is not None:
      o_master_name = reset_state[i]
      cell_idx, cell_size = get_type(o_master_name, cell_dict, cell_name_dict)
      episode_inst_dict[inst_name]['cell_type'] = (cell_idx,cell_size)
      episode_G.ndata['cell_types'][i] = torch.tensor((cell_idx,cell_size))
    else:
      cell_size = episode_G.ndata['cell_types'][i,1].item()
      cell_idx = episode_G.ndata['cell_types'][i,0].item()
      o_master_name = cell_dict[str(cell_idx)]['name']+\
              cell_dict[str(cell_idx)]['sizes'][cell_size]

    db = ord.get_db()
    o_master = db.findMaster(o_master_name)
    if o_master_name != inst.getMaster().getName():
      inst.swapMaster(o_master)
    if reset_state is not None:
      episode_inst_dict[inst_name]['area']= o_master.getWidth() * o_master.getHeight()
      new_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
      episode_G.ndata['area'][i] = new_area

  #     if reset_state is not None:
  new_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slews = torch.zeros(len(episode_inst_dict.keys()))
  new_loads = torch.zeros(len(episode_inst_dict.keys()))

  for n, inst in inst_names.items():
    tmp_sta_pin = ord_design.getStaPin(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    tmp_db_pin = timing.staToDBPin_ITerm(tmp_sta_pin)
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
      episode_inst_dict[inst]['slew'],
      episode_inst_dict[inst]['load']) = pin_properties(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'], CLKset, ord_design, timing)
      new_slacks[n] = episode_inst_dict[inst]['slack']
      new_slews[n] = episode_inst_dict[inst]['slew']
      new_loads[n] = episode_inst_dict[inst]['load']
  episode_G.ndata['slack'] = new_slacks.to(device)/norm_data['clk_period']
  for i in range(len(episode_G.ndata['slack'])):
    if episode_G.ndata['slack'][i] > 1:
      episode_G.ndata['slack'][i] = 1
  episode_G.ndata['slack'][torch.isinf(episode_G.ndata['slack'])] = 1
  episode_G.ndata['slew'] = new_slews.to(device)/ norm_data['max_slew']
  episode_G.ndata['load'] = new_loads.to(device)/ norm_data['max_load']

  return episode_G, episode_inst_dict, clk


def calc_cost(ep_G, Slack_Lambda):
  cost = torch.sum(ep_G.ndata['area'].to('cpu'))
  x = ep_G.ndata['slack'].to('cpu')
  new_slacks = torch.min(x, torch.zeros_like(x))
  cost += torch.sum(Slack_Lambda*(-new_slacks))
  return cost

def get_state_cells(ep_dict, inst_names, cell_dict):
  cells = []
  for x in inst_names.values():
    cell_size = ep_dict[x]['cell_type'][1]
    cell_idx = ep_dict[x]['cell_type'][0]
    cell_name = cell_dict[str(cell_idx)]['name']+\
                cell_dict[str(cell_idx)]['sizes'][cell_size]
    cells.append(cell_name)
  return cells

def pareto(pareto_points, pareto_cells, area, clk, ep_dict, inst_names,\
            cell_dict, inst_dict, ord_design, timing):
  cells = get_state_cells(ep_dict, inst_names, cell_dict)
  if len(pareto_points) <= 0:
    pareto_points.append((area, clk))
    pareto_cells.append(cells)
    return 1
  dominated_points = set()
  for n, pareto_point in enumerate(pareto_points):
    # if new point is dominated we skip
    if pareto_point[0] <= area and  pareto_point[1] <= clk:
      return 0
    # if new point dominates any other point
    elif pareto_point[0] >= area and  pareto_point[1] >= clk:
      dominated_points.add(n)

  print("updating point")
  print((area, clk))
  print("dominated points: ", [pareto_points[n] for n in dominated_points])
  pareto_points.append((area, clk))
  pareto_cells.append(cells)
  pareto_points = [val for n, val in enumerate(pareto_points) if n not in dominated_points]
  pareto_cells = [val for n, val in enumerate(pareto_cells) if n not in dominated_points]
  print("new pareto points: ",pareto_points)
  slacks = [min_slack(x, cell_dict, inst_dict, ord_design, timing) for x in inst_names.values()]
  test_sl = np.min(slacks)
  print(test_sl)
  return 1

def rmdir(directory):
  directory=Path(directory)
  for  item in directory.iterdir():
    if item.is_dir():
      rmdir(directory)
    else:
      item.unlink()
  directory.rmdir()







