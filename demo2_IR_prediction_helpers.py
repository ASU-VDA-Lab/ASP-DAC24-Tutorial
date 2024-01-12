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

def OpenROAD_map_creation(map_type, tech_design, design, corner, congestion_layer, timing):
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
  #row_height = 20
  #################
  #get track width#
  #################
  track_width = block.getTrackGrids()[0].getGridX()[1] - block.getTrackGrids()[0].getGridX()[0]
  #track_width = 20
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
  ###############################
  #assign congestion and IR drop#
  ###############################
  if map_type == "static_IR" or map_type == "congestion":
    db_tech = tech_design.getDB().getTech()
    if map_type == "static_IR":
      feature_map = np.full((image_width, image_height), 0.0)
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
      return row_height, track_width, feature_map
    else:
      feature_map = np.full((image_width, image_height), -np.inf)
      layers = db_tech.getLayers()
      layer = layers[congestion_layer]
      min_ = np.inf
      for x in range(len(gcell_grid_x)):
        for y in range(len(gcell_grid_y)):
          capacity = block.getGCellGrid().getHorizontalCapacity(layer, x, y)
          usage = block.getGCellGrid().getHorizontalUsage(layer, x, y)
          if block.getGCellGrid().getHorizontalCapacity(layer, x, y) == 0:
            capacity = block.getGCellGrid().getVerticalCapacity(layer, x, y)
            usage = block.getGCellGrid().getVerticalUsage(layer, x, y)
          congestion = usage - capacity
          min_ = min([min_, congestion])
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
      for x in range(len(feature_map)):
        for y in range(len(feature_map[0])):
          if feature_map[x][y] == -np.inf:
            feature_map[x][y] = min_
      return row_height, track_width, feature_map
  ################################################################################
  #assign static_power and dynamic_power value by iterating through each instance#
  ################################################################################
  if map_type == "static_power" or map_type == "dynamic_power":
    feature_map = np.full((image_width, image_height), 0.0)
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
        feature = timing.staticPower(inst, corner)
        feature /= ((inst_x1 - inst_x0) * (inst_y1 - inst_y0))
      elif map_type == "dynamic_power":
        feature = timing.dynamicPower(inst, corner)
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
  def __init__(self, path):
    ### SET DESIGN ###
    self.DESIGN_NAME = "gcd"
    #self.DESIGN_NAME = "aes"
    #self.DESIGN_NAME = "bp_fe"
    #self.DESIGN_NAME = "bp_be"

    ### SET PLATFORM ###
    self.PLATFORM = "nangate45"

    ### INTERNAL DEFINTIONS: DO NOT MODIFY BELOW ####
    self.CIRCUIT_OPS_DIR = path
    self.DESIGN_DIR = self.CIRCUIT_OPS_DIR + "/designs/" + self.PLATFORM + "/" + self.DESIGN_NAME
    self.PLATFORM_DIR = self.CIRCUIT_OPS_DIR + "/platforms/" + self.PLATFORM

    self.DEF_FILE = self.DESIGN_DIR + "/6_final.def.gz"
    self.TECH_LEF_FILE = [os.path.join(root, file) for root, _, files in os.walk(self.PLATFORM_DIR + "/lef/") for file in files if file.endswith("tech.lef")]
    self.LEF_FILES = [os.path.join(root, file) for root, _, files in os.walk(self.PLATFORM_DIR + "/lef/") for file in files if file.endswith(".lef")]
    self.LIB_FILES = [os.path.join(root, file) for root, _, files in os.walk(self.PLATFORM_DIR + "/lib/") for file in files if file.endswith(".lib")]
    self.SDC_FILE = self.DESIGN_DIR + "/6_final.sdc.gz"
    self.NETLIST_FILE = self.DESIGN_DIR + "/6_final.v"
    self.SPEF_FILE = self.DESIGN_DIR + "/6_final.spef.gz"

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
  design.getGlobalRouter().globalRoute()
  return tech_design, design

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

