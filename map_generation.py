import openroad as ord
import pdn, odb, utl
from openroad import Tech, Design
from collections import defaultdict
import argparse
from openroad_helpers import CircuitOps_File_DIR, load_design
import numpy as np
#from numba import jit

def OpenROAD_map_generation(map_type, design, block, corner, insts, nets):
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
  #################################################
  #assign value by iterating through each instance#
  #################################################
  for inst in insts:
    if map_type == "static_power":
      feature = design.staticPower(inst, corner)
    elif map_type == "dynamic_power":
      feature = design.dynamicPower(inst, corner)
    elif map_type == "static_IR":
      feature = -1
    ###############
    #get cell bbox#
    ###############
    inst_x0 = inst.getBBox().xMin()
    inst_y0 = inst.getBBox().yMin()
    inst_x1 = inst.getBBox().xMax()
    inst_y1 = inst.getBBox().yMax()

    anchor_index_x = (inst_x0 - core_x0) // track_width
    anchor_index_y = (inst_y0 - core_y0) // row_height
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
        pixel_area = row_height * track_width
        
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
        percentage = cover_area / pixel_area
        feature_map[anchor_index_x + x][anchor_index_y + y] += feature * percentage


  return feature_map

if __name__ == "__main__":
  ############
  #parse flag#
  ############
  parser = argparse.ArgumentParser(description="types of 2-D data map of the design")
  parser.add_argument("-static_power", default=False, action = "store_true")
  parser.add_argument("-dynamic_power", default=False, action = "store_true")
  parser.add_argument("-static_IR", default=False, action = "store_true")
  args = parser.parse_args()
  #############
  #load design#
  #############
  _CircuitOps_File_DIR = CircuitOps_File_DIR()
  design = load_design(_CircuitOps_File_DIR)
  block = ord.get_db_block()
  insts = block.getInsts()
  nets = block.getNets()
  corner = design.getCorners()[0]
  #########################
  #create feature map dict#
  #########################
  data = defaultdict()
  if args.static_power:
    data["static_power"] = OpenROAD_map_generation("static_power", design, block, corner, insts, nets)  
  if args.dynamic_power:
    data["dynamic_power"] = OpenROAD_map_generation("dynamic_power", design, block, corner, insts, nets)
  if args.static_IR:
    data["static_IR"] = OpenROAD_map_generation("static_IR", design, block, corner, insts, nets)
  





