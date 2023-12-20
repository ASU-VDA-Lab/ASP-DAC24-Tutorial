import openroad as ord
import pdn, odb, utl
from openroad import Tech, Design
from collections import defaultdict
import argparse
from ASPDAC_helpers import OpenROAD_map_creation, CircuitOps_File_DIR, load_design
import numpy as np
import math


if __name__ == "__main__":
  ############
  #parse flag#
  ############
  parser = argparse.ArgumentParser(description="types of 2-D data map of the design")
  parser.add_argument("-static_power", default=False, action = "store_true")
  parser.add_argument("-dynamic_power", default=False, action = "store_true")
  parser.add_argument("-static_IR", default=False, action = "store_true")
  parser.add_argument("-congestion", nargs="+", type=int)
  args = parser.parse_args()
  if args.congestion is None:
    congestion_list = np.array([])
  elif isinstance(args.congestion, list):  
    congestion_list = np.array(args.congestion)
  else:
    congestion_list = np.array([args.congestion])
  #############
  #load design#
  #############
  _CircuitOps_File_DIR = CircuitOps_File_DIR()
  tech_design, design = load_design(_CircuitOps_File_DIR)
  corner = design.getCorners()[0]
  #########################
  #create feature map dict#
  #########################
  data = defaultdict()

  if args.static_power:
    row_height, track_width, data["static_power"] = OpenROAD_map_creation("static_power", tech_design, design, corner, -1)
  if args.dynamic_power:
    row_height, track_width, data["dynamic_power"] = OpenROAD_map_creation("dynamic_power", tech_design, design, corner, -1)
  if args.static_IR:
    row_height, track_width, data["static_IR"] = OpenROAD_map_creation("static_IR", tech_design, design, corner, -1)  
  if len(congestion_list) >= 0:
    grt_obj = design.getGlobalRouter()
    grt_obj.globalRoute()
    data["congestion"] = defaultdict()
    for i in range(len(congestion_list)):
      row_height, track_width, data["congestion"][str(congestion_list[i])] = OpenROAD_map_creation("congestion", tech_design, design, corner, congestion_list[i])      



