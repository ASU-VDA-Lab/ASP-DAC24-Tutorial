import torch
import torch.nn as nn
import numpy as np
from demo2_IR_helpers import OpenROAD_map_creation, load_design
from demo2_IR_helpers import UNet
import argparse
from openroad import Tech, Design, Timing
from pathlib import Path

if __name__ == "__main__":
  #############
  #load design#
  #############
  parser = argparse.ArgumentParser(description="path of your CircuitOps clone (must include /CircuitOps)")
  parser.add_argument("--path", type = Path, default='./', action = 'store')
  pyargs = parser.parse_args()
  tech_design, design = load_design(pyargs.path)  
  
  timing = Timing(design)

  corner = timing.getCorners()[0]
  ############
  #load model#
  ############
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = UNet()
  #################
  #get feature map#
  #################
  row_height, track_width, static_power_map = OpenROAD_map_creation("static_power", tech_design, design, corner, -1, timing)
  row_height, track_width, dynamic_power_map = OpenROAD_map_creation("dynamic_power", tech_design, design, corner, -1, timing)
  row_height, track_width, m1_congestion_map = OpenROAD_map_creation("congestion", tech_design, design, corner, 2, timing)
  row_height, track_width, m2_congestion_map = OpenROAD_map_creation("congestion", tech_design, design, corner, 4, timing)
  row_height, track_width, m3_congestion_map = OpenROAD_map_creation("congestion", tech_design, design, corner, 6, timing)
  ################
  #get golden map#
  ################
  row_height, track_width, static_IR_map = OpenROAD_map_creation("static_IR", tech_design, design, corner, -1, timing)        

  static_power_map = torch.Tensor(static_power_map).unsqueeze(0).unsqueeze(0)
  dynamic_power_map = torch.Tensor(dynamic_power_map).unsqueeze(0).unsqueeze(0)
  m1_congestion_map = torch.Tensor(m1_congestion_map).unsqueeze(0).unsqueeze(0)
  m2_congestion_map = torch.Tensor(m2_congestion_map).unsqueeze(0).unsqueeze(0)
  m3_congestion_map = torch.Tensor(m3_congestion_map).unsqueeze(0).unsqueeze(0)
  static_IR_map = torch.Tensor(static_IR_map)

  output_tensor = model([static_power_map, dynamic_power_map, m1_congestion_map, m2_congestion_map, m3_congestion_map]).squeeze()

  output_array = output_tensor.squeeze().detach().numpy()
  
  l1loss = nn.L1Loss()
  loss = l1loss(static_IR_map, output_tensor)
  
  print(f"L1 Loss: {loss.item():7.5f}")
  

print("#################Done#################")
