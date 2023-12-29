import torch
import torch.nn as nn
import numpy as np
from ASPDAC_helpers import OpenROAD_map_creation, CircuitOps_File_DIR, load_design
from ASPDAC_helpers import handle_size, UNet

if __name__ == "__main__":
  #############
  #load design#
  #############
  _CircuitOps_File_DIR = CircuitOps_File_DIR()
  tech_design, design = load_design(_CircuitOps_File_DIR)  
  corner = design.getCorners()[0]
  ############
  #load model#
  ############
  model = UNet()
  #################
  #get feature map#
  #################
  row_height, track_width, static_power_map = OpenROAD_map_creation("static_power", tech_design, design, corner, -1)
  row_height, track_width, dynamic_power_map = OpenROAD_map_creation("dynamic_power", tech_design, design, corner, -1)
  row_height, track_width, m1_congestion_map = OpenROAD_map_creation("congestion", tech_design, design, corner, 2)
  row_height, track_width, m2_congestion_map = OpenROAD_map_creation("congestion", tech_design, design, corner, 3)
  row_height, track_width, m3_congestion_map = OpenROAD_map_creation("congestion", tech_design, design, corner, 4)
  ################
  #get golden map#
  ################
  row_height, track_width, static_IR_map = OpenROAD_map_creation("static_IR", tech_design, design, corner, -1)        

  #handle size by padding or cropping
  static_power_map = handle_size(static_power_map, (256, 256))
  dynamic_power_map = handle_size(dynamic_power_map, (256, 256))
  m1_congestion_map = handle_size(m1_congestion_map, (256, 256))
  m2_congestion_map = handle_size(m2_congestion_map, (256, 256))
  m3_congestion_map = handle_size(m3_congestion_map, (256, 256))
  static_IR_map = handle_size(static_IR_map, (256, 256))

  for i in range(len(m3_congestion_map)):
    static_power_map[i] = torch.Tensor(static_power_map[i]).unsqueeze(0).unsqueeze(0)
    dynamic_power_map[i] = torch.Tensor(dynamic_power_map[i]).unsqueeze(0).unsqueeze(0)
    m1_congestion_map[i] = torch.Tensor(m1_congestion_map[i]).unsqueeze(0).unsqueeze(0)
    m2_congestion_map[i] = torch.Tensor(m2_congestion_map[i]).unsqueeze(0).unsqueeze(0)
    m3_congestion_map[i] = torch.Tensor(m3_congestion_map[i]).unsqueeze(0).unsqueeze(0)
    static_IR_map[i] = torch.Tensor(static_IR_map[i])

  output_tensor = model([static_power_map[0], dynamic_power_map[0], m1_congestion_map[0], m2_congestion_map[0], m3_congestion_map[0]]).squeeze()

  output_array = output_tensor.squeeze().detach().numpy()
  
  l1loss = nn.L1Loss()
  loss = l1loss(static_IR_map[0], output_tensor)
  print(loss)
  

print("#################Done#################")
