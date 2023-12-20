import torch
import torch.nn as nn
import numpy as np
from ASPDAC_helpers import OpenROAD_map_creation, CircuitOps_File_DIR, load_design

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
  
  
