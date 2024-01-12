import openroad as ord
from openroad import Tech, Design
import os, odb, drt

def load_design(demo_path, circuitops_path):
  #Read Files
  tech = Tech()
  for lib_file in [os.path.join(root, file) for root, _, files in os.walk(circuitops_path + "/platforms/nangate45/lib/") for file in files if file.endswith(".lib")]:
    tech.readLiberty(lib_file)
  for techFile in [os.path.join(root, file) for root, _, files in os.walk(circuitops_path + "/platforms/nangate45/lef/") for file in files if file.endswith("tech.lef")] :
    tech.readLef(techFile)
  for lefFile in [os.path.join(root, file) for root, _, files in os.walk(circuitops_path + "/platforms/nangate45/lef/") for file in files if file.endswith(".lef")]:
    tech.readLef(lefFile)
  design = Design(tech)
  #design.readVerilog(demo_path + "/gcd.v")
  design.readDef(demo_path + "/data/gcd.def")
  #design.link("gcd")
  design.evalTclString("read_sdc "+ circuitops_path + "/designs/nangate45/gcd/6_final.sdc.gz")

  design.evalTclString("create_clock -period 20 [get_ports clk] -name core_clock")
  
  design.evalTclString("set_propagated_clock [all_clocks]")
  
  VDD_net = design.getBlock().findNet("VDD")
  if VDD_net is None:
    VDD_net = odb.dbNet_create(design.getBlock(), "VDD")
  VDD_net.setSpecial()
  VDD_net.setSigType("POWER")
  design.getBlock().addGlobalConnect(None, ".*", "VDD", VDD_net, True)
  
  VSS_net = design.getBlock().findNet("VSS")
  if VSS_net is None:
    VSS_net = odb.dbNet_create(design.getBlock(), "VSS")
  VSS_net.setSpecial()
  VSS_net.setSigType("GROUND")
  design.getBlock().addGlobalConnect(None, ".*", "VDD", VSS_net, True)
  
  design.getBlock().globalConnect()
  return tech, design


