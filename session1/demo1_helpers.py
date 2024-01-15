import openroad as ord
from openroad import Tech, Design
import os, odb, drt
from pathlib import Path

def load_design(demo_path, verilog = False):
  #Read Files
  tech = Tech()
  libDir = demo_path/"platforms/lib/"
  lefDir = demo_path/"platforms/lef/"
  designDir = demo_path/"designs/"
  # Read technology files
  libFiles = libDir.glob('*.lib')
  techLefFiles = lefDir.glob('*tech.lef')
  lefFiles = lefDir.glob('*.lef')
  for libFile in libFiles:
    tech.readLiberty(libFile.as_posix())
  for techLefFile in techLefFiles:
    tech.readLef(techLefFile.as_posix())
  for lefFile in lefFiles:
    tech.readLef(lefFile.as_posix())
  design = Design(tech)
  #Read design files

  if verilog:
    verilogFile = designDir/"gcd.v"
    design.readVerilog(f"{verilogFile}")
    design.link("gcd")
  else:
    defFile = designDir/'gcd.def'
    design.readDef(f"{defFile}")
  
  # Read the SDC file and set the clocks
  sdcFile = designDir/"gcd.sdc.gz" 
  design.evalTclString(f"read_sdc {sdcFile}")
  design.evalTclString("create_clock -period 20 [get_ports clk] -name core_clock")
  design.evalTclString("set_propagated_clock [all_clocks]")
  
  return tech, design


