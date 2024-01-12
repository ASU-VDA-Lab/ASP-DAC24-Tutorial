import openroad as ord
from openroad import Tech, Design, Timing
import os, odb, drt
from demo1_2_helpers import load_design

path_CircuitOps = "../CircuitOps/"
path_Demo = "./"
tech, design = load_design(path_Demo, path_CircuitOps)

timing = Timing(design)
corner = timing.getCorners()[0]
block = design.getBlock()

############
#cell query#
############
print("name       library_type     dynamic_power     static_power")
insts = block.getInsts()[-10:]
for inst in insts:
  inst_static_power = timing.staticPower(inst, corner)
  inst_dynamic_power = timing.dynamicPower(inst, corner)
  inst_name = inst.getName()
  libcell_name = inst.getMaster().getName()
  print("{} {} {} {}".format(inst_name, libcell_name, \
    inst_dynamic_power, inst_static_power))
  #hit tab for all available apis (ex. inst.[tab])
  #the return type of power is float!!!
print("return type: {}".format(type(inst_static_power)))
print("###################################################################")
###########
#net query#
###########
print("name   net_type   pin&wire_capacitance")
nets = block.getNets()[:10]
for net in nets:
  pin_and_wire_cap = timing.getNetCap(net, corner, timing.Max)
  net_name = net.getName()
  net_type = net.getSigType()
  print("{} {} {}".format(net_name, net_type, pin_and_wire_cap))
  #hit tab for all available apis (ex. net.[tab])
  #the return type of pin_and_wire_cap is float!!!
print("###################################################################")
###########
#pin query#
###########
print("name rise_arrival_time fall_arrival_time rise_slack fall_slack slew")
for inst in insts:
  inst_ITerms = inst.getITerms()
  for pin in inst_ITerms:
    pin_name = design.getITermName(pin)
    pin_rise_arr = timing.getPinArrival(pin, timing.Rise)
    pin_fall_arr = timing.getPinArrival(pin, timing.Fall)
    pin_rise_slack = timing.getPinSlack(pin, timing.Fall, timing.Max)
    pin_fall_slack = timing.getPinSlack(pin, timing.Rise, timing.Max)
    pin_slew = timing.getPinSlew(pin)
    print("{} {} {} {} {} {}".format( pin_name, pin_rise_arr, \
      pin_fall_arr, pin_rise_slack, pin_fall_slack, pin_slew \
      ))
    #hit tab for all available apis (ex. pin.[tab])
    #the return type of slack is float!!!
    #timing-related properties go through timing.[tab] apis
print("###################################################################")
print("name rise_arrival_time fall_arrival_time rise_slack fall_slack slew")
for net in nets:
  net_ITerms = net.getITerms()
  for pin in net_ITerms:
    pin_name = design.getITermName(pin)
    pin_rise_arr = timing.getPinArrival(pin, timing.Rise)
    pin_fall_arr = timing.getPinArrival(pin, timing.Fall)
    pin_rise_slack = timing.getPinSlack(pin, timing.Fall, timing.Max)
    pin_fall_slack = timing.getPinSlack(pin, timing.Rise, timing.Max)
    pin_slew = timing.getPinSlew(pin)    
    print("{} {} {} {} {} {}".format( pin_name, pin_rise_arr, \
      pin_fall_arr, pin_rise_slack, pin_fall_slack, pin_slew \
      ))

