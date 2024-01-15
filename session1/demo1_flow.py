import openroad as ord
from openroad import Tech, Design
import os, odb, drt
from pathlib import Path
from demo1_helpers import load_design

############
#Read Files#
############
tech, design = load_design(Path("./"), verilog = True)

###############
#Floorplanning#
###############
floorplan = design.getFloorplan()
die_area = odb.Rect(design.micronToDBU(0), design.micronToDBU(0), design.micronToDBU(45), design.micronToDBU(45))
core_area = odb.Rect(design.micronToDBU(5), design.micronToDBU(5), design.micronToDBU(40), design.micronToDBU(40))
floorplan.initFloorplan(die_area, core_area)
floorplan.makeTracks()

############
#Place Pins#
############
design.getIOPlacer().addHorLayer(design.getTech().getDB().getTech().findLayer("metal8"))
design.getIOPlacer().addVerLayer(design.getTech().getDB().getTech().findLayer("metal7"))
design.getIOPlacer().run(True)

################
#Power Planning#
################
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

##################
#Global Placement#
##################
gpl = design.getReplace()
gpl.setTimingDrivenMode(False)
gpl.setRoutabilityDrivenMode(False)
gpl.setUniformTargetDensityMode(False)
gpl.setInitialPlaceMaxIter(20)
gpl.setTargetDensity(0.7)
gpl.setInitDensityPenalityFactor(0.001)
gpl.doInitialPlace()

####################
#Detailed Placement#
####################
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0) / site.getWidth())
max_disp_y = int(design.micronToDBU(0) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)

######################
#Clock Tree Synthesis#
######################
design.evalTclString("set_propagated_clock [core_clock]")
design.evalTclString("set_wire_rc -clock -resistance 3.574e-02 -capacitance 7.516e-02")
design.evalTclString("set_wire_rc -signal -resistance 3.574e-02 -capacitance 7.516e-02")

cts = design.getTritonCts()
parms = cts.getParms()
parms.setWireSegmentUnit(20)
cts.setBufferList("CLKBUF_X3")
cts.setRootBuffer("CLKBUF_X3")
cts.setSinkBuffer("CLKBUF_X1")
cts.runTritonCts()

################
#Global Routing#
################
signal_low_layer = design.getTech().getDB().getTech().findLayer("metal1").getRoutingLevel()
signal_high_layer = design.getTech().getDB().getTech().findLayer("metal8").getRoutingLevel()
clk_low_layer = design.getTech().getDB().getTech().findLayer("metal3").getRoutingLevel()
clk_high_layer = design.getTech().getDB().getTech().findLayer("metal8").getRoutingLevel()
grt = design.getGlobalRouter()
grt.setMinRoutingLayer(signal_low_layer)
grt.setMaxRoutingLayer(signal_high_layer)
grt.setMinLayerForClock(clk_low_layer)
grt.setMaxLayerForClock(clk_high_layer)
grt.setAdjustment(0.5)
grt.setVerbose(True)
grt.globalRoute(True)

###########
#Write DEF#
###########
design.writeDef("./designs/demo1_flow.def")
