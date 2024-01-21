#BSD 3-Clause License
#
#Copyright (c) 2023, ASU-VDA-Lab
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import openroad as ord
from openroad import Tech, Design
import os, odb, drt
from pathlib import Path
from demo1_helpers import load_design
import argparse 

############
#Read Files#
############

parser = argparse.ArgumentParser(description="Path to root of the tutorial directory")
parser.add_argument("--path", type = Path, default='./', action = 'store')
pyargs = parser.parse_args()
tech, design = load_design(pyargs.path, verilog = True) 
# tech, design = load_design(Path("../", verilog = True) # For demo to be copied.

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

####################
#Detailed Placement#
####################
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0) / site.getWidth())
max_disp_y = int(design.micronToDBU(0) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)

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

##################
#Detailed Routing#
##################
drter = design.getTritonRoute()
params = drt.ParamStruct()
params.outputMazeFile = ""
params.outputDrcFile = ""
params.outputCmapFile = ""
params.outputGuideCoverageFile = ""
params.dbProcessNode = ""
params.enableViaGen = True
params.drouteEndIter = 1
params.viaInPinBottomLayer = ""
params.viaInPinTopLayer = ""
params.orSeed = -1
params.orK = 0
params.bottomRoutingLayer = "metal1"
params.topRoutingLayer = "metal8"
params.verbose = 1
params.cleanPatches = True
params.doPa = True
params.singleStepDR = False
params.minAccessPoints = 1
params.saveGuideUpdates = False
drter.setParams(params)
drter.main()

design.writeDef("tmp.def")

