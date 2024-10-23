
# OpenROAD Python API Tutorial
This page shows example scripting of OpenROAD Python APIs.
# Building OpenROAD Binary
Clone OpenROAD
```
git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD.git
```
Build OpenROAD Binary
```
cd ./OpenROAD/
mkdir build
cd build
cmake ..
make -j
```

# Running Openroad in Python env
## Execute a Simple Script
```
./<openroadBinary> -python <fileNmae>.py
```
## Execute a Script with arguments
```
./<openroadBinary> -python <fileNmae>.py <arg1> <arg2>

```
## Using OpenROAD in a Python Shell
Import OpenROAD into Python env
```
import openroad
```
Read EDA Files using OpenROAD Python API
```
import openroad as ord
from openroad import Tech, Design, Timing
from pathlib import Path

# Must declare a Tech Object
tech = Tech()

libDir = Path("libDir")
lefDir = Path("lefDir")
techDir = Path("techDir")

libFiles = libDir.glob('*.lib')
lefFiles = lefDir.glob('*.lef')
techFiles = techDir.glob("tech.lefFile")

# Reading lib, lef and tech Files
for libFile in libFiles:
  tech.readLiberty(libFile.as_posix())
for techFile in techFiles:
  tech.readLef(techFile.as_posix())
for lefFile in lefFiles:
  tech.readLef(lefFile.as_posix())

# Must follow the hierarchy
design = Design(tech)
timing = Timing(design)

# You Can Only Pick Either a Verilog File Or a DEF File
# If Reading Verilog File
verilogFile = "verilogFile.v"
designName  = "designName"
design.readVerilog(verilogFile)
# Link the Top Module
design.link(designName)

# If Reading DEF File
defFile = "defFile.def"
design.readDef(defFile)

# Read SDC File
sdcFile = "sdcFile.sdc"
design.evalTclString("read_sdc %s"%sdcFile)
```
Write files
```
design.writeDef("final.def")
design.evalTclString("write_verilog %s.v"%"designName")
design.evalTclString("write_db .%s.odb"%"designName")
```
## Other necessary steps
Set Unit RC Value of Layers
```
design.evalTclString("set_layer_rc -layer M1 -resistance 1.3889e-01 -capacitance 1.1368e-01")
# M2 and the rest of the layers can follow the same method
```
Connect VDD/VSS pins to nets
```
import odb

# Find the VDD net
VDDNet = design.getBlock().findNet("VDD")
# Create VDD net if it does not exist
if VDDNet is None:
  VDDNet = odb.dbNet_create(design.getBlock(), "VDD")
# Raise the special flag of the VDD net
VDDNet.setSpecial()
# Assign the "VDD" net to the "POWER" type
VDDNet.setSigType("POWER")
# Find the VSS net
VSSNet = design.getBlock().findNet("VSS")
# Create VSS net if it does not exist
if VSSNet is None:
  VSSNet = odb.dbNet_create(design.getBlock(), "VSS")
# Raise the special flag of the VSS net
VSSNet.setSpecial()
# Assign the "VSS" net to the "GROUND" type
VSSNet.setSigType("GROUND")
# Connect the pins to the nets
design.getBlock().addGlobalConnect(None, ".*", "VDD", VDDNet, True)
design.getBlock().addGlobalConnect(None, ".*", "VSS", VSSNet, True)
# Establish global connect
design.getBlock().globalConnect()
```
Set the clock signal
```
# Create clock signal
design.evalTclString("create_clock -period "period in ps" [get_ports "portName"] -name "clockName"")
# Propagate the clock signal
design.evalTclString("set_propagated_clock [all_clocks]")
```
## Using OpenROAD Python APIs to Perform Physical Design Steps
Floorplan using utilization rate
```
import odb

# Get OpenROAD's Floorplanner
floorplan = design.getFloorplan()
# Set the floorplan utilization to 45%
floorplan_utilization = 45
# Set the aspect ratio of the design (height/width) as 1.5
floorplan_aspect_ratio = 1.5
# Set the spacing between core and die as 14 um
floorplan_core_spacing = [design.micronToDBU(14) for i in range(4)]
# Find the site name in lef
site = floorplan.findSite("siteName")
floorplan.initFloorplan(floorplan_utilization, floorplan_aspect_ratio,
floorplan_core_spacing[0], floorplan_core_spacing[1],
floorplan_core_spacing[2], floorplan_core_spacing[3], site)
# Create Tracks
floorplan.makeTracks()
```
Floorplan using manually set area
```
import odb

# Get OpenROAD's Floorplanner
floorplan = design.getFloorplan()
# Set the core and die area
# The four args are bottom-left x, bottom-left y, top-right x and top-right y
die_area = odb.Rect(design.micronToDBU(0), design.micronToDBU(0), design.micronToDBU(40), design.micronToDBU(60))
core_area = odb.Rect(design.micronToDBU(10), design.micronToDBU(10), design.micronToDBU(30), design.micronToDBU(50))
# Find the site in lef
site = floorplan.findSite("site_name")
floorplan.initFloorplan(die_area, core_area, site)
# Create Tracks
floorplan.makeTracks()
```
Place IO pins
```
params = design.getIOPlacer().getParameters()
params.setRandSeed(42)
params.setMinDistanceInTracks(False)
params.setMinDistance(design.micronToDBU(0))
params.setCornerAvoidance(design.micronToDBU(0))
# Place the pins on M8 and M9
design.getIOPlacer().addHorLayer(design.getTech().getDB().getTech().findLayer("M8"))
design.getIOPlacer().addVerLayer(design.getTech().getDB().getTech().findLayer("M9"))
IOPlacer_random_mode = True
design.getIOPlacer().run(IOPlacer_random_mode)
```
Global Placement
```
gpl = design.getReplace()
gpl.setTimingDrivenMode(False)
gpl.setRoutabilityDrivenMode(True)
gpl.setUniformTargetDensityMode(True)
# Set the max iteration of global placement to 30 times
gpl.setInitialPlaceMaxIter(30)
gpl.setInitDensityPenalityFactor(0.05)
gpl.doInitialPlace()
gpl.doNesterovPlace()
gpl.reset()
```
Macro Placement
```
macros = [inst for inst in ord.get_db_block().getInsts() if inst.getMaster().isBlock()] 
if len(macros) > 0:
  mpl = design.getMacroPlacer()
  # Set the halo around macros to 5 microns
  mpl_halo_x, mpl_halo_y = 5, 5
  mpl.setHalo(mpl_halo_x, mpl_halo_y)
  # Set the channel width between macros to 5 microns
  mpl_channel_x, mpl_channel_y = 5, 5
  mpl.setChannel(mpl_channel_x, mpl_channel_y)
  # Set the fence region as a user defined area in microns
  design.getMacroPlacer().setFenceRegion(32, 55, 32, 60)
  # Snap the macro to layer M4 (usually M4)
  layer = design.getTech().getDB().getTech().findLayer("M4")
  mpl.setSnapLayer(layer)
  mpl.placeMacrosCornerMaxWl()
```
Detailed Placement
```
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0.5) / site.getWidth())
max_disp_y = int(design.micronToDBU(1) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)
```
Clock Tree Synthesis
```
design.evalTclString("set_propagated_clock [core_clock]")
design.evalTclString("set_wire_rc -clock -resistance 0.0435 -capacitance 0.0817")
design.evalTclString("set_wire_rc -signal -resistance 0.0435 -capacitance 0.0817")

cts = design.getTritonCts()
parms = cts.getParms()
parms.setWireSegmentUnit(20)
# Can choose different buffer cells for cts
cts.setBufferList("BUF_X3")
cts.setRootBuffer("BUF_X3")
cts.setSinkBuffer("BUF_X3")
cts.runTritonCts()
# Followed by detailed placement to legalize the clock buffers
site = design.getBlock().getRows()[0].getSite()
max_disp_x = int(design.micronToDBU(0.5) / site.getWidth())
max_disp_y = int(design.micronToDBU(1) / site.getHeight())
design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)
```
Add Filler Cells
```
db = ord.get_db()
filler_masters = list()
# Filler cell prefix may be different when using different library
filler_cells_prefix = "filler*"
for lib in db.getLibs():
  for master in lib.getMasters():
    master_name = master.getConstName()
    if re.fullmatch(filler_cells_prefix, master_name) != None:
      filler_masters.append(master)
if len(filler_masters) == 0:
  print("wrong filler cell prefix")
else:
  design.getOpendp().fillerPlacement(filler_masters, filler_cells_prefix)
```
Power Planning
```
import pdn, odb

# Global Connect
for net in design.getBlock().getNets():
  if net.getSigType() == "POWER" or net.getSigType() == "GROUND":
    net.setSpecial()
VDD_net = design.getBlock().findNet("VDD")
VSS_net = design.getBlock().findNet("VSS")
switched_power = None
secondary = list()
if VDD_net == None:
  VDD_net = odb.dbNet_create(design.getBlock(), "VDD")
  VDD_net.setSpecial()
  VDD_net.setSigType("POWER")
if VSS_net == None:
  VSS_net = odb.dbNet_create(design.getBlock(), "VSS")
  VSS_net.setSpecial()
  VSS_net.setSigType("GROUND")
design.getBlock().addGlobalConnect(region = None, instPattern = ".*", 
                                  pinPattern = "^VDD$", net = VDD_net, 
                                  do_connect = True)
design.getBlock().addGlobalConnect(region = None, instPattern = ".*",
                                  pinPattern = "^VDDPE$", net = VDD_net,
                                  do_connect = True)
design.getBlock().addGlobalConnect(region = None, instPattern = ".*",
                                  pinPattern = "^VDDCE$", net = VDD_net,
                                  do_connect = True)
design.getBlock().addGlobalConnect(region = None, instPattern = ".*",
                                  pinPattern = "^VSS$", net = VSS_net, 
                                  do_connect = True)
design.getBlock().addGlobalConnect(region = None, instPattern = ".*",
                                  pinPattern = "^VSSE$", net = VSS_net,
                                  do_connect = True)
design.getBlock().globalConnect()
# Voltage Domains
pdngen = design.getPdnGen()
pdngen.setCoreDomain(power = VDD_net, switched_power = switched_power, 
                    ground = VSS_net, secondary = secondary)
# Set the width of the PDN ring and the spacing between VDD and VSS rings
core_ring_width = [design.micronToDBU(5), design.micronToDBU(5)]
core_ring_spacing = [design.micronToDBU(5), design.micronToDBU(5)]
core_ring_core_offset = [design.micronToDBU(0) for i in range(4)]
core_ring_pad_offset = [design.micronToDBU(0) for i in range(4)]
# When the two layers are parallel, specify the distance between via cuts.
pdn_cut_pitch = [design.micronToDBU(2) for i in range(2)]

ring_connect_to_pad_layers = list()
for layer in design.getTech().getDB().getTech().getLayers():
  if layer.getType() == "ROUTING":
    ring_connect_to_pad_layers.append(layer)

# Define power grid for core
domains = [pdngen.findDomain("Core")]
halo = [design.micronToDBU(0) for i in range(4)]
for domain in domains:
  pdngen.makeCoreGrid(domain = domain, name = "top_pdn", starts_with = pdn.GROUND, 
                      pin_layers = [], generate_obstructions = [], powercell = None,
                      powercontrol = None, powercontrolnetwork = "STAR")
m1 = design.getTech().getDB().getTech().findLayer("M1")
m4 = design.getTech().getDB().getTech().findLayer("M4")
m7 = design.getTech().getDB().getTech().findLayer("M7")
m8 = design.getTech().getDB().getTech().findLayer("M8")
grid = pdngen.findGrid("top_pdn")
for g in grid:
  # Make Ring for the core
  pdngen.makeRing(grid = g, layer0 = m7, width0 = core_ring_width[0], spacing0 = core_ring_spacing[0],
                  layer1 = m8, width1 = core_ring_width[0], spacing1 = core_ring_spacing[0],
                  starts_with = pdn.GRID, offset = core_ring_core_offset, pad_offset = core_ring_pad_offset, extend = False,
                  pad_pin_layers = ring_connect_to_pad_layers, nets = [])
  # Add power and ground grid on M1 and attach to cell's VDD/VSS pin
  pdngen.makeFollowpin(grid = g, layer = m1, 
                      width = design.micronToDBU(0.07), extend = pdn.CORE)
  # Create the rest of the power delivery network
  pdngen.makeStrap(grid = g, layer = m4, width = design.micronToDBU(1.2), 
                  spacing = design.micronToDBU(1.2), pitch = design.micronToDBU(6), offset = design.micronToDBU(0), 
                  number_of_straps = 0, snap = False, starts_with = pdn.GRID, extend = pdn.CORE, nets = [])
  pdngen.makeStrap(grid = g, layer = m7, width = design.micronToDBU(1.4),
                  spacing = design.micronToDBU(1.4), pitch = design.micronToDBU(10.8), offset = design.micronToDBU(0),
                  number_of_straps = 0, snap = False, starts_with = pdn.GRID, extend = pdn.RINGS, nets = [])
  pdngen.makeConnect(grid = g, layer0 = m1, layer1 = m4, 
                  cut_pitch_x = pdn_cut_pitch[0], cut_pitch_y = pdn_cut_pitch[1], vias = [], techvias = [],
                  max_rows = 0, max_columns = 0, ongrid = [], split_cuts = dict(), dont_use_vias = )
  pdngen.makeConnect(grid = g, layer0 = m4, layer1 = m7,
                  cut_pitch_x = pdn_cut_pitch[0], cut_pitch_y = pdn_cut_pitch[1], vias = [], techvias = [],
                  max_rows = 0, max_columns = 0, ongrid = [], split_cuts = dict(), dont_use_vias = )
  pdngen.makeConnect(grid = g, layer0 = m7, layer1 = m8,
                  cut_pitch_x = pdn_cut_pitch[0], cut_pitch_y = pdn_cut_pitch[1], vias = [], techvias = [],
                  max_rows = 0, max_columns = 0, ongrid = [], split_cuts = dict(), dont_use_vias = )
# Create power delivery network for macros
# Set the width of the PDN ring for macros and the spacing between VDD and VSS rings for macros
macro_ring_width = [design.micronToDBU(2), design.micronToDBU(2)]
macro_ring_spacing = [design.micronToDBU(2), design.micronToDBU(2)]
macro_ring_core_offset = [design.micronToDBU(0) for i in range(4)]
macro_ring_pad_offset = [design.micronToDBU(0) for i in range(4)]
m5 = design.getTech().getDB().getTech().findLayer("M5")
m6 = design.getTech().getDB().getTech().findLayer("M6")
for i in range(len(macros)):
  for domain in domains:
    pdngen.makeInstanceGrid(domain = domain, name = "Macro_core_grid_" + str(i),
                            starts_with = pdn.GROUND, inst = macros[i], halo = halo,
                            pg_pins_to_boundary = True, default_grid = False, 
                            generate_obstructions = [], is_bump = False)
  grid = pdngen.findGrid("Macro_core_grid_" + str(i))
  for g in grid:
    pdngen.makeRing(grid = g, layer0 = m5, width0 = macro_ring_width[0], spacing0 = macro_ring_spacing[0],
                    layer1 = m6, width1 = macro_ring_width[0], spacing1 = macro_ring_spacing[0],
                    starts_with = pdn.GRID, offset = macro_ring_core_offset, pad_offset = macro_ring_pad_offset, extend = False,
                    pad_pin_layers = macro_ring_connect_to_pad_layers, nets = [])
    pdngen.makeStrap(grid = g, layer = m5, width = design.micronToDBU(1.2), 
                    spacing = design.micronToDBU(1.2), pitch = design.micronToDBU(6), offset = design.micronToDBU(0),
                    number_of_straps = 0, snap = True, starts_with = pdn.GRID, extend = pdn.RINGS, nets = [])
    pdngen.makeStrap(grid = g, layer = m6, width = design.micronToDBU(1.2),
                    spacing = design.micronToDBU(1.2), pitch = design.micronToDBU(6), offset = design.micronToDBU(0),
                    number_of_straps = 0, snap = True, starts_with = pdn.GRID, extend = pdn.RINGS, nets = [])
    pdngen.makeConnect(grid = g, layer0 = m4, layer1 = m5,
                    cut_pitch_x = pdn_cut_pitch[0], cut_pitch_y = pdn_cut_pitch[1], vias = [], techvias = [],
                    max_rows = 0, max_columns = 0, ongrid = [], split_cuts = dict(), dont_use_vias = )
    pdngen.makeConnect(grid = g, layer0 = m5, layer1 = m6,
                    cut_pitch_x = pdn_cut_pitch[0], cut_pitch_y = pdn_cut_pitch[1], vias = [], techvias = [],
                    max_rows = 0, max_columns = 0, ongrid = [], split_cuts = dict(), dont_use_vias = )
    pdngen.makeConnect(grid = g, layer0 = m6, layer1 = m7,
                    cut_pitch_x = pdn_cut_pitch[0], cut_pitch_y = pdn_cut_pitch[1], vias = [], techvias = [],
                    max_rows = 0, max_columns = 0, ongrid = [], split_cuts = dict(), dont_use_vias = )

pdngen.checkSetup()
pdngen.buildGrids(False)
pdngen.writeToDb(True, )
pdngen.resetShapes()
```
Global Routing
```
signal_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
signal_high_layer = design.getTech().getDB().getTech().findLayer("M6").getRoutingLevel()
clk_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
clk_high_layer = design.getTech().getDB().getTech().findLayer("M6").getRoutingLevel()
grt = design.getGlobalRouter()
grt.setMinRoutingLayer(signal_low_layer)
grt.setMaxRoutingLayer(signal_high_layer)
grt.setMinLayerForClock(clk_low_layer)
grt.setMaxLayerForClock(clk_high_layer)
grt.setAdjustment(0.5)
grt.setVerbose(True)
grt.globalRoute(True)
design.getBlock().writeGuides("%s.guide"%designName)
design.evalTclString("estimate_parasitics -global_routing")
```
Detailed Routing
```
import drt

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
params.bottomRoutingLayer = "M1"
params.topRoutingLayer = "M6"
params.verbose = 1
params.cleanPatches = True
params.doPa = True
params.singleStepDR = False
params.minAccessPoints = 1
params.saveGuideUpdates = False
drter.setParams(params)
drter.main()
```
Static IR drop Analysis
```
psm_obj = design.getPDNSim()
psm_obj.setNet(ord.Tech().getDB().getChip().getBlock().findNet("VDD"))
design.evalTclString(f"psm::set_corner [sta::cmd_corner]")
psm_obj.analyzePowerGrid('', False, '', '')
drops = psm_obj.getIRDropForLayer(tech.getDB().getTech().findLayer("M2"))
```
Gate Sizing
```
timing.makeEquivCells()
# First pick an instance
inst = block.findInst("cellName")
# Then get the library cell information
instMaster = inst.getMaster()
equivCells = timing.equivCells(instMaster)
# Perform gate sizing with randomly select an available equivalent cell 
inst.swapMaster(equivCells[0])
```
## Query information from OpenDB
Query library cell information
```
import openroad as ord

# Get OpenDB
db = ord.get_db()
# Get all cell libraries from different files read
libs = db.getLibs()
for lib in libs:
  # Get library name
    lib_name = lib.getName()
    # Get all library cells in that library
    lib_masters = lib.getMasters()
    for master in lib_masters:
    # Get the name of the library cell
      libcell_name = master.getName()
      # Get the area of it by getting the product of its height and width
      libcell_area = master.getHeight() * master.getWidth()
```
Query cell information
```
block = design.getBlock()
# Get all cells in the design
insts = block.getInsts()
# Get the available design corner (change the index to use different corners)
corner = timing.getCorners()[0]

for inst in insts:
  # Get cell name
  cell_name = inst.getName()
  #location
  BBox = inst.getBBox()
  x0 = BBox.xMin()
  y0 = BBox.yMin()
  x1 = BBox.xMax()
  y1 = BBox.yMax()

  masterCell = inst.getMaster()
  # Return True if it's a flipflop
  isSeq = design.isSequential(masterCell)
  # Return True if it's a macro
  isMacro = masterCell.isBlock()
  # Return True if it's a filler cell
  isFiller = masterCell.isFiller()
  # Return True if it's a buffer
  isBuffer = design.isBuffer(masterCell)
  # Return True if it's an inverter
  isInv = design.isInverter(masterCell)
  # Return True if it's in a clock net
  isInClk = design.isInClock(inst)
  # Get the static power of a cell
  cellStaticPower = timing.staticPower(inst, corner)
  # Get the dynamic power of a cell
  cellDynamicPower = timing.dynamicPower(inst, corner)
  # Get all pins of the cell
  ITerms = inst.getITerms()
```
Query net information
```
block = design.getBlock()
# Get all nets in the design
nets = block.getNets()
# Get the available design corner (change the index to use different corners)
corner = timing.getCorners()[0]
for net in nets:
  # Return "POWER" if the net is a power (VDD) net. Return "GROUND" is the net is a ground (VSS) net.
  sigType = net.getSigType()
  # Get the name of the net
  net_name = net.getName()
  # Get all the pins connected to this net
  net_ITerms = net.getITerms()
  # Get the total wire capacitance of the net
  net_cap = net.getTotalCapacitance()
  # Get the total wire resistance of the net
  net_res = net.getTotalResistance()
  # Get the total wire coupling capacitance of the net
  net_coupling = net.getTotalCouplingCap()
  # Get the pin capacitance + wire capacitance of the net
  total_cap = timing.getNetCap(net, corner, timing.Max)
  # Get the number of fanout of the net
  outputPins = []
  net_ITerms = net.getITerms()
    for ITerm in net_ITerms:
      if (ITerm.isInputSignal()):
        outputPins.append(ITerm)
  fanOut = len(outputPins)
  # Get the length of the net
  netRouteLength = design.getNetRoutedLength(net)
```
Query pin information
```
block = design.getBlock()
# Get all pins in the design
ITerms = block.getITerms()
for ITerm in ITerms:
  # Get the pin name
  pinName = design.getITermName(ITerm)
  # The net connects to this pin
  net = ITerm.getNet()
  # Get the cell that this pin belongs to
  cell = ITerm.getInst()
  # Return True if the pin is an output pin of the cell
  outputPin = ITerm.isOutputSignal()
  # Return True if the pin is an input pin of the cell
  inputPin = ITerm.isInputSignal()
  # Get the x and y location of the pin
  PinXY_list = ITerm.getAvgXY()
  if PinXY_list[0]:
    x = PinXY_list[1]
    y = PinXY_list[2]
  # Return True if the pin is a sink pin of any timing path
  is_endpoint = timing.isEndpoint(ITerm)
  # Get the slew of the pin
  pinSlew = timing.getPinSlew(ITerm)
  # Get the falling slack of the pin
  pinFallSlack = timing.getPinSlack(ITerm, timing.Fall, timing.Max)
  # Get the rising slack of the pin
  pinRiseSlack = timing.getPinSlack(ITerm, timing.Rise, timing.Max)
  # Get the rising arrival time of the pin
  pinRiseArr = timing.getPinArrival(ITerm, timing.Rise)
  # Get the falling arrival time of the pin
  pinFallArr = timing.getPinArrival(ITerm, timing.Fall)
  # Get the max permitted load capacitance limit of the pin
  maxCap = timing.getMaxCapLimit(library_cell_pin)
  # Get the max permitted slew of the pin
  maxSlew = timing.getMaxSlewLimit(library_cell_pin)
  # Get the input capacitance of a pin if it is an input pin of a cell
  if ITerm.isInputSignal():
    inputPinCap = timing.getPortCap(ITerm, corner, timing.Max)
```
 





