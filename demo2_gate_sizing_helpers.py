import openroad as ord
import pdn, odb, utl
from openroad import Tech, Design, Timing
from collections import defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, RelGraphConv
import random
from itertools import count
import dgl
import copy
from pathlib import Path

# replay memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('graph', 'action', 'next_state', 'reward'))
        
    # insert if not yet filled else trea it like a circular buffer and add.
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # random sampling for the training step
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

  def __init__(self, n_state, n_cells, n_features, device):
    super(DQN, self).__init__()
    self.conv1 = RelGraphConv(n_state,64,2)
    self.conv2 = RelGraphConv(64,64,2)
    self.conv3 = RelGraphConv(64,2,2)
    self.device = device
    self.n_state = n_state
    self.n_cells = n_cells
    self.n_features = n_features
  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def  forward(self, graph, x=None):
    if x is None:
      x = get_state(graph, self.n_state, self.n_cells, self.n_features).to(self.device)
    e = graph.edata['types']
    x = F.relu(self.conv1(graph, x, e))
    x = F.relu(self.conv2(graph, x, e))
    x = self.conv3(graph, x, e)
    # Get a list of actions that are not valid and ensure they cant be selected.
    mask = generate_masked_actions(graph)
    x = x*(~mask) + (mask)*(x.min() - 1)
    return x

def get_type(cell_type, cell_dict, cell_name_dict):
  cell, drive = cell_type.split("_")
  drive = "_"+drive
  if cell in cell_name_dict:
    cell_values = cell_dict[cell_name_dict[cell]]
    if drive in cell_values['sizes']:
      idx = cell_values['sizes'].index(drive)
      return int(cell_name_dict[cell]), idx
    else:
      print("Drive strength "+drive+" not found in cell :"+cell)
      print("Possible sizes"+cell_values['sizes'])
      return None,None
  else:
    print("cell: "+cell+" not in dictionary")
    return None,None

def pin_properties(dbpin, CLKset, ord_design, timing):
  ITerms = dbpin.getNet().getITerms()
  #slack
  slack = min(timing.getPinSlack(dbpin, timing.Fall, timing.Max), timing.getPinSlack(dbpin, timing.Rise, timing.Max))
  if slack < -0.5*CLKset[0]:
    slack = 0
  #slew
  slew = timing.getPinSlew(dbpin)  
  #load
  #Corners = timing.getCorners()
  load = 0
  for ITerm in ITerms:
    if ITerm.isInputSignal():
      new_load = 0
      for corner in timing.getCorners():
        tmp_load = timing.getPortCap(ITerm, corner, timing.Max)
        if tmp_load > new_load:
          new_load = tmp_load
      load += new_load

  return slack, slew, load

def min_slack(dbpin, timing):
  slack = min(timing.getPinSlack(dbpin, timing.Fall, timing.Max), timing.getPinSlack(dbpin, timing.Rise, timing.Max))
  return slack

def generate_masked_actions(graph):
  # max size keep track of the index of the maximum size.
  # If the current size is maximum size we mask it out as an action
  upper_mask = graph.ndata['cell_types'][:,1] >= graph.ndata['max_size']-1
  lower_mask = graph.ndata['cell_types'][:,1] == 0
  ###########################
  #upper_mask = upper_mask | (~slack_mask | ~slew_mask| ~load_mask | ~DFF_mask)
  #         if len(delay_mask)!=0:
  #             upper_mask[torch.tensor(delay_mask)] = True
  ##########################
  # if the criteria for the mask is met we replace it with the minimum
  # to make sure that that action is never chosen
  mask = torch.cat((upper_mask.view(-1,1), lower_mask.view(-1,1)),1)
  return mask

def update_lambda(initial_lambda, slacks, K):
    Slack_Lambda = initial_lambda * ((1-slacks)**K)
    return Slack_Lambda

def optimize_model(memory, BATCH_SIZE, device, GAMMA, policy_net,\
                  target_net, optimizer, loss_history):
  if len(memory) < BATCH_SIZE:
    return optimizer, loss_history
  transitions = memory.sample(BATCH_SIZE)
  Transition = namedtuple('Transition', ('graph', 'action', 'next_state', 'reward'))
  batch = Transition(*zip(*transitions))
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = torch.zeros(BATCH_SIZE, device=device,dtype=torch.float32)
  for n_state, graph in enumerate(batch.graph):
    actions = policy_net(graph)
    state_action_values[n_state] = actions.view(-1)[action_batch[n_state,0]]
  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(BATCH_SIZE, device=device,dtype=torch.float32)

  for n_state, state in enumerate(batch.next_state):
    if state is not None:
      graph = batch.graph[n_state]
      state_g = state.to(device)
      next_state_values[n_state] = target_net(graph, state_g.view(graph.num_nodes(),-1)).max().detach()
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  loss_history.append(loss.item())
  optimizer.step()
  return optimizer, loss_history

def select_action(graph, inference = False, total_taken = False,\
                  steps_done = False, random_taken = False, policy_net = False,\
                  EPS_END = False, EPS_START = False, EPS_DECAY = False, device = False):
  total_taken +=1
  if inference:
    with torch.no_grad():
      action = policy_net(graph)
      return torch.argmax(action.view(-1)).view(1,1), total_taken, steps_done, random_taken

  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * 0.95**(steps_done / EPS_DECAY)
  steps_done += 1
  #get the mask
  mask = generate_masked_actions(graph)

  if int(sum(~mask.view(-1)))==0 :
    return -1, total_taken, steps_done, random_taken
  #Threshold keeps decreasing, so over time it takes more from the policy net.
  if sample > eps_threshold:
    with torch.no_grad():
      action = policy_net(graph)
      return torch.argmax(action.view(-1)).view(1,1), total_taken, steps_done, random_taken
  else:
    action = torch.randn_like(mask,dtype=torch.float32)
    action = (action-action.min()+1)*(~mask)
    random_taken+=1
    return torch.tensor([[torch.argmax(action.view(-1))]], device=device, dtype=torch.long),\
            total_taken, steps_done, random_taken

def get_subgraph(graph, nodes):
  node_set = {x.item() for x in nodes}
  #level 1
  in_nodes, _ = graph.in_edges(list(node_set))
  _, out_nodes = graph.out_edges(list(node_set))
  node_set.update(in_nodes.tolist())
  node_set.update(out_nodes.tolist())
  #level 2
  in_nodes, _ = graph.in_edges(list(node_set))
  _, out_nodes = graph.out_edges(list(node_set))
  node_set.update(in_nodes.tolist())
  node_set.update(out_nodes.tolist())

  subgraph = dgl.node_subgraph(graph, list(node_set))

  return subgraph

def get_critical_path_nodes(graph, ep_num, TOP_N_NODES, n_cells):
  topk = min(len(graph.ndata['slack'])-1 , int(TOP_N_NODES*(1+0.01*ep_num)))
  min_slacks, critical_path = torch.topk(graph.ndata['slack'], topk, largest=False)
  critical_path = critical_path[min_slacks<0]

  if critical_path.numel() <=0:
    critical_path = torch.arange(0,graph.num_nodes())

  return critical_path

def get_state(graph, n_state, n_cells, n_features):
  state = torch.zeros(graph.num_nodes(), n_state)
  state[:,-1] = graph.ndata['area']
  state[:,-2] = graph.ndata['slack']
  state[:,-3] = graph.ndata['slew']
  state[:,-4] = graph.ndata['load']
  state[:,:-n_features] =F.one_hot(graph.ndata['cell_types'][:,0],n_cells)*graph.ndata['cell_types'][:,1:2]
  return state

def env_step(episode_G, graph, state, action, CLKset, ord_design, timing,\
            cell_dict, norm_data, inst_names, episode_inst_dict, inst_dict,\
            n_cells, n_features, block, device, Slack_Lambda, eps):
  next_state = state.clone()
  reward = 0
  done =0
  #based on the selected action you choose the approriate cell and upsize it or downsize
  cell_sub = int(action/2)
  cell = graph.ndata['_ID'][cell_sub].item()
  inst_name = inst_names[cell]
  cell_size = episode_inst_dict[inst_name]['cell_type'][1]
  cell_idx = episode_inst_dict[inst_name]['cell_type'][0]
  dbpin = block.findITerm(inst_name + cell_dict[str(inst_dict[inst_name]['cell_type'][0])]['out_pin'])
  old_slack = min_slack(dbpin, timing)
  o_master_name = cell_dict[str(cell_idx)]['name']+\
                  cell_dict[str(cell_idx)]['sizes'][cell_size]
  if(action%2 == 0):
      cell_size +=1
  else:
      cell_size -=1
  if(cell_size>=cell_dict[str(cell_idx)]['n_sizes']):
    print("Above max")
    print(action,cell_dict[str(cell_idx)]['n_sizes'], cell_idx, cell_size)
  if(cell_size<0):
    print("below min")
    print(action,cell_dict[str(cell_idx)]['n_sizes'], cell_idx, cell_size)
  episode_inst_dict[inst_name]['cell_type'] = (cell_idx,cell_size)
  size = cell_dict[str(cell_idx)]['sizesi'][cell_size] #actual size

  #one hot encode the relavant feature with the magnitude of size.
  next_state[cell_sub,:-n_features] = F.one_hot(torch.tensor([cell_idx]),n_cells)*size
  episode_G.ndata['cell_types'][cell] = torch.tensor((cell_idx,cell_size))

  #replace the master node in the code and find the new slack,
  inst = block.findInst(inst_name)
  n_master_name = cell_dict[str(cell_idx)]['name']+\
                  cell_dict[str(cell_idx)]['sizes'][cell_size]
  db = ord.get_db()
  n_master = db.findMaster(n_master_name)
  inst.swapMaster(n_master)
  dbpin = block.findITerm(inst_name + cell_dict[str(inst_dict[inst_name]['cell_type'][0])]['out_pin'])
  new_slack = min_slack(dbpin, timing)

  old_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
  episode_inst_dict[inst_name]['area']= n_master.getWidth() * n_master.getHeight()
  new_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
  episode_G.ndata['area'][cell] = new_area

  # update_area
  next_state[cell_sub,-1] = new_area
  reward += torch.tensor(old_area-new_area)
  old_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slews = torch.zeros(len(episode_inst_dict.keys()))
  new_loads = torch.zeros(len(episode_inst_dict.keys()))

  for n, inst in inst_names.items():
    old_slacks[n] = episode_inst_dict[inst]['slack']
    tmp_db_pin = block.findITerm(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
      episode_inst_dict[inst]['slew'],
      episode_inst_dict[inst]['load']) = pin_properties(tmp_db_pin, CLKset, ord_design, timing)
      new_slacks[n] = episode_inst_dict[inst]['slack']
      new_slews[n] = episode_inst_dict[inst]['slew']
      new_loads[n] = episode_inst_dict[inst]['load']
  episode_G.ndata['slack'] = new_slacks.to(device)/norm_data['clk_period']
  for i in range(len(episode_G.ndata['slack'])):
    if episode_G.ndata['slack'][i] > 1:
      episode_G.ndata['slack'][i] = 1
  episode_G.ndata['slack'][torch.isinf(episode_G.ndata['slack'])] = 1
  episode_G.ndata['slew'] = new_slews.to(device)/ norm_data['max_slew']
  episode_G.ndata['load'] = new_loads.to(device)/ norm_data['max_load']

  next_state[:,-2] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['slack']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['clk_period']
  next_state[:,-3] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['slew']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['max_slew']
  next_state[:,-4] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['load']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['max_load']
  next_state[torch.isinf(next_state[:,-2]),-2] = 1 # remove infinity from primary outputs. hadle it better.

  #Check TNS
  new_TNS = torch.min(new_slacks,torch.zeros_like(new_slacks))
  new_TNS = Slack_Lambda*new_TNS

  old_TNS = torch.min(old_slacks,torch.zeros_like(old_slacks))
  old_TNS = Slack_Lambda*old_TNS

  factor = torch.max(torch.abs(0.1*old_TNS), eps*torch.ones_like(old_TNS))
  factor = torch.max(torch.ones_like(old_TNS), 1/factor)
  reward += (torch.sum((new_TNS - old_TNS) * factor)).item()

  return reward, done, next_state, episode_inst_dict, episode_G

def env_reset(reset_state = None, episode_num = None, cell_name_dict = None,\
              CLKset = None, ord_design = None, timing = None, G = None,\
              inst_dict = None, CLK_DECAY = None, CLK_DECAY_STRT = None,\
              clk_init = None, clk_range = None, clk_final = None, inst_names = None,\
              block = None, cell_dict = None, norm_data = None, device = None):
  episode_G = copy.deepcopy(G)
  episode_inst_dict = copy.deepcopy(inst_dict)

  if episode_num is not None:

    if episode_num<CLK_DECAY_STRT:
      clk = clk_init
    elif episode_num<CLK_DECAY+CLK_DECAY_STRT:
      clk = clk_init - clk_range*(episode_num -CLK_DECAY_STRT) /CLK_DECAY
    else:
      clk = clk_final
    ord_design.evalTclString("create_clock [get_ports i_clk] -name core_clock -period " + str(clk*1e-9))

  for i in range(len(inst_names)):
    inst_name = inst_names[i]
    inst = block.findInst(inst_name)
    if reset_state is not None:
      o_master_name = reset_state[i]
      cell_idx, cell_size = get_type(o_master_name, cell_dict, cell_name_dict)
      episode_inst_dict[inst_name]['cell_type'] = (cell_idx,cell_size)
      episode_G.ndata['cell_types'][i] = torch.tensor((cell_idx,cell_size))
    else:
      cell_size = episode_G.ndata['cell_types'][i,1].item()
      cell_idx = episode_G.ndata['cell_types'][i,0].item()
      o_master_name = cell_dict[str(cell_idx)]['name']+\
              cell_dict[str(cell_idx)]['sizes'][cell_size]

    db = ord.get_db()
    o_master = db.findMaster(o_master_name)
    if o_master_name != inst.getMaster().getName():
      inst.swapMaster(o_master)
    if reset_state is not None:
      episode_inst_dict[inst_name]['area']= o_master.getWidth() * o_master.getHeight()
      new_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
      episode_G.ndata['area'][i] = new_area

  #if reset_state is not None:
  new_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slews = torch.zeros(len(episode_inst_dict.keys()))
  new_loads = torch.zeros(len(episode_inst_dict.keys()))

  for n, inst in inst_names.items():
    tmp_db_pin = block.findITerm(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
      episode_inst_dict[inst]['slew'],
      episode_inst_dict[inst]['load']) = pin_properties(tmp_db_pin, CLKset, ord_design, timing)
      new_slacks[n] = episode_inst_dict[inst]['slack']
      new_slews[n] = episode_inst_dict[inst]['slew']
      new_loads[n] = episode_inst_dict[inst]['load']
  episode_G.ndata['slack'] = new_slacks.to(device)/norm_data['clk_period']
  for i in range(len(episode_G.ndata['slack'])):
    if episode_G.ndata['slack'][i] > 1:
      episode_G.ndata['slack'][i] = 1
  episode_G.ndata['slack'][torch.isinf(episode_G.ndata['slack'])] = 1
  episode_G.ndata['slew'] = new_slews.to(device)/ norm_data['max_slew']
  episode_G.ndata['load'] = new_loads.to(device)/ norm_data['max_load']

  return episode_G, episode_inst_dict, clk


def calc_cost(ep_G, Slack_Lambda):
  cost = torch.sum(ep_G.ndata['area'].to('cpu'))
  x = ep_G.ndata['slack'].to('cpu')
  new_slacks = torch.min(x, torch.zeros_like(x))
  cost += torch.sum(Slack_Lambda*(-new_slacks))
  return cost

def get_state_cells(ep_dict, inst_names, cell_dict):
  cells = []
  for x in inst_names.values():
    cell_size = ep_dict[x]['cell_type'][1]
    cell_idx = ep_dict[x]['cell_type'][0]
    cell_name = cell_dict[str(cell_idx)]['name']+\
                cell_dict[str(cell_idx)]['sizes'][cell_size]
    cells.append(cell_name)
  return cells

def pareto(pareto_points, pareto_cells, area, clk, ep_dict, inst_names,\
            cell_dict, inst_dict, block, ord_design, timing):
  cells = get_state_cells(ep_dict, inst_names, cell_dict)
  if len(pareto_points) <= 0:
    pareto_points.append((area, clk))
    pareto_cells.append(cells)
    return 1
  dominated_points = set()
  for n, pareto_point in enumerate(pareto_points):
    # if new point is dominated we skip
    if pareto_point[0] <= area and  pareto_point[1] <= clk:
      return 0
    # if new point dominates any other point
    elif pareto_point[0] >= area and  pareto_point[1] >= clk:
      dominated_points.add(n)

  print("updating point")
  print((area, clk))
  print("dominated points: ", [pareto_points[n] for n in dominated_points])
  pareto_points.append((area, clk))
  pareto_cells.append(cells)
  pareto_points = [val for n, val in enumerate(pareto_points) if n not in dominated_points]
  pareto_cells = [val for n, val in enumerate(pareto_cells) if n not in dominated_points]
  print("new pareto points: ",pareto_points)
  slacks = [min_slack(block.findITerm(x + cell_dict[str(inst_dict[x]['cell_type'][0])]['out_pin']), timing) for x in inst_names.values()]
  test_sl = np.min(slacks)
  return 1

def rmdir(directory):
  directory=Path(directory)
  for  item in directory.iterdir():
    if item.is_dir():
      rmdir(directory)
    else:
      item.unlink()
  directory.rmdir()

unit_micron = 2000
design = 'pid'
semi_opt_clk = '0.65'
clock_name = "i_clk"
CLK_DECAY=100; CLK_DECAY_STRT=25
n_features = 4
BATCH_SIZE = 64#128
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 200
LR = 0.001
BUF_SIZE = 1500#10000
STOP_BADS = 50
MAX_STEPS = 50#150#200 c432#51#300
TARGET_UPDATE = MAX_STEPS*25 #MAX_STEPS*5
EPISODE = 2 #150 # c880 #50 c432 #15#200
LOADTH = 0
DELTA = 0.000001
UPDATE_STOP = 250

TOP_N_NODES = 100
eps = 1e-5
inference = False#True#False
retrain = False
max_TNS = -100
max_WNS = -100
count_bads = 0
best_delay = 5
min_working_clk = 100
K = 4# set seprately to critical and non critical # accelration factor
CLKset = [0.6]
clk_final = CLKset[0]
clk_range = 0.98*(float(semi_opt_clk) - CLKset[0])
clk_init = clk_final + clk_range


def load_design(path):
  ord_tech = Tech()
  ord_tech.readLiberty(path + "/data/NangateOpenCellLibrary_typical_para2.lib")
  ord_tech.readLef(path + "/data/NangateOpenCellLibrary.lef")
  ord_design = Design(ord_tech)
  timing = Timing(ord_design)
  ord_design.readVerilog(path + "/data/%s_%s.v" % (design, semi_opt_clk))
  ord_design.link(design)
  ord_design.evalTclString("create_clock [get_ports i_clk] -name core_clock -period " + str(clk_init*1e-9))
  db = ord.get_db()
  chip = db.getChip()
  block = ord.get_db_block()
  nets = block.getNets()
  return ord_tech, ord_design, timing, db, chip, block, nets


def iterate_nets_get_properties(ord_design, timing, nets, block, cell_dict, cell_name_dict):
  #This must eventually be put into a create graph function.
  #source and destination instances for the graph function.
  srcs = []
  dsts = []
  #Dictionary that stores all the properties of the instances.
  inst_dict = {}
  #Dictionary that keeps a stores the fanin and fanout of the instances in an easily indexable way.
  fanin_dict = {}
  fanout_dict = {}
  #storing all the endpoints(here they are flipflops)
  endpoints = []
  for net in nets:
    iterms = net.getITerms()
    net_srcs = []
    net_dsts = []
    # create/update the instance dictionary for each net.
    for s_iterm in iterms:
      inst = s_iterm.getInst()
      inst_name = s_iterm.getInst().getName()
      term_name = s_iterm.getInst().getName() + "/" + s_iterm.getMTerm().getName()
      cell_type = s_iterm.getInst().getMaster().getName()
  
      if inst_name not in inst_dict:
        i_inst = block.findInst(inst_name)
        m_inst = i_inst.getMaster()
        area = m_inst.getWidth() * m_inst.getHeight()
        inst_dict[inst_name] = {
          'idx':len(inst_dict),
          'cell_type_name':cell_type,
          'cell_type':get_type(cell_type, cell_dict, cell_name_dict),
          'slack':0,
          'slew':0,
          'load':0,
          'cin':0,
          'area': area}
      if s_iterm.isInputSignal():
        net_dsts.append((inst_dict[inst_name]['idx'],term_name))
        if inst_dict[inst_name]['cell_type'][0] == 16: # check for flipflops
          endpoints.append(inst_dict[inst_name]['idx'])
      elif s_iterm.isOutputSignal():
        net_srcs.append((inst_dict[inst_name]['idx'],term_name))
        (inst_dict[inst_name]['slack'],
         inst_dict[inst_name]['slew'],
         inst_dict[inst_name]['load'])= pin_properties(s_iterm, CLKset, ord_design, timing)
      else:
        print("Should not be here")
    # list the connections for the graph creation step and the fainin/fanout dictionaries
    for src,src_term in net_srcs:
      for dst,dst_term in net_dsts:
        srcs.append(src)
        dsts.append(dst)
        src_key = list(inst_dict.keys())[src]
        dst_key = list(inst_dict.keys())[dst]
        if src_key in fanout_dict.keys():
          fanout_dict[src_key].append(dst_key)
        else:
          fanout_dict[src_key] = [dst_key]
        if dst_key in fanin_dict.keys():
          fanin_dict[dst_key].append(src_key)
        else:
          fanin_dict[dst_key] = [src_key]
  return inst_dict, endpoints, srcs, dsts, fanin_dict, fanout_dict


