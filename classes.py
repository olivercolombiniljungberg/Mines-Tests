import numpy as np

from target_points import target_point

# Global constants
DRONE_SIZE                      = 0.2
MAX_AGENT_OCCUPATION_RATE       = 0.45
MAX_OBSTACLE_NON_ACC_RATE       = 0.5
MAX_OBSTACLE_OCCUPATION_RATE    = 0.2
D_MIN_SCALE                     = 1.5
DT                              = 0.01

# Default values
n_a, n_o = 10, 0
l_x, l_y = 1, 1
x_m, y_m = 0, 0
x_M, y_M = None, None

class Agent:
    def __init__(self, id=0, group=0, r=DRONE_SIZE/2, 
                 p=np.array([[0],[0]]), v=np.array([[0],[0]]), a=np.array([[0],[0]])):
        self.id = id
        self.group = group
        self.r = r
        self.p,self.v,self.a = p,v,a
        self.p_field,self.v_field = np.array([[],[]]),np.array([[],[]])
        self.p_target,self.v_target = np.array([[],[]]),np.array([[],[]])
        self.p_des,self.v_des = np.array([[],[]]),np.array([[],[]])

class Obstacle:
    def __init__(self, id=0, x_min=0, y_min=0, x_max=0, y_max=0):
        self.id = id
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.len_x, self.len_y = x_max-x_min, y_max-y_min

class Map:
    @property
    def a_map(self):
        return self.len_x * self.len_y
    @property
    def a_obsts(self):
        return self.n_obstacles_walls_excluded * (self.len_x_obst * self.len_y_obst)
    @property
    def a_map_acc(self):
        return (self.len_x + self.d_signed) * (self.len_y + self.d_signed)
    @property
    def a_obst_non_acc(self):
        return self.n_obstacles_walls_excluded * (self.len_x_obst + self.d_min_ao) * (self.len_y_obst + self.d_min_ao)
    @property
    def a_free(self):
        return self.a_map_acc - self.a_obst_non_acc
    @property
    def p_ob_rate(self):
        ob_occupation_rate = self.a_obst_non_acc / self.a_map if self.a_map > 0 else 0
        return ob_occupation_rate / MAX_OBSTACLE_NON_ACC_RATE
    @property
    def p_ag_rate(self):
        a_agents = np.pi * (self.d_min_aa / 2)**2 * self.n_agents
        ag_occupation_rate = a_agents / self.a_free if self.a_free > 0 else 0
        return ag_occupation_rate / MAX_AGENT_OCCUPATION_RATE
    
    @property
    def all_groups(self):
        groups = np.unique([a.group for a in self.all_agents])
        return groups
    
    def __init__(self, fixed_map_size=False, map_walls=False, n_agents=n_a, n_obstacles=n_o, dt=DT,
                 len_x=l_x, len_y=l_y, x_min=x_m, y_min=y_m, x_max=x_M, y_max=y_M):

        self.fixed_map_size = fixed_map_size
        self.map_walls = map_walls
        self.dt = dt
        self.d_min_aa = D_MIN_SCALE * 2 * Agent().r
        self.d_min_ao = self.d_min_aa - Agent().r
        self.d_min_oo = self.d_min_aa
        self.d_signed = -self.d_min_aa if self.map_walls else self.d_min_aa
        self.n_obstacles_walls_excluded = n_obstacles
        self.n_obstacles = self.n_obstacles_walls_excluded + 4 if self.map_walls else self.n_obstacles_walls_excluded
        self.n_agents = n_agents
        
        self.x_min = x_min
        self.y_min = y_min
        min_len_x, min_len_y = max(len_x,DRONE_SIZE), max(len_y,DRONE_SIZE)
        self.x_max = x_min+min_len_x if (x_max is None or x_max-x_min<=DRONE_SIZE) else x_max
        self.y_max = y_min+min_len_y if (y_max is None or y_max-y_min<=DRONE_SIZE) else y_max
        self.len_x = self.x_max-self.x_min
        self.len_y = self.y_max-self.y_min
        a_map = self.len_x * self.len_y
        a_obsts = a_map * MAX_OBSTACLE_OCCUPATION_RATE
        a_obst = a_obsts / self.n_obstacles_walls_excluded if self.n_obstacles_walls_excluded else 0
        self.len_x_obst, self.len_y_obst = np.sqrt(a_obst), np.sqrt(a_obst)

        self.all_agents = np.array([])
        self.all_obstacles = np.array([])
        self.C_O_M = np.array([])