import numpy as np

from classes import Agent, Obstacle

def init_map(map):
    modify_map(map,a=False)
    modify_map(map,a=True)
    generate_obstacles(map)
    generate_agents(map)

def modify_map(map,a):
    p_rate = map.p_ag_rate if a else map.p_ob_rate
    name = "agents" if a else "obstacles"
    if p_rate > 1:
        if map.fixed_map_size:
            old_n = map.n_agents if a else map.n_obstacles_walls_excluded
            new_n = int(old_n/p_rate)
            if a:
                map.n_agents = new_n
            else:
                map.n_obstacles_walls_excluded = new_n
        else:
            if a:
                a_pol = map.a_map - map.a_obsts
                b_pol = (map.len_x + map.len_y) * map.d_signed - 2 * map.d_min_ao * np.sqrt(map.a_obsts * map.n_obstacles_walls_excluded)
                c_pol = (1 - p_rate) * map.d_min_ao**2 * map.n_obstacles_walls_excluded - p_rate * (a_pol + b_pol)
            else:
                c_pol = p_rate * map.d_min_ao**2  * map.n_obstacles_walls_excluded
                b_pol = p_rate * 2 * map.d_min_ao * np.sqrt(map.a_obsts * map.n_obstacles_walls_excluded)
                a_pol = (p_rate - 1) * map.a_obsts - (1/p_rate) * (b_pol + c_pol)
            roots = np.roots([a_pol,b_pol,c_pol])
            real_roots = roots[np.real(roots)==roots]
            usable_roots = real_roots[real_roots>0]
            new_scale = min(usable_roots) if usable_roots.size else None
            if new_scale is not None:
                new_scale_rounded = (100*new_scale//1+1)/100
                rescale_map(map, new_scale_rounded)
                print(f"Too many {name}, rescaling map using n =",new_scale_rounded)
            else:
                while(p_rate > 1):
                    p_rate_rounded = (100*p_rate//1+1)/100
                    rescale_map(map, p_rate_rounded)
                    print(f"Too many {name}, rescaling map using p =", p_rate_rounded)
                    p_rate = map.p_ag_rate if a else map.p_ob_rate

def rescale_map(map,new_scale):
    map.len_x, map.len_y = map.len_x*new_scale, map.len_y*new_scale
    map.x_max = map.x_min + map.len_x
    map.y_max = map.y_min + map.len_y
    
    map.len_x_obst, map.len_y_obst = map.len_x_obst*new_scale, map.len_y_obst*new_scale
    if np.any(map.all_obstacles):
        for obst in map.all_obstacles:
            if obst.id == "left_wall":
                obst.y_max = map.y_max
            elif obst.id == "right_wall":
                obst.x_min,obst.x_max,obst.y_max = map.x_max,map.x_max,map.y_max
            elif obst.id == "down_wall":
                obst.x_max = map.x_max
            elif obst.id == "up_wall":
                obst.x_max,obst.y_min,obst.y_max = map.x_max,map.y_max,map.y_max
            else:
                obst.x_min, obst.y_min = obst.x_min*new_scale, obst.y_min*new_scale
                obst.x_max, obst.y_max = obst.x_max*new_scale, obst.y_max*new_scale

def generate_obstacles(map):
    if map.map_walls:
        left_wall = Obstacle(id="left_wall",x_min=map.x_min,y_min=map.y_min,
                                x_max=map.x_min,y_max=map.y_max)
        right_wall = Obstacle(id="right_wall",x_min=map.x_max,y_min=map.y_min,
                                x_max=map.x_max,y_max=map.y_max)
        down_wall = Obstacle(id="down_wall",x_min=map.x_min,y_min=map.y_min,
                                x_max=map.x_max,y_max=map.y_min)
        up_wall = Obstacle(id="up_wall",x_min=map.x_min,y_min=map.y_max,
                            x_max=map.x_max,y_max=map.y_max)
        map.all_obstacles = np.append(map.all_obstacles,[left_wall,right_wall,down_wall,up_wall])
    iter,j_max,idx_j_max = 0,0,0
    for i in range(map.n_obstacles_walls_excluded):
        j=0
        while True:
            x_min = np.random.uniform(map.x_min, map.x_max-map.len_x_obst)
            y_min = np.random.uniform(map.y_min, map.y_max-map.len_y_obst)
            x_max, y_max = x_min+map.len_x_obst, y_min+map.len_y_obst
            obst = Obstacle(id=i,x_min=x_min,y_min=y_min,x_max=x_max,y_max=y_max)
            if not np.any(map.all_obstacles):
                map.all_obstacles = np.append(map.all_obstacles,obst)
                break
            rectangles = [create_bigger_rectangle(o,map.d_min_oo/2) for o in map.all_obstacles]
            rect_i = create_bigger_rectangle(obst,map.d_min_oo/2)
            overlap = False
            for rect in rectangles:
                if rectangles_overlap(rect,rect_i):
                    overlap = True
                    break
            if overlap:
                j += 1
                continue
            iter += j+1
            if j_max < j+1:
                j_max = j+1
                idx_j_max = i
            map.all_obstacles = np.append(map.all_obstacles,obst)
            break
    print(iter,j_max,idx_j_max)

def generate_agents(map):
    iter,j_max,idx_j_max = 0,0,0
    d_closest_agents,id_closest_agents = np.inf,[0,0]
    rectangles = [create_bigger_rectangle(o,0) for o in map.all_obstacles]
    for i in range(map.n_agents):
        j=0
        while True:
            overlap = False
            x = np.random.uniform(map.x_min, map.x_max)
            y = np.random.uniform(map.y_min, map.y_max)
            pos = np.array([[x],[y]])

            if not np.any(map.all_agents):
                for rect in rectangles:
                    if rectangle_circle_overlap(rect,[x,y,map.d_min_ao]):
                        overlap = True
                        break
                if overlap:
                    j += 1
                    continue
                iter += j+1
                a = Agent(id=i,p=pos)
                map.all_agents = np.append(map.all_agents,a)
                map.init_positions_array = np.array([[x,y]])
                break

            dists_sq = np.sum((map.init_positions_array - np.array([x,y]))**2,axis=1)
            d_closest = np.sqrt(np.min(dists_sq))
            id_closest = int(np.argmin(dists_sq))

            if d_closest < map.d_min_aa:
                j += 1
                continue
            else:
                for rect in rectangles:
                    if rectangle_circle_overlap(rect,[x,y,map.d_min_ao]):
                        overlap = True
                        break
                if overlap:
                    j += 1
                    continue
                iter += j+1
                if j_max < j+1:
                    j_max = j+1
                    idx_j_max = i
                if d_closest_agents > d_closest:
                    d_closest_agents = d_closest
                    id_closest_agents = [i,id_closest]
                a = Agent(id=i,p=pos)
                map.all_agents = np.append(map.all_agents,a)
                map.init_positions_array = np.append(map.init_positions_array,[[x,y]],axis=0)
                break
    #     print(i,iter,j_max,idx_j_max,d_closest_agents,id_closest_agents)
    print(iter,j_max,idx_j_max,d_closest_agents,id_closest_agents)
        
    def reinitialize_pos(map):
        for i in range(len(map.all_agents)):
            x,y = map.init_positions_array
            map.all_agents[i].p = np.array([[x],[y]])
            map.all_agents[i].v = np.array([[0],[0]])
            map.all_agents[i].a = np.array([[0],[0]])
            map.all_agents[i].p_des = np.array([[],[]])
            map.all_agents[i].v_des = np.array([[],[]])

def create_bigger_rectangle(obst,d_min):
    x_min = obst.x_min - d_min
    y_min = obst.y_min - d_min
    x_max = obst.x_max + d_min
    y_max = obst.y_max + d_min
    rect = [x_min,y_min,x_max,y_max]
    return rect

def rectangles_overlap(rect1, rect2):
    x_min1, y_min1, x_max1, y_max1 = rect1
    x_min2, y_min2, x_max2, y_max2 = rect2
    overlap = not (
        x_max1 < x_min2 or
        x_min1 > x_max2 or
        y_max1 < y_min2 or
        y_min1 > y_max2
    )
    return overlap

def rectangle_circle_overlap(rect, circle): 
    x_min, y_min, x_max, y_max = rect
    cx, cy, r = circle
    closest_x = np.clip(cx, x_min, x_max) # !!! WARNING !!! Only works if rectangle is aligned with map
    closest_y = np.clip(cy, y_min, y_max) # !!! WARNING !!! Only works if rectangle is aligned with map
    dist_x = cx - closest_x
    dist_y = cy - closest_y
    dist = np.sqrt(dist_x**2 + dist_y**2)
    return dist <= r