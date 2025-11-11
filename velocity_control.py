import numpy as np

from target_points import target_point

V_MAX       = 3.0
R_REP       = 0.5
R_ATT       = 1.0
K_REP       = 8.0
K_ATT       = 0.2
K_TARGET    = 1.5

def append_vel_pos(map, t=-1, v_max=V_MAX, r_rep=R_REP, r_att=R_ATT, k_rep=K_REP, k_att=K_ATT, k_target=K_TARGET):
    all_obstacles, groups, dt = map.all_obstacles, map.all_groups, map.dt
    t_idx = t if t >= 0 else len(map.all_agents[0].p[0])-1
    time = t_idx * map.dt
    diffs, dists, center_of_mass = compute_diffs_dists_com(map.all_agents, groups, t_idx)
    map.C_O_M = np.append(map.C_O_M,np.array([center_of_mass]),axis=0) if map.C_O_M.size else np.array([center_of_mass])
    target = target_point(map.C_O_M[0], time)
    for i, a in enumerate(map.all_agents):
        g_idx = np.where(groups == a.group)
        c_o_m = center_of_mass[g_idx]
        v_field, v_target, v_des = compute_v_des(i, t_idx, a, all_obstacles, diffs, dists, target, c_o_m,
                                                 v_max, r_rep, r_att, k_rep, k_att, k_target)
        v_field, v_target, v_des = v_field.reshape(2,1), v_target.reshape(2,1), v_des.reshape(2,1)
        p = (a.p[:,t] + a.v[:,t]*dt).reshape(2,1)
        p_target = target.reshape(2,1)
        a.p = np.hstack((a.p, p))
        a.p_target = np.hstack((a.p_target, p_target)) if a.p_target.size else p_target
        a.v = np.hstack((a.v, v_des))
        a.v_target = np.hstack((a.v_target, v_target)) if a.v_target.size else v_target
        a.v_field = np.hstack((a.v_field, v_field)) if a.v_field.size else v_field
        a.v_des = np.hstack((a.v_des, v_des)) if a.v_des.size else v_des

def compute_diffs_dists_com(agents, groups, t):
    positions = np.array([[a.p[0][t],a.p[1][t]] for a in agents])
    diffs = positions[None, :, :] - positions[:, None, :]   # (N, N, 2) diffs[i,j] = positions[j] - positions[i]
    dists = np.linalg.norm(diffs, axis=2)                   # (N, N)    dists[i,j] = dists[j,i],  dists[i,i] = 0
    center_of_mass = []
    for g in groups:
        agents_idx = [i for i in range(len(positions)) if agents[i].group == g]
        c_o_m_g = np.mean(positions[agents_idx], axis=0)
        center_of_mass.append(c_o_m_g)
    return diffs, dists, np.array(center_of_mass)

def compute_v_des(i, t_idx, agent_i, all_obstacles, diffs, dists, target, center_of_mass,
                  v_max, r_rep, r_att, k_rep, k_att, k_target):
    v_target = compute_v_target(k_target, target, center_of_mass)
    v_field = compute_v_field(i, agent_i, all_obstacles, t_idx, diffs, dists, r_rep, k_rep, r_att, k_att)
    v_des = v_field + v_target
    norm = np.linalg.norm(v_des)
    v_des_clipped = (v_des / norm) * np.minimum(norm, v_max) if norm else np.zeros(2)
    return(v_field,v_target,v_des_clipped)

def compute_v_target(k_target, target, center_of_mass):
    v_target = k_target * (target - center_of_mass)
    return v_target

def compute_v_field(i, agent_i, all_obstacles, t_idx, diffs, dists, r_rep, k_rep, r_att, k_att):
    v_rep = compute_v_rep(i, diffs, dists, r_rep, k_rep)
    v_att = compute_v_att(i, diffs, dists, r_att, k_att)
    v_obst = compute_v_obst(agent_i, all_obstacles, t_idx, r_rep, k_rep)
    v_field = v_rep + v_att + v_obst
    return v_field

def compute_v_rep(i, diffs, dists, r_rep, k_rep):
    v_rep = np.zeros(2)
    for j in range(len(dists)):
        if i == j:
            continue
        diff = diffs[i,j]
        dist = dists[i,j]
        if 0 < dist < r_rep:
            v_rep += k_rep * (dist - r_rep) * (diff/dist)
    return v_rep

def compute_v_att(i, diffs, dists, r_att, k_att):
    v_att = np.zeros(2)
    n = len(dists)
    for j in range(n):
        if i == j:
            continue
        diff = diffs[i,j]
        dist = dists[i,j]
        if dist > r_att:
            v_att += k_att/n * (dist - r_att) * (diff/dist)
    return v_att

def compute_v_frict(i, v_i, all_agents, dists, t, r_align, k_align):
    mask = (0 < dists[i]) & (dists[i] < r_align)
    if not np.any(mask):
        return np.zeros(2)

    v_others = np.array([[a.v[0][t], a.v[1][t]] for a in all_agents])[mask]
    v_mean = np.mean(v_others, axis=0)
    v_frict = k_align * (v_mean - v_i)
    return v_frict

def compute_v_obst(agent, obstacles, t, r_rep, k_rep):
    v_rep = np.zeros(2)
    x, y = agent.p[0][t], agent.p[1][t]
    for o in obstacles:
        closest_x = np.clip(x, o.x_min, o.x_max) # !!! WARNING !!! Only works if rectangle is aligned with map
        closest_y = np.clip(y, o.y_min, o.y_max) # !!! WARNING !!! Only works if rectangle is aligned with map
        diff = [closest_x - x, closest_y - y]
        dist = np.linalg.norm(diff)
        if 0 < dist < r_rep:
            v_rep += 1.5 * k_rep * (dist - r_rep) * (diff/dist) # 2x stronger than v_rep ag-ag because obs don't move
    return v_rep