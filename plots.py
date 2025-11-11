import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

MAX_FRAMES = 200

def compute_swarm_center(agents,t,group=-1):
    group_centers = []
    groups = [group] if group != -1 else np.unique([a.group for a in agents])
    for group in groups:
        group_agents = [a for a in agents if a.group == 0]
        x_coords = [a.p[0][t] for a in group_agents]
        y_coords = [a.p[1][t] for a in group_agents]
        group_centers.append([np.mean(x_coords),np.mean(y_coords)])
    return(group_centers)

def plot_map(map, rectangle=None, t_idx=0):
    agents, obstacles = map.all_agents, map.all_obstacles
    if rectangle is None:
        x_min, y_min, x_max, y_max = map.x_min, map.y_min, map.x_max, map.y_max
    else:
        x_min, y_min, x_max, y_max = rectangle

    fig, ax = plt.subplots(figsize=(6, 6))

    for a in agents:
        circle = patches.Circle(
            (a.p[0][t_idx], a.p[1][t_idx]),
            radius=a.r,
            facecolor='blue',
            edgecolor='k',
            alpha=0.6
        )
        ax.add_patch(circle)

    for o in obstacles:
        rect = patches.Rectangle(
            (o.x_min, o.y_min), # bottom left
            o.x_max - o.x_min,  # width
            o.y_max - o.y_min,  # height
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'    # transparent inside
        )
        ax.add_patch(rect)

    swarm_centers = compute_swarm_center(agents,t_idx)

    for c in swarm_centers:
        circle = patches.Circle(
                (c[0], c[1]), 
                radius=2*a.r, 
                color='green', 
                alpha=0.6)
        ax.add_patch(circle)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Agent Map with Obstacles and Swarm Centers")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    plt.show()

def animate_map(map, frame_min=0, frame_max=-1, rectangle=None, interval=None, downscale=True):
    agents, obstacles, interval = map.all_agents, map.all_obstacles, map.dt
    if rectangle is None:
        x_min, y_min, x_max, y_max = map.x_min, map.y_min, map.x_max, map.y_max
    else:
        x_min, y_min, x_max, y_max = rectangle
    if frame_max == -1:
        frame_max = len(agents[0].p[0])
    n_frames = frame_max - frame_min
    if downscale and n_frames > MAX_FRAMES:
       scaling = (n_frames // MAX_FRAMES)
       n_frames = MAX_FRAMES
    else:
        scaling = 1

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    agents_circles = []
    for a in agents:
        circle = patches.Circle(
                (a.p[0][frame_min], a.p[1][frame_min]), 
                radius=a.r, 
                color='blue', 
                alpha=0.6)
        agents_circles.append(circle)
        ax.add_patch(circle)

    for o in obstacles:
        rect = patches.Rectangle(
            (o.x_min, o.y_min), # bottom left
            o.x_max - o.x_min,  # width
            o.y_max - o.y_min,  # height
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'    # transparent inside
        )
        ax.add_patch(rect)

    swarm_centers = compute_swarm_center(agents,frame_min)
    center_circles = []
    for c in swarm_centers:
        circle = patches.Circle(
                (c[0], c[1]), 
                radius=2*a.r, 
                color='green', 
                alpha=0.6)
        center_circles.append(circle)
        ax.add_patch(circle)

    def update(frame):
        for i, c in enumerate(agents_circles):
            c.center = (agents[i].p[0][frame*scaling], agents[i].p[1][frame*scaling])
        swarm_centers = compute_swarm_center(agents,frame*scaling)
        for i, c in enumerate(center_circles):
            c.center = (swarm_centers[i][0], swarm_centers[i][1])
        return agents_circles + center_circles
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Agent Map with Obstacles and Swarm Centers")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000*interval*scaling, blit=False)
    return(anim)