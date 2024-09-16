import numpy as np
import random
from math import sqrt, cos, sin, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global flag to enable/disable Barnes-Hut approximation
BARNESHUT = True  # Set to False to disable Barnes-Hut approximation

# Node class representing each node in the simulation
class Node:
    def __init__(self, index, radius, x=None, y=None, vx=0, vy=0, fx=None, fy=None):
        self.index = index
        self.radius = radius
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.fx = fx  # Fixed x position
        self.fy = fy  # Fixed y position

# Base class for all forces
class Force:
    def apply(self, nodes, alpha):
        pass

# Center force to center the nodes around a point
class CenterForce(Force):
    def __init__(self, x=0, y=0, strength=0.1):
        self.x = x
        self.y = y
        self.strength = strength

    def apply(self, nodes, alpha):
        n = len(nodes)
        sx = sum(node.x for node in nodes) / n
        sy = sum(node.y for node in nodes) / n
        dx = (sx - self.x) * self.strength * alpha
        dy = (sy - self.y) * self.strength * alpha
        for node in nodes:
            node.vx -= dx / n
            node.vy -= dy / n

# Soft Collision force to prevent node overlap
class CollideForce(Force):
    def __init__(self, radius_func, strength=0.7, iterations=2):
        self.radius_func = radius_func
        self.strength = strength
        self.iterations = iterations

    def apply(self, nodes, alpha):
        radii = [self.radius_func(node) for node in nodes]
        n = len(nodes)
        quad_tree = QuadTree(nodes)  # Use a quadtree for efficient collision detection
        for _ in range(self.iterations):
            for node in nodes:
                r = radii[node.index]
                neighbors = []
                # Find potential colliding nodes using the quadtree
                quad_tree.visit(lambda quad, x0, y0, x1, y1: self.find_neighbors(quad, node, r, neighbors))
                for neighbor in neighbors:
                    if neighbor is not node:
                        dx = node.x - neighbor.x
                        dy = node.y - neighbor.y
                        dist = sqrt(dx * dx + dy * dy) or 1e-6  # Avoid division by zero
                        min_dist = r + radii[neighbor.index]
                        if dist < min_dist:
                            # Nodes are overlapping; adjust positions
                            overlap = (min_dist - dist) / dist * self.strength * alpha
                            nx = dx * overlap * 0.5
                            ny = dy * overlap * 0.5
                            node.x += nx
                            node.y += ny
                            neighbor.x -= nx
                            neighbor.y -= ny

    def find_neighbors(self, quad, node, radius, neighbors):
        if quad.point is not None:
            point = quad.point
            while point:
                if point != node:
                    dx = point.x - node.x
                    dy = point.y - node.y
                    dist = abs(dx) + abs(dy)
                    if dist < (radius + point.radius):
                        neighbors.append(point)
                point = getattr(point, 'next', None)
        return quad.x0 > node.x + radius or quad.x1 < node.x - radius or quad.y0 > node.y + radius or quad.y1 < node.y - radius

# Quadtree classes for Barnes-Hut approximation
class QuadTree:
    def __init__(self, nodes):
        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        dx = (x1 - x0) * 0.1 if x1 - x0 > 0 else 1
        dy = (y1 - y0) * 0.1 if y1 - y0 > 0 else 1
        self.x0 = x0 - dx
        self.y0 = y0 - dy
        self.x1 = x1 + dx
        self.y1 = y1 + dy
        self.root = None
        for node in nodes:
            self.insert(node)

    def insert(self, node):
        if self.root is None:
            self.root = QuadTreeNode(self.x0, self.y0, self.x1, self.y1)
        self.root.insert(node)

    def visit(self, callback):
        if self.root:
            self.root.visit(callback)

    def visit_after(self, callback):
        if self.root:
            self.root.visit_after(callback)

class QuadTreeNode:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0  # Bounds
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.length = 0  # Number of children (internal node if length > 0)
        self.children = [None, None, None, None]  # Quadrants NW, NE, SW, SE
        self.point = None  # For leaf nodes
        self.next = None  # Linked list for points at the same position
        self.value = 0  # Accumulated strength
        self.x = 0  # Center of mass
        self.y = 0

    def insert(self, node):
        x = node.x
        y = node.y
        if self.length == 0:
            if self.point is None:
                # Empty leaf node, store the point
                self.point = node
                self.next = None
            else:
                # Leaf node with existing point
                if self.point.x == x and self.point.y == y:
                    # Same position, add to linked list
                    node.next = self.point
                    self.point = node
                else:
                    # Subdivide and re-insert points
                    self.subdivide()
                    self.insert_point(self.point)
                    self.point = None
                    self.insert_point(node)
        else:
            self.insert_point(node)

    def insert_point(self, node):
        x = node.x
        y = node.y
        sx = (self.x0 + self.x1) / 2
        sy = (self.y0 + self.y1) / 2
        index = 0
        if x >= sx: index += 1
        if y >= sy: index += 2
        if self.children[index] is None:
            x0 = sx if index % 2 else self.x0
            x1 = self.x1 if index % 2 else sx
            y0 = sy if index >= 2 else self.y0
            y1 = self.y1 if index >= 2 else sy
            self.children[index] = QuadTreeNode(x0, y0, x1, y1)
        self.children[index].insert(node)

    def subdivide(self):
        self.length = 4
        self.children = [None, None, None, None]

    def is_leaf(self):
        return self.length == 0

    def visit(self, callback):
        quads = [self]
        while quads:
            quad = quads.pop()
            if not callback(quad, quad.x0, quad.y0, quad.x1, quad.y1):
                if not quad.is_leaf():
                    quads.extend([child for child in quad.children if child])

    def visit_after(self, callback):
        quads = []
        nodes = [self]
        while nodes:
            quad = nodes.pop()
            if not quad.is_leaf():
                nodes.extend([child for child in quad.children if child])
            quads.append(quad)
        for quad in reversed(quads):
            callback(quad)

# ManyBodyForce class with BARNESHUT flag
class ManyBodyForce(Force):
    def __init__(self, strength=-30, distance_min=1, distance_max=float('inf'), theta=0.9):
        self.strength = strength  # Can be a constant or a function
        self.distance_min2 = distance_min * distance_min
        self.distance_max2 = distance_max * distance_max
        self.theta2 = theta * theta
        self.nodes = None
        self.strengths = None
        self.random = random.Random()
        self.alpha = None
        self.node = None  # Current node being processed

    def initialize(self, nodes):
        self.nodes = nodes
        n = len(nodes)
        self.strengths = [0] * n
        for i, node in enumerate(nodes):
            if callable(self.strength):
                self.strengths[i] = self.strength(node, i, nodes)
            else:
                self.strengths[i] = self.strength

    def apply(self, nodes, alpha):
        if self.nodes is None or self.nodes != nodes:
            self.initialize(nodes)
        self.alpha = alpha
        n = len(nodes)
        if BARNESHUT:
            # Build the quadtree
            quadtree = QuadTree(nodes)
            quadtree.visit_after(self.accumulate)
            for i in range(n):
                self.node = nodes[i]
                quadtree.visit(self.apply_force)
        else:
            # Direct computation between all pairs
            for i in range(n):
                node_i = nodes[i]
                xi = node_i.x
                yi = node_i.y
                si = self.strengths[i]
                for j in range(i + 1, n):
                    node_j = nodes[j]
                    x = node_j.x - xi
                    y = node_j.y - yi
                    l = x * x + y * y
                    if l < self.distance_max2:
                        if x == 0 and y == 0:
                            x = (self.random.random() - 0.5) * 1e-6
                            y = (self.random.random() - 0.5) * 1e-6
                            l = x * x + y * y
                        if l < self.distance_min2:
                            l = self.distance_min2
                        l = sqrt(l)
                        factor = (si * self.alpha) / l / l
                        node_i.vx += x * factor
                        node_i.vy += y * factor
                        factor = (self.strengths[j] * self.alpha) / l / l
                        node_j.vx -= x * factor
                        node_j.vy -= y * factor

    def accumulate(self, quad):
        strength = 0
        weight = 0
        if quad.is_leaf():
            # Leaf node
            q = quad
            q.x = q.point.x
            q.y = q.point.y
            point = q.point
            while point is not None:
                index = point.index
                strength += self.strengths[index]
                point = getattr(point, 'next', None)
        else:
            x = y = 0
            for child in quad.children:
                if child is not None and child.value:
                    c = abs(child.value)
                    strength += child.value
                    x += c * child.x
                    y += c * child.y
                    weight += c
            if weight > 0:
                quad.x = x / weight
                quad.y = y / weight
        quad.value = strength

    def apply_force(self, quad, x0, y0, x1, y1):
        if not quad.value:
            return True  # Skip this quad
        x = quad.x - self.node.x
        y = quad.y - self.node.y
        w = x1 - x0
        l = x * x + y * y

        # Apply the Barnes-Hut approximation if possible.
        if w * w / self.theta2 < l:
            if l < self.distance_max2:
                if x == 0 and y == 0:
                    x = (self.random.random() - 0.5) * 1e-6
                    y = (self.random.random() - 0.5) * 1e-6
                    l = x * x + y * y
                if l < self.distance_min2:
                    l = self.distance_min2
                factor = quad.value * self.alpha / l
                self.node.vx += x * factor
                self.node.vy += y * factor
            return True  # Don't recurse into children
        elif quad.is_leaf() or l >= self.distance_max2:
            return False  # Don't recurse into children
        else:
            # Continue traversal
            return False

    def set_strength(self, strength):
        self.strength = strength
        if self.nodes:
            self.initialize(self.nodes)
# LinkForce to create spring-like forces between nodes
class LinkForce(Force):
    def __init__(self, links,multiplier=1.0):
        """
        links: list of tuples (source_index, target_index, link_strength)
        distance: desired distance between nodes
        strength: base strength of the spring
        """
        self.multiplier=multiplier
        self.links = links

    def apply(self, nodes, alpha):
        for link in self.links:
            i, j, link_strength = link
            node_i = nodes[i]
            node_j = nodes[j]
            dx = node_j.x - node_i.x
            dy = node_j.y - node_i.y
            dist_sq = dx * dx + dy * dy
            dist = sqrt(dist_sq)
            if dist == 0:
                dx = (random.random() - 0.5) * 1e-6
                dy = (random.random() - 0.5) * 1e-6
                dist = sqrt(dx * dx + dy * dy)
            desired_dist = 0
            force = (dist - desired_dist) *self.multiplier * alpha / dist
            fx = dx * force
            fy = dy * force
            node_i.vx += fx
            node_i.vy += fy
            node_j.vx -= fx
            node_j.vy -= fy


# Simulation class
class Simulation:
    def __init__(self, nodes, alpha=1, alpha_min=0.001, alpha_decay=0.02, alpha_target=0, velocity_decay=0.6):
        self.nodes = nodes
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_target = alpha_target
        self.velocity_decay = velocity_decay
        self.forces = {}
        self.alpha_decay = alpha_decay
        self.initialize_nodes()

    def initialize_nodes(self):
        initial_radius = 10
        initial_angle = pi * (3 - sqrt(5))
        for i, node in enumerate(self.nodes):
            node.index = i
            if node.fx is not None:
                node.x = node.fx
            if node.fy is not None:
                node.y = node.fy
            if node.x is None or node.y is None:
                radius = initial_radius * sqrt(0.5 + i)
                angle = i * initial_angle
                node.x = radius * cos(angle)
                node.y = radius * sin(angle)
            if node.vx is None or node.vy is None:
                node.vx = 0
                node.vy = 0

    def tick(self):
        self.alpha += (self.alpha_target - self.alpha) * self.alpha_decay
        if self.alpha < self.alpha_min:
            self.alpha = self.alpha_min
        for force in self.forces.values():
            force.apply(self.nodes, self.alpha)
        for node in self.nodes:
            if node.fx is None:
                node.vx *= self.velocity_decay
                node.x += node.vx
            else:
                node.x = node.fx
                node.vx = 0
            if node.fy is None:
                node.vy *= self.velocity_decay
                node.y += node.vy
            else:
                node.y = node.fy
                node.vy = 0

    def add_force(self, name, force):
        self.forces[name] = force

    def remove_force(self, name):
        del self.forces[name]

    def set_nodes(self, nodes):
        self.nodes = nodes
        self.initialize_nodes()

    def run(self, steps=500):
        for _ in range(steps):
            if self.alpha < self.alpha_min:
                break
            self.tick()


def example():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    num_nodes = 200
    nodes = []
    for i in range(num_nodes):
        radius = random.uniform(5, 30)
        node = Node(index=i, radius=radius, x=random.uniform(-100, 100)*10, y=random.uniform(-100, 100)*10)
        nodes.append(node)

    # Adjusted simulation parameters
    simulation = Simulation(nodes, alpha_decay=0.001, velocity_decay=0.70)

    # Define a radius function for CollideForce
    def node_radius(node):
        return node.radius

    # User-provided links (edges) with strengths
    # For example, connect node 0 and 1 with strength 1, node 1 and 2 with strength 0.5, etc.
    links = []
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            s = random.random()
            if s<0.1:
                links.append((i,j,s))

    # Add forces with specified parameters
    center_force = CenterForce(x=0, y=0, strength=0.01)  # Very small center force
    collide_force = CollideForce(radius_func=node_radius, strength=1.0, iterations=5)
    link_force = LinkForce(links=links,multiplier=0.03/len(nodes))
    many_body_force = ManyBodyForce(strength=5,theta=0.90)

    # Add forces to the simulation
    simulation.add_force('center', center_force)
    simulation.add_force('collide', collide_force)
    simulation.add_force('link', link_force)
    simulation.add_force('charge', many_body_force)

    # Prepare for animation
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_aspect('equal')
    ax.axis('on')

    circles = []
    for node in nodes:
        circle = plt.Circle((node.x, node.y), node.radius, color='skyblue', ec='black', alpha=0.6)
        ax.add_artist(circle)
        circles.append(circle)

    DRAW_LINKS=0
    # Draw links
    lines = []
    if DRAW_LINKS:
        for link in links:
            i, j, _ = link
            line, = ax.plot([nodes[i].x, nodes[j].x], [nodes[i].y, nodes[j].y], 'k-', alpha=0.20)
            lines.append(line)

    # Function to update the frame
    def update(frame_num):
        # Perform one simulation step
        simulation.tick()
        # Update line positions
        if DRAW_LINKS:
            for idx, link in enumerate(links):
                i, j, _ = link
                lines[idx].set_data([nodes[i].x, nodes[j].x], [nodes[i].y, nodes[j].y])
        # Update circle positions
        for i, node in enumerate(simulation.nodes):
            circles[i].center = (node.x, node.y)
        return circles + lines

    # Create the animation
    ani = FuncAnimation(fig, update, frames=5000, interval=50, blit=False)
    plt.show()

example()

print('Finished!')
