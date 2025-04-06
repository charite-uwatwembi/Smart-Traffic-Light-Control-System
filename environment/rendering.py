import pygame
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import imageio
import time

# Constants
WINDOW_SIZE = 600
TRAFFIC_LIGHT_POS = (0, 0)
ROAD_WIDTH = 100
CAR_SIZE = 20
FRAME_RATE = 10

# Traffic Light Colors
LIGHT_STATES = {
    'NS_Green': [(0, 1, 0), (1, 0, 0)],  # North-South Green, East-West Red
    'EW_Green': [(1, 0, 0), (0, 1, 0)],  # East-West Green, North-South Red
    'All_Red': [(1, 0, 0), (1, 0, 0)]    # All Red (transition state)
}

# Vehicle Movements
CAR_MOVEMENTS = {
    'NS_Green': [(0, 1), (0, -1)],
    'EW_Green': [(1, 0), (-1, 0)],
    'All_Red': []
}

class TrafficSimulation:
    def __init__(self):
        self.light_state = 'NS_Green'
        self.frames = []
        
    def draw_road(self):
        """Draw the cross roads."""
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex2f(-ROAD_WIDTH, -WINDOW_SIZE)
        glVertex2f(ROAD_WIDTH, -WINDOW_SIZE)
        glVertex2f(ROAD_WIDTH, WINDOW_SIZE)
        glVertex2f(-ROAD_WIDTH, WINDOW_SIZE)
        
        glVertex2f(-WINDOW_SIZE, -ROAD_WIDTH)
        glVertex2f(WINDOW_SIZE, -ROAD_WIDTH)
        glVertex2f(WINDOW_SIZE, ROAD_WIDTH)
        glVertex2f(-WINDOW_SIZE, ROAD_WIDTH)
        glEnd()
    
    def draw_traffic_lights(self):
        """Draw traffic lights at the center."""
        glColor3f(*LIGHT_STATES[self.light_state][0])  # NS Light
        glBegin(GL_QUADS)
        glVertex2f(-10, 20)
        glVertex2f(10, 20)
        glVertex2f(10, 40)
        glVertex2f(-10, 40)
        glEnd()
        
        glColor3f(*LIGHT_STATES[self.light_state][1])  # EW Light
        glBegin(GL_QUADS)
        glVertex2f(-10, -40)
        glVertex2f(10, -40)
        glVertex2f(10, -20)
        glVertex2f(-10, -20)
        glEnd()
    
    def draw_cars(self):
        """Draw cars moving in the allowed direction."""
        glColor3f(0, 0, 1)
        for dx, dy in CAR_MOVEMENTS[self.light_state]:
            glBegin(GL_QUADS)
            glVertex2f(dx * ROAD_WIDTH - CAR_SIZE, dy * ROAD_WIDTH - CAR_SIZE)
            glVertex2f(dx * ROAD_WIDTH + CAR_SIZE, dy * ROAD_WIDTH - CAR_SIZE)
            glVertex2f(dx * ROAD_WIDTH + CAR_SIZE, dy * ROAD_WIDTH + CAR_SIZE)
            glVertex2f(dx * ROAD_WIDTH - CAR_SIZE, dy * ROAD_WIDTH + CAR_SIZE)
            glEnd()
    
    def render_frame(self):
        glClear(GL_COLOR_BUFFER_BIT)
        self.draw_road()
        self.draw_traffic_lights()
        self.draw_cars()
        glFlush()
        
        # Save frame for GIF
        pixels = glReadPixels(0, 0, WINDOW_SIZE, WINDOW_SIZE, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(WINDOW_SIZE, WINDOW_SIZE, 3)
        img = np.flipud(img)
        self.frames.append(img)
        
    def update_light(self):
        """Switch traffic light states."""
        if self.light_state == 'NS_Green':
            self.light_state = 'All_Red'
        elif self.light_state == 'All_Red':
            self.light_state = 'EW_Green'
        else:
            self.light_state = 'NS_Green'
        

#TODO: add some animations


    def run_simulation(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGB)
        glutInitWindowSize(WINDOW_SIZE, WINDOW_SIZE)
        glutCreateWindow(b"Traffic Light Simulation")
        glOrtho(-WINDOW_SIZE//2, WINDOW_SIZE//2, -WINDOW_SIZE//2, WINDOW_SIZE//2, -1, 1)
        
        for _ in range(10):
            self.render_frame()
            self.update_light()
            time.sleep(0.5)
        
        imageio.mimsave("traffic_simulation.gif", self.frames, duration=0.5)
        print("GIF saved as 'traffic_simulation.gif'")
        
if __name__ == "__main__":
    sim = TrafficSimulation()
    sim.run_simulation()
