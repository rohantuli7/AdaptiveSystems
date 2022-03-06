import random
import sys
# path folder which contains situsim_v1_2
sys.path.insert(1, '..')
sys.path.insert(1, '../situsim_extensions')
from situsim_v1_2 import *
import pygame
import matplotlib.pyplot as plt
import time
import pandas as pd
import copy as cp

from situsim_extensions.plots2 import *
from situsim_extensions.disturbances import *

# A subclass of Controller, which implements Braitenberg's aggressor (i.e. lightseeker)
class AdaptiveController(Controller):

    # init controller with passed in noisemakers and control parameters
    def __init__(self, genotype, gain1=1, gain2 = 1, left_bias1 = 0, right_bias1 = 0, left_bias2 = 0, right_bias2 = 0, wait_time = 2, be_adaptive = True):
        # NOTE: THIS CALL TO SUPER MUST BE HERE FOR NOISYCONTROLLERS!
        super().__init__() # call NoisyController.__init__() to set up noisemakers
        self.left_input1_weight = 1
        self.right_input0_weight = 1
        self.left_input0_weight = 0
        self.right_input1_weight = 0
        self.genotype = genotype
        self.t = 0
        self.adapting = False
        self.was_adapting = [self.adapting]
        self.wait_time = wait_time
        self.been_waiting = 0
        self.be_adaptive = be_adaptive
        self.gain1 = gain1
        self.gain2 = gain2
        self.left_bias1 = left_bias1
        self.right_bias1 = right_bias1
        self.left_bias2 = left_bias2
        self.right_bias2 = right_bias2
        self.left_gain_1 = 1
        self.left_gain_2 = 1
        self.right_gain_1 = 1
        self.right_gain_2 = 1
        self.robot = None

    # step method. depending on the values of speed and ratio, the robot will drive along a circular path
    #   - but noise will be added to the control outputs, so the robot might not achieve its goal!
    def step(self, inputs, dt):
        if self.be_adaptive:
            self.t += dt

            if self.adapting:
                self.been_waiting += dt

                if self.been_waiting > self.wait_time:
                    self.been_waiting = 0
                    self.adapting = False

            else:
                l = len(self.inputs)
                if l > 1:
                    l_sensor_change1 = (self.inputs[l - 1][0] - self.inputs[l - 2][0])
                    r_sensor_change1 = (self.inputs[l - 1][1] - self.inputs[l - 2][1])

                    if (l_sensor_change1 < 0.5) or (r_sensor_change1 < 0.5):

                        self.adapting = True
                        self.left_input1_weight = self.genotype[0]["values"][self.genotype[0]["ind"]]
                        self.right_input0_weight = self.genotype[1]["values"][self.genotype[1]["ind"]]
                        self.left_input0_weight = self.genotype[2]["values"][self.genotype[2]["ind"]]
                        self.right_input1_weight = self.genotype[3]["values"][self.genotype[3]["ind"]]
                        self.left_gain_1 = self.genotype[4]["values"][self.genotype[4]["ind"]]
                        self.left_gain_2 = self.genotype[5]["values"][self.genotype[5]["ind"]]
                        self.right_gain_1 = self.genotype[6]["values"][self.genotype[4]["ind"]]
                        self.right_gain_2 = self.genotype[7]["values"][self.genotype[5]["ind"]]
                        self.left_bias1 = self.genotype[8]["values"][self.genotype[6]["ind"]]
                        self.right_bias1 = self.genotype[9]["values"][self.genotype[7]["ind"]]
                        self.left_bias2 = self.genotype[10]["values"][self.genotype[8]["ind"]]
                        self.right_bias2 = self.genotype[11]["values"][self.genotype[9]["ind"]]

        self.left_speed_command = (self.left_input1_weight*inputs[1]*self.left_gain_1 + self.left_input0_weight*inputs[0]*self.left_gain_2 + self.left_bias1 - self.left_bias2)
        self.right_speed_command = (self.right_input0_weight*inputs[0]*self.right_gain_1 + self.right_input1_weight*inputs[1]*self.right_gain_2 + self.right_bias1 - self.right_bias2)
        self.was_adapting.append(self.adapting)

        return super().step(inputs, dt)

# set up the pygame window, if we are animating the simulation
def setup_pygame_window(screen_width):
    # initialise pygame and set parameters
    pygame.init()
    screen = pygame.display.set_mode([screen_width, screen_width])
    pygame.display.set_caption("Motor Inversion")
    # scale factor and offsets for converting simulation coordinates to pygame animation display coordinates
    pygame_scale = 30
    pygame_x_offset = screen_width/2
    pygame_y_offset = screen_width/2

    return screen

# draw SituSim systems in pygame window
def pygame_drawsim(screen, systems, width, paused, delay):

    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_UP:
                delay -= 1
            elif event.key == pygame.K_DOWN:
                delay += 1

    delay = np.max([delay, 0])

    time.sleep(delay/100)

    screen.fill('black')

    # initial scale factor and offsets for converting simulation coordinates
    # to pygame animation display coordinates
    pygame_x_offset = width/2
    pygame_y_offset = width/2

    # find extremes of system trajectories for resizing animation window
    max_xs = []
    max_ys = []
    for system in systems:
        if system.has_position:
            max_xs.append(max(np.abs(system.xs)))
            max_ys.append(max(np.abs(system.ys)))

    # reset scale according to where systems are and have been
    pygame_scale = width / (2 * max(max(max_xs), max(max_ys)) + 1)

    # draw all systems
    for system in systems:
        system.pygame_draw(screen, scale=pygame_scale, shiftx=pygame_x_offset, shifty=pygame_y_offset)

    # flip the pygame display
    screen.blit(pygame.transform.flip(screen, False, True), (0, 0))
    # update the pygame display
    pygame.display.update()

    return running, paused, delay

# get a noisemaker. used here for robot's motors, but could also be used for its
# sensors or controllers. this is only for convenience - for all except the
# BrownNoiseSource, it would be better to have more control over the parameters
def get_noisemaker(noise_type, noise_param):
    noisemaker = None
    if noise_param > 0:
        if noise_type == 'brown':
            noisemaker = BrownNoiseSource(noise_param)
        elif noise_type == 'white':
            noisemaker = WhiteNoiseSource(min_val=-noise_param, max_val=noise_param)
        elif noise_type == 'spike':
            noisemaker = SpikeNoiseSource(prob=0.05, pos_size=noise_param, neg_size=-noise_param)
    return noisemaker

# main function, to run simulation and generate plots
def run_simulation_once(screen_width,
                        controller,
                        animate=False,
                        field_of_view=0.9*np.pi,
                        left_sensor_angle=np.pi/4,
                        right_sensor_angle=-np.pi/4,
                        duration=60,
                        left_motor_noise=0.1,
                        right_motor_noise=0.1,
                        noise_type='brown',
                        disturb_times=[]
                        ):

    # get noisemakers for robot's motors
    left_motor_noisemaker = get_noisemaker(noise_type, left_motor_noise)
    right_motor_noisemaker = get_noisemaker(noise_type, right_motor_noise)

    # set up light sources
    light_sources = [LightSource(x=0, y=0, brightness=100)]

    # robots are always started from a position and orientation where they can see the light
    # it might be better if they started from all sides of the light, but this is slightly easier and roughly the
    # same
    x = -10
    y = -10
    #y = random_in_interval(-10, -7)
    #y = random_in_interval(-7, 7)
    theta = random_in_interval(-np.pi/2, np.pi/2)  # at least one sensor should always have the light in view from here

    # construct the robot
    robot = Robot(x=x, y=y, theta=theta,
                  controller=controller,
                  field_of_view=field_of_view,
                  left_light_sources=light_sources,
                  right_light_sources=light_sources,
                  left_sensor_angle=left_sensor_angle,
                  right_sensor_angle=right_sensor_angle,
                  left_motor_inertia=0,
                  right_motor_inertia=0,
                  left_motor_noisemaker=left_motor_noisemaker,
                  right_motor_noisemaker=right_motor_noisemaker
                  )
    controller.robot = robot
    # unlike some other disturbances, the SensoryInversionDisturbanceSource is a one-shot disturbance;
    # it doesn't happen repeatedly in a specified interval. for this reason, it is only passed start_times, and no
    # end_times
    # at each start_time, the connections between sensors and motors are effectively swapped, turning a
    # light-seeking robot to a light-avoiding one, or vice versa
    disturb = bool(disturb_times) # only disturb if the times list is not empty
    if disturb:
        disturbance = InvertMotors(robot, start_times=disturb_times)  # disturb once, half-way through
    # create list of agents - even though we only have one here, I always code
    # using a list, as it makes it easy to add more agents
    agents = [robot]

    # only run pygame code if animating the simulation
    if animate:
        screen = setup_pygame_window(screen_width)

    # animation variables
    delay = 0 # can be used to slow animation down
    running = True # can be used to exit animation early
    paused = False # can be used to pause simulation/animation

    # prepare simulation time variables
    t = 0
    ts = [t]
    dt = 0.1
    # begin simulation main loop
    while t < duration and running:

        # only move simulation forwards in time if not paused
        if not paused:

            # step disturbance
            if disturb:
                disturbance.step(dt)

            # step all robots
            for agent in agents:
                agent.step(dt)

            # increment time variable and store in ts list for plotting later
            t += dt
            ts.append(t)

        # only run pygame code if animating the simulation
        if animate:
            running, paused, delay = pygame_drawsim(screen, agents + light_sources, screen_width, paused, delay)
    # simulation has completed

    # only run pygame code if animating the simulation
    if animate:
        # Quit pygame.
        pygame.display.quit()
        pygame.quit()

    return ts, agents, light_sources

# plot outputs for all robots
# - note: these are not all of the outputs which we can plot,
# but they are the ones which will always be available,
# and which we will probably look at the most
def do_plots(all_ts, agents, light_sources, filepath):

    # parameters for plots
    plt.rcParams["font.weight"] = "bold"
    font_size = 18

    plot_all_agents_trajectories(all_ts, agents, light_sources, draw_agents=False, filepath=filepath)
    plot_all_robots_motors(all_ts, agents, filepath=filepath)
    plot_all_robots_controllers(all_ts, agents, filepath=filepath)
    plot_all_robots_sensors(all_ts, agents, filepath=filepath)

    plt.show()

'''

select controller and parameters and run simulation

'''
def run_sim(genotype, runs=1, animate=True, disturb_times=[], adapt = False,left_motor_noise=0.1, right_motor_noise=0.1, noise_type='brown'):
    # set noise levels for controller outputs
    left_noise = 0
    right_noise = 0

    all_robots = []
    all_ts = []

    # run the simulation the specified number of times
    for i in range(runs):

        # NOTE: ALL SYSTEMS ARE CREATED INSIDE THIS LOOP, OR IN FUNCTIONS CALLED BY IT
        # - if we created a system outside of this loop, e.g. one of the noisemakers,
        # then it would be used in every run

        field_of_view = 0.9*np.pi
        left_sensor_angle = np.pi/4
        right_sensor_angle = -np.pi/4

        # create a controller object to pass to the robot
        controller = AdaptiveController(genotype=genotype, be_adaptive=adapt)

        # if you uncomment this line, only the first run of the simulation will be animated
        animate = animate and i == 0

        duration = 200 # longer for monocular controller

        # use a copy of the disturb_times for every run
        # - this is required due to what *seemed like* a good idea when I first
        #   programmed the DisturbanceSource class - times get popped from the lists
        #   as they are used, meaning the lists can only be used once (doh!)
        disturb_times2 = []
        for t in disturb_times:
            disturb_times2.append(t)

        # run the simulation once, with the given parameters
        ts, robots, light_sources = run_simulation_once(screen_width=700,
                                                        controller=controller,
                                                        animate=animate,
                                                        field_of_view=field_of_view,
                                                        left_sensor_angle=left_sensor_angle,
                                                        right_sensor_angle=right_sensor_angle,
                                                        duration=duration,
                                                        left_motor_noise=left_motor_noise,
                                                        right_motor_noise=right_motor_noise,
                                                        noise_type=noise_type,
                                                        disturb_times=disturb_times2
                                                        )

        all_robots = all_robots + robots
        all_ts.append(ts)

    #do_plots(all_ts, all_robots, light_sources)
    return evaluate_genotype(robots[-1]), all_ts, all_robots, light_sources

filepath = '/Users/rt/Desktop/College/M.Sc/Semester 2/Adaptive systems/Assignment1/data'

def mutate_gene(thing):
    ind = thing["ind"] + np.random.choice([-1, 1])
    if ind < 0:
        ind = thing["size"] - 1 # wrap around
    if ind == thing["size"]:
        ind = 0
    thing["ind"] = ind

def make_gene(values):
    return {"values" : values, "size" : len(values), "ind" : np.random.randint(low=0, high=len(values))}

def mutate_genotype(genotype):
    ind = np.random.choice(len(genotype))
    mutate_gene(genotype[ind])

def evaluate_genotype(robot):
    dists = np.sqrt(np.square(robot.xs) + np.square(robot.ys))
    return np.mean(np.abs(dists))

def create_discrete_genes():
    return [np.random.randint(-1, 1) for i in range(10)]

def create_continous_genes(start, end):
    return [np.random.uniform(start, end) for i in range(10)]

def print_genotype_vals(genotype):
    s = ''
    for i, gene in enumerate(genotype):
        vals = gene["values"]
        s += str(round(vals[gene["ind"]], 5))
        if i < gene["size"]:
             s += ', '
    return s

# g1_values = create_discrete_genes()
# g2_values = create_discrete_genes()
# g3_values = create_discrete_genes()
# g4_values = create_discrete_genes()
# g5_values = create_continous_genes(-2, 2)
# g6_values = create_continous_genes(-2, 2)
# g7_values = create_continous_genes(-2, 2)
# g8_values = create_continous_genes(-2, 2)
# g9_values = create_continous_genes(-1, 1)
# g10_values = create_continous_genes(-1, 1)
# g11_values = create_continous_genes(-1, 1)
# g12_values = create_continous_genes(-1, 1)
#
# g1 = make_gene(g1_values)
# g2 = make_gene(g2_values)
# g3 = make_gene(g3_values)
# g4 = make_gene(g4_values)
# g5 = make_gene(g5_values)
# g6 = make_gene(g6_values)
# g7 = make_gene(g7_values)
# g8 = make_gene(g8_values)
# g9 = make_gene(g9_values)
# g10 = make_gene(g10_values)
# g11 = make_gene(g11_values)
# g12 = make_gene(g12_values)
#
# all_ts_test = []
# all_robots_test = []
# all_light_sources_test = []
#
# genotype1 = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]
# cost, all_ts, all_robots, light_sources = run_sim(runs=1, animate=False, disturb_times=[20], adapt = True,
#                                                   left_motor_noise=0.5, right_motor_noise=0.5, noise_type='none', genotype=genotype1)
#
# costs = []
# genes = []
# costs.append(cost)
# genes.append(genotype1)
#
# all_ts_test.append(all_ts)
# all_robots_test.append(all_robots)
# all_light_sources_test.append(light_sources)
# epochs = 0
# # without noise
# while True:
#     if cost < 2 or epochs == 1000:
#         break
#     genotype2 = cp.deepcopy(genotype1)
#     mutate_genotype(genotype2)
#     animate = False
#     if epochs == 999:
#         animate = True
#     cost2, all_ts, all_robots, light_sources = run_sim(runs=1, animate=animate, disturb_times=[20], adapt = True,
#                                                        left_motor_noise=0.5, right_motor_noise=0.5, noise_type='none', genotype=genotype2)
#     costs.append(cost2)
#     genes.append(genotype2)
#     all_ts_test.append(all_ts)
#     all_robots_test.append(all_robots)
#     all_light_sources_test.append(light_sources)
#     if cost2 < cost:
#         cost = cost2
#         print(f"After {epochs} generation, cost : {cost}")
#         genotype1 = cp.deepcopy(genotype2)
#     epochs +=1
# print(f"Genotype after simulation without any noise : {print_genotype_vals(genotype1)}\nFitness : {cost}\n\n")
# cols = ["pass", "left_ip1", "right_ip0", "left_ip0", "right_ip1",
#         "left_gain1", "left_gain2", "right_gain1", "right_gain2",
#         "left_bias1", "right_bias1", "left_bias2", "right_bias2", "fitness"]
#
# temp = []
# for count, cost in enumerate(costs):
#     temp1 = []
#     temp1.append(count)
#     for c in range(len(cols)-2):
#         temp1.append(genes[count][c]["values"][genes[count][c]["ind"]])
#     temp1.append(cost)
#     temp.append(temp1)
#
# df = pd.DataFrame(temp, columns=cols)
# df.to_csv(filepath+'/tables/invertMotors/invert_motors.csv')
# do_plots(all_ts_test[-1], all_robots_test[-1], all_light_sources_test[-1], filepath+'/graphs/invertMotors/none')
#
#
#
#
#
# g1_values = create_discrete_genes()
# g2_values = create_discrete_genes()
# g3_values = create_discrete_genes()
# g4_values = create_discrete_genes()
# g5_values = create_continous_genes(-2, 2)
# g6_values = create_continous_genes(-2, 2)
# g7_values = create_continous_genes(-2, 2)
# g8_values = create_continous_genes(-2, 2)
# g9_values = create_continous_genes(-1, 1)
# g10_values = create_continous_genes(-1, 1)
# g11_values = create_continous_genes(-1, 1)
# g12_values = create_continous_genes(-1, 1)
#
# g1 = make_gene(g1_values)
# g2 = make_gene(g2_values)
# g3 = make_gene(g3_values)
# g4 = make_gene(g4_values)
# g5 = make_gene(g5_values)
# g6 = make_gene(g6_values)
# g7 = make_gene(g7_values)
# g8 = make_gene(g8_values)
# g9 = make_gene(g9_values)
# g10 = make_gene(g10_values)
# g11 = make_gene(g11_values)
# g12 = make_gene(g12_values)
#
# all_ts_test = []
# all_robots_test = []
# all_light_sources_test = []
#
# genotype1 = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]
# cost, all_ts, all_robots, light_sources = run_sim(runs=1, animate=False, disturb_times=[20], adapt = True,
#                                                   left_motor_noise=0.5, right_motor_noise=0.5, noise_type='none', genotype=genotype1)
#
# costs = []
# genes = []
# costs.append(cost)
# genes.append(genotype1)
#
# all_ts_test.append(all_ts)
# all_robots_test.append(all_robots)
# all_light_sources_test.append(light_sources)
# epochs = 0
# # without noise
# while True:
#     if cost < 2 or epochs == 1000:
#         break
#     genotype2 = cp.deepcopy(genotype1)
#     mutate_genotype(genotype2)
#     animate = False
#     if epochs == 999:
#         animate = True
#     cost2, all_ts, all_robots, light_sources = run_sim(runs=1, animate=animate, disturb_times=[20], adapt = True,
#                                                        left_motor_noise=0.5, right_motor_noise=0.5, noise_type='spike', genotype=genotype2)
#     costs.append(cost2)
#     genes.append(genotype2)
#     all_ts_test.append(all_ts)
#     all_robots_test.append(all_robots)
#     all_light_sources_test.append(light_sources)
#     if cost2 < cost:
#         cost = cost2
#         print(f"After {epochs} generation, cost : {cost}")
#         genotype1 = cp.deepcopy(genotype2)
#     epochs +=1
# print(f"Genotype after simulation without any noise : {print_genotype_vals(genotype1)}\nFitness : {cost}\n\n")
# cols = ["pass", "left_ip1", "right_ip0", "left_ip0", "right_ip1",
#         "left_gain1", "left_gain2", "right_gain1", "right_gain2",
#         "left_bias1", "right_bias1", "left_bias2", "right_bias2", "fitness"]
#
# temp = []
# for count, cost in enumerate(costs):
#     temp1 = []
#     temp1.append(count)
#     for c in range(len(cols)-2):
#         temp1.append(genes[count][c]["values"][genes[count][c]["ind"]])
#     temp1.append(cost)
#     temp.append(temp1)
#
# df = pd.DataFrame(temp, columns=cols)
# df.to_csv(filepath+'/tables/invertMotors/invert_motors_spike.csv')
# do_plots(all_ts_test[-1], all_robots_test[-1], all_light_sources_test[-1], filepath+'/graphs/invertMotors/spike')


g1_values = create_discrete_genes()
g2_values = create_discrete_genes()
g3_values = create_discrete_genes()
g4_values = create_discrete_genes()
g5_values = create_continous_genes(-2, 2)
g6_values = create_continous_genes(-2, 2)
g7_values = create_continous_genes(-2, 2)
g8_values = create_continous_genes(-2, 2)
g9_values = create_continous_genes(-1, 1)
g10_values = create_continous_genes(-1, 1)
g11_values = create_continous_genes(-1, 1)
g12_values = create_continous_genes(-1, 1)

g1 = make_gene(g1_values)
g2 = make_gene(g2_values)
g3 = make_gene(g3_values)
g4 = make_gene(g4_values)
g5 = make_gene(g5_values)
g6 = make_gene(g6_values)
g7 = make_gene(g7_values)
g8 = make_gene(g8_values)
g9 = make_gene(g9_values)
g10 = make_gene(g10_values)
g11 = make_gene(g11_values)
g12 = make_gene(g12_values)

all_ts_test = []
all_robots_test = []
all_light_sources_test = []

genotype1 = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]
cost, all_ts, all_robots, light_sources = run_sim(runs=1, animate=False, disturb_times=[20], adapt = True,
                                                  left_motor_noise=0.5, right_motor_noise=0.5, noise_type='none', genotype=genotype1)

costs = []
genes = []
costs.append(cost)
genes.append(genotype1)

all_ts_test.append(all_ts)
all_robots_test.append(all_robots)
all_light_sources_test.append(light_sources)
epochs = 0
# without noise
while True:
    if cost < 2 or epochs == 1000:
        break
    genotype2 = cp.deepcopy(genotype1)
    mutate_genotype(genotype2)
    animate = False
    if epochs == 999:
        animate = True
    cost2, all_ts, all_robots, light_sources = run_sim(runs=1, animate=animate, disturb_times=[20], adapt = True,
                                                       left_motor_noise=0.5, right_motor_noise=0.5, noise_type='white', genotype=genotype2)
    costs.append(cost2)
    genes.append(genotype2)
    all_ts_test.append(all_ts)
    all_robots_test.append(all_robots)
    all_light_sources_test.append(light_sources)
    if cost2 < cost:
        cost = cost2
        print(f"After {epochs} generation, cost : {cost}")
        genotype1 = cp.deepcopy(genotype2)
    epochs +=1
print(f"Genotype after simulation without any noise : {print_genotype_vals(genotype1)}\nFitness : {cost}\n\n")
cols = ["pass", "left_ip1", "right_ip0", "left_ip0", "right_ip1",
        "left_gain1", "left_gain2", "right_gain1", "right_gain2",
        "left_bias1", "right_bias1", "left_bias2", "right_bias2", "fitness"]

temp = []
for count, cost in enumerate(costs):
    temp1 = []
    temp1.append(count)
    for c in range(len(cols)-2):
        temp1.append(genes[count][c]["values"][genes[count][c]["ind"]])
    temp1.append(cost)
    temp.append(temp1)

df = pd.DataFrame(temp, columns=cols)
df.to_csv(filepath+'/tables/invertMotors/invert_motors_white.csv')
do_plots(all_ts_test[-1], all_robots_test[-1], all_light_sources_test[-1], filepath+'/graphs/invertMotors/white')