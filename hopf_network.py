"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import numpy as np
import matplotlib
from sys import platform
# STUDENTS: disabled this below as it resulted in bad behavior of robot on mac
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
if platform!="darwin" : # linux
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv


class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing=1*2*np.pi,  # MUST EDIT
                omega_stance=1*2*np.pi, # MUST EDIT
                gait="TROT",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step 
                ground_clearance=0.05,  # foot swing height 
                ground_penetration=0.01,# foot stance penetration into ground 
                robot_height=0.25,      # in nominal case (standing) 
                des_step_len=0.04,      # desired step length
                gait_feedback=True,      # wether to output gait chosen or not
                ):
    
    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters 
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._gait_feedback = gait_feedback
    self._set_gait(gait) # Defines self.PHI : +- symmetrical 4x4 matrix of offsets from i to j

    # set oscillator initial conditions  
    self.X[0,:] = np.random.rand(4) * .1 # X[0,:] are the ri of each leg : current oscillator amplitude
    self.X[1,:] = self.PHI[0,:] # One choice of initialization but X[1,:] is just phase theta of each oscillator

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len
    

  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [TODO] update all coupling matrices
    """
    self.PHI_trot = np.zeros((4,4))
    self.PHI_walk = np.zeros((4,4))
    self.PHI_bound = np.zeros((4,4))
    self.PHI_pace = np.zeros((4,4))

    # Create actual coupling matrices for phi (w will be all identical set by self._couple)
    # From the definition of the different gait sequences from the course, we have the following values
    # We always define offset from 0 to 2*pi
    # NB: each diagonal is zero as phase offset with itself is zero !
    # FL(i) where i stands for define indice of that leg, to have quick reminde
    FR = 0
    FL = 1
    RR = 2
    RL = 3
    
    # Trot (diagonals in sync)
    # From FL to non same RL and FR 
    self.PHI_trot[FL,RL]=self.PHI_trot[FL,FR]=np.pi
    self.PHI_trot[RL,FL]=self.PHI_trot[FR,FL]=-np.pi
    # From RR to non same RL and FR
    self.PHI_trot[RR,FR]=self.PHI_trot[RR,RL]=np.pi
    self.PHI_trot[FR,RR]=self.PHI_trot[RL,RR]=-np.pi
    
    # PACE (each side in sync)
    # From FL to others
    self.PHI_pace[FL,FR]=self.PHI_pace[FL,RR]=np.pi
    self.PHI_pace[FR,FL]=self.PHI_pace[RR,FL]=-np.pi 
    # From RL to others
    self.PHI_pace[RL,FR]=self.PHI_pace[RL,RR]=np.pi
    self.PHI_pace[FR,RL]=self.PHI_pace[RR,RL]=-np.pi
    
    # BOUND (front in sync, back in sync side in sync)
    # From RL to others
    self.PHI_bound[RL,FL]=self.PHI_bound[RL,FR]=np.pi
    self.PHI_bound[FL,RL]=self.PHI_bound[FR,RL]=-np.pi 
    # From RR to others
    self.PHI_bound[RR,FL]=self.PHI_bound[RR,FR]=np.pi
    self.PHI_bound[FL,RR]=self.PHI_bound[FR,RR]=-np.pi
    
    # WALK (lateral sequence)
    # From each to the next with 2*pi/4 offset starting with FL
    self.PHI_walk[FL,RR]=self.PHI_walk[RR,FR]=self.PHI_walk[FR,RL]=np.pi/2
    self.PHI_walk[RR,FL]=self.PHI_walk[FR,RR]=self.PHI_walk[RL,FR]=-np.pi/2
    # Each being 2*pi/2
    self.PHI_walk[FL,FR]=self.PHI_walk[RR,RL]=np.pi
    self.PHI_walk[FR,FL]=self.PHI_walk[RL,RR]=-np.pi
    # Last one which is 2pi*3/4
    self.PHI_walk[FL,RL]=np.pi*3/2
    self.PHI_walk[RL,FL]=-np.pi*3/2
    
    if gait == "TROT":
      if self._gait_feedback: print('TROT')
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      if self._gait_feedback: print('PACE') 
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      if self._gait_feedback: print('BOUND')
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      if self._gait_feedback: print('WALK')
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations()
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    # knowing Foot Order is FR, FL, RR, RL 
#     x = np.zeros(4) # [TODO]
    x = -self._des_step_len*self.X[0,:]*np.cos(self.X[1,:]) # do directly element wise operation from vectors
    z = np.zeros(4) # [TODO]
    idx_sinsup = np.sin(self.X[1,:])>0
    z[idx_sinsup] = -self._robot_height+self._ground_clearance*np.sin(self.X[1,idx_sinsup])
    z[~idx_sinsup] = -self._robot_height+self._ground_penetration*np.sin(self.X[1,~idx_sinsup])

    return x, z
      
        
  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot = np.zeros((2,4))
    alpha = 50 
    
    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[:,i] # [TODO]
      # compute r_dot (Equation 6)
      r_dot = alpha*(self._mu-r**2)*r # [TODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      theta_dot = self._omega_swing if theta<=np.pi else self._omega_stance # [TODO]
      
      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        r_js = X[0,:]
        theta_js = X[1,:]
        w_ijs = self.PHI[i,:]
        sinns = theta_js - theta - w_ijs
        couple_vals = r_js*self._coupling_strength*np.sin(sinns)
        theta_dot += sum(couple_vals) # [TODO]
        # Debug information
#         print("")
#         print("r_js")
#         print(r_js)
#         print("theta_js")
#         print(theta_js)
#         print("theta")
#         print(theta)
#         print("w_ijs")
#         print(w_ijs)
#         print("sinns")
#         print(sinns)
#         print("couple_vals")
#         print(couple_vals)

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X = X + self._dt*X_dot # [TODO]
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)



if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      # record_video=True
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP)

  TEST_STEPS = int(10 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states


  ############## Sample Gains
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([2,0.5,0.5])
  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)


  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12) 
    # get desired foot positions from CPG 
    xs,zs = cpg.update()
    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    # q = 
    # dq = 

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_q = np.zeros(3) # [TODO] 
      # Add joint PD contribution to tau for leg i (Equation 4)
      tau += np.zeros(3) # [TODO] 

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        # [TODO] 
        # Get current foot velocity in leg frame (Equation 2)
        # [TODO] 
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau += np.zeros(3) # [TODO]

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 

    # [TODO] save any CPG or robot states



  ##################################################### 
  # PLOTS
  #####################################################
  # example
  # fig = plt.figure()
  # plt.plot(t,joint_pos[1,:], label='FR thigh')
  # plt.legend()
  # plt.show()