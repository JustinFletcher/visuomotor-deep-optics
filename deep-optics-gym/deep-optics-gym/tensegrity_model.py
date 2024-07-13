

#####
#
# This code uses the tnsgrt package. Please download the package using `pip install tnsgrt`
# https://pypi.org/project/tnsgrt/
# 
### Example usage
# # `mirror_weight` is optional, default 10, unit in kg.
# env = tnsgrt_model(mirror_weight = 100) 
# 
# # The output `state` is a 4x6 array, each row represents the state of one of the "mirrors" and contains the displacement in x,y,z and the normal vector (angle).
# # Defaul assumes all input is 0.
# state = env.state_output()
# 
# # `action` is the applied force and should be a 1x8 numpy array e.g. [1,1,1,1,1,1,1,1]. Model here is the user's ML model.
# action = model(state)
# 
# # `alpha` is optional, it is the pointing angle of the telescope. 0 is assumed to be straight up. 
# state = env.state_output(force = action, alpha = 0) 
# 
#####

import numpy as np
from tnsgrt.structure import Structure
from tnsgrt.stiffness import NodeConstraint

class tnsgrt_model:
  def __init__(self, mirror_weight = 10):
    # Assume mirror weight is in kg.
    
    # Define the location of the nodes
    self.nodes = np.array([[0,0,0.5],
                      [0,0,-0.5],
                      [1,1,0.1],
                      [1,1,-0.1],
                      [-1,1,0.1],
                      [-1,1,-0.1],
                      [-1,-1,0.1],
                      [-1,-1,-0.1],
                      [1,-1,0.1],
                      [1,-1,-0.1],
                      ]).transpose()

    # Define the node connections
    self.members = np.array([[0,2],
                        [0,4],
                        [0,6],
                        [0,8],
                        [1,3],
                        [1,5],
                        [1,7],
                        [1,9],
                        [2,3],
                        [4,5],
                        [6,7],
                        [8,9],
                        [2,4],
                        [4,6],
                        [6,8],
                        [8,2],
                        [3,5],
                        [5,7],
                        [7,9],
                        [9,3],                  
                        ]).transpose()

    
    self.number_of_strings = 8
    self.mirror_weight_force = mirror_weight*9.81
    
    
    self.s = Structure(self.nodes, self.members, number_of_strings=self.number_of_strings)
    self.s.set_node_properties([0,1], 'constraint', NodeConstraint())

    # Material property
    self.s.set_member_properties([i for i in range(0,19)],'modulus',2.1e11) # Pa
    self.s.set_member_properties([i for i in range(0,19)],'yld',620422000) # N/m2
    self.s.set_member_properties([i for i in range(0,19)],'density',7700) # kg/m3

    # Bar and string properties
    self.s.set_member_properties(self.s.get_members_by_tag('bar'),'radius',0.05)
    self.s.set_member_properties(self.s.get_members_by_tag('string'),'radius',0.01)
    self.s.update_member_properties()
    
    self.f_vector = -self.s.get_member_vectors()[:,:8] # Force direction along the string.


  # Input force expected to be a positive numpy array of size 1x8. Corresponds to the force applied to the 8 cables. Unit in Newtons.
  # alpha corresponds to the current pointing angle of the telescope, assume to be about X-Axis, default to be 0. Unit in Degrees.
  def state_output(self, force, alpha = 0):
  
    self.f_base = np.zeros((3,10),dtype="float32")  
    
    # Vectorize the input force
    input_force = force*self.f_vector
    self.f_base[:,2] = input_force[:,0]
    self.f_base[:,4] = input_force[:,1]
    self.f_base[:,6] = input_force[:,2]
    self.f_base[:,8] = input_force[:,3]
    self.f_base[:,3] = input_force[:,4]
    self.f_base[:,5] = input_force[:,5]
    self.f_base[:,7] = input_force[:,6]
    self.f_base[:,9] = input_force[:,7]   
    
    # Superposition/add the force due to the mirror weight.
    self.f_base[:,2] = self.f_base[:,2] + np.array([0, self.mirror_weight_force*np.sin(np.deg2rad(alpha)), self.mirror_weight_force*np.cos(np.deg2rad(alpha))]).transpose()
    self.f_base[:,4] = self.f_base[:,4] + np.array([0, self.mirror_weight_force*np.sin(np.deg2rad(alpha)), self.mirror_weight_force*np.cos(np.deg2rad(alpha))]).transpose()
    self.f_base[:,6] = self.f_base[:,6] + np.array([0, self.mirror_weight_force*np.sin(np.deg2rad(alpha)), self.mirror_weight_force*np.cos(np.deg2rad(alpha))]).transpose()
    self.f_base[:,8] = self.f_base[:,8] + np.array([0, self.mirror_weight_force*np.sin(np.deg2rad(alpha)), self.mirror_weight_force*np.cos(np.deg2rad(alpha))]).transpose()
    
    # Calculate dispalcement
    self.s.equilibrium(self.f_base)
    self.stiffness, _, _ = self.s.stiffness(storage='dense', apply_rigid_body_constraint=True)
    self.disp = self.stiffness.displacements(self.f_base)

    return self.pos_calc()


  # Method to extract the deflection values and calculate the normal vector
  def pos_calc(self):
    output = np.zeros([4,6])
    
    output[0,0:3] = self.disp[:,2]
    output[0,3:6] = (self.nodes[:,2]+self.disp[:,2]) - (self.nodes[:,3]+self.disp[:,3])
    
    output[1,0:3] = self.disp[:,4]
    output[1,3:6] = (self.nodes[:,4]+self.disp[:,4]) - (self.nodes[:,5]+self.disp[:,5])
    
    output[2,0:3] = self.disp[:,6]
    output[2,3:6] = (self.nodes[:,6]+self.disp[:,6]) - (self.nodes[:,7]+self.disp[:,7])
    
    output[3,0:3] = self.disp[:,8]
    output[3,3:6] = (self.nodes[:,8]+self.disp[:,8]) - (self.nodes[:,9]+self.disp[:,9])
    
    return output


  # Update the mirror weight
  def set_mirror_weight(self,mirror_weight):
    # Assume mirror weight is in kg.
    self.mirror_weight_force = mirror_weight * 9.81