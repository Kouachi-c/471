# Python imports
import random

# Webots imports
from controller import Supervisor

# Global constants
PI = 3.141592653589793
RECEIVER_SAMPLING_PERIOD = 64	# In milliseconds

# /!\ Set the number of sparring partner cars
NB_SPARRINGPARTNER_CARS = 3

# Clipping function
def value_clip(x, low, up):
	return low if x < low else up if x > up else x

# Angle normalization function (Webots angle values range between -pi and pi)
def angle_clip(x):
	a = x % (2*PI)
	return a if a < PI else a - 2*PI

# Positions intiales aléatoires 
# (plusieurs positions initiales auxquelles s'ajoute un peu d'aléatoire)
# 	[[x_min, x_max], [y_min, y_max], rotation]
# Les angles correspondents au même sens de rotation pour les voitures


# Train track
starting_positions = [
 	[[ 0.00,  4.00], [ 5.10,  5.30],  0.00],
 	[[ 4.95,  5.15], [ 0.30,  4.00], -PI/2],
 	[[ 2.70,  4.50], [-1.25, -0.75],  PI  ],
 	[[ 3.20,  3.45], [ 2.00,  2.40],  PI/2],
 	[[ 1.90,  2.70], [ 3.10,  3.30],  PI  ],
 	[[ 0.00,  0.25], [-0.50,  1.50], -PI/2],
 	[[-2.20, -1.90], [-0.50,  2.30],  PI/2]
]

"""
# Test track
starting_positions = [
	[[ 0.60,  4.20], [ 5.40,  5.60],  0.00],
	[[ 5.40,  5.60], [-4.30,  4.30], -PI/2],
	[[ 1.40,  1.60], [-5.00, -4.30],  PI/2],
	[[ 3.40,  3.60], [-2.85, -1.45],  PI/2],
	[[ 3.40,  3.60], [ 2.20,  2.50],  PI/2],
	[[-0.60, -0.40], [ 0.30,  0.90], -PI/2],
	[[-3.20, -2.00], [-4.60, -4.40],  PI],
	[[-2.60, -2.40], [-0.80,  2.30],  PI/2],
]
"""

# Initialisation
supervisor = Supervisor()
basicTimeStep = int(supervisor.getBasicTimeStep())
print("basic time step : "+str(basicTimeStep) + "ms")

# Receiver et emitter
receiver = supervisor.getDevice("receiver")
receiver.enable(RECEIVER_SAMPLING_PERIOD)
emitter = supervisor.getDevice("emitter")
packet_number = 0

# Recuperation des liens vers les noeuds voitures
tt_02 = supervisor.getFromDef("TT02_2023b_RL")
tt_02_translation = tt_02.getField("translation")
tt_02_rotation = tt_02.getField("rotation")
sparringpartner_car_nodes = [supervisor.getFromDef(f"sparringpartner_car_{i}") for i in range(NB_SPARRINGPARTNER_CARS)]
sparringpartner_car_translation_fields = [sparringpartner_car_nodes[i].getField("translation") for i in range(NB_SPARRINGPARTNER_CARS)]
sparringpartner_car_rotation_fields = [sparringpartner_car_nodes[i].getField("rotation") for i in range(NB_SPARRINGPARTNER_CARS)]

erreur_position = 0
# Main loop
while supervisor.step(basicTimeStep) != -1:
	# detection de positions incoherentes. Ne devrait pas servir...
	if abs(tt_02_translation.getSFVec3f()[0]) > 30 or abs(tt_02_translation.getSFVec3f()[1]) > 30 or abs(tt_02_translation.getSFVec3f()[2]) > 10 \
	or ((abs(tt_02_rotation.getSFVec3f()[3]) > PI/2) and ((abs(tt_02_rotation.getSFVec3f()[0]) > 0.5) or (abs(tt_02_rotation.getSFVec3f()[1]) > 0.5))): 
            	start_rot = [0.0, 0.0, 1.0, -PI/2]
            	tt_02_rotation.setSFRotation(start_rot)
            	tt_02.setVelocity([0,0,0,0,0,0])
            	tt_02_translation.setSFVec3f([2.98, 0.34, 0.039]) #devant un mur
            	erreur_position += 1
            	supervisor.step(basicTimeStep)
            	print("erreur position numero : ",erreur_position)
	
	# If reset signal : replace the cars
	if receiver.getQueueLength() > 0:
		# Get the data off the queue
		try :
			data = receiver.getString()
			receiver.nextPacket()
			print(data)
		except :
			print("souci de réception")
		
		# Choose driving direction
		direction = random.choice([0, 1])
		# Select random starting points
		coordinates_idx = random.sample(range(len(starting_positions)), 1+NB_SPARRINGPARTNER_CARS)

		# Replace TT-02 avec une position pseudo aleatoire
		# print("replacement de la voiture violette")
		coords = starting_positions[coordinates_idx[0]]
		start_x = random.uniform(coords[0][0], coords[0][1])
		start_y = random.uniform(coords[1][0], coords[1][1])
		start_z = 0.039
		
		angle = coords[2]
		start_angle = random.uniform(angle - PI/12, angle + PI/12)
		
		if direction:
			start_angle = start_angle + PI
		start_rot = [0.0, 0.0, 1.0, angle_clip(start_angle)]
		tt_02_rotation.setSFRotation(start_rot)		
		tt_02_translation.setSFVec3f([start_x, start_y, start_z])
		tt_02.setVelocity([0,0,0,0,0,0])		
		
		packet_number += 1
		  
		# Replace sparring partner cars
		for i in range(NB_SPARRINGPARTNER_CARS):
			sparringpartner_car_nodes[i].setVelocity([0,0,0,0,0,0])
			coords = starting_positions[coordinates_idx[i + 1]]
			start_x = random.uniform(coords[0][0], coords[0][1])
			start_y = random.uniform(coords[1][0], coords[1][1])
			start_z = 0.039
			sparringpartner_car_translation_fields[i].setSFVec3f([start_x, start_y, start_z])
			# Rotate sparringpartner cars
			angle = coords[2]
			start_angle = random.uniform(angle - PI/12, angle + PI/12)
			if direction:
				start_angle = start_angle + PI
			start_rot = [0, 0, 1, angle_clip(start_angle)]
			sparringpartner_car_rotation_fields[i].setSFRotation(start_rot)
			sparringpartner_car_nodes[i].setVelocity([0,0,0,0,0,0])
		
		#attente pour que les voitures se stabilisent dans la position de depart
		# for i in range(20) :
            		# supervisor.step(basicTimeStep)	
		supervisor.step(basicTimeStep)
		emitter.send("voiture replacee num : " + str(packet_number))
