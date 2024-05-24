from vehicle import Driver
from controller import Camera
import numpy as np
import time

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep


camera = Camera("Astra")
camera.enable(sensorTimeStep)
print("start ...")
print(f"VOITURE VIOLETTE : Sampling_period = {camera.getSamplingPeriod}")
camera.setExposure(2)
camera.setFocalDistance(1)
r = camera.getImage()
print(f"R = {r}")