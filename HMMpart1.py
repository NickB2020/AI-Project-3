#Nick Bischoff
#AI
#Project 3 HMM for car tracking part 1
#This program utilizes the HMM alogrithm, Part 1 locates a stationary car using
#emission probabilities. This takes in through command line argument, a  file
#for the sensor readings. E.G. stationaryCarReading10.csv The N(grid-world size)
#is then determined from this file. and t or the length of the observation
#is taken in through command line argument as well.
#This part of the program will output the probability map at time t after seeing
#the readings from time 1 to time t (pMap_atTime20.csv).
#The probability map indicates the posterior
#probabilities of the car at a location.

import math
import pandas as pd
from scipy.stats import norm
import sys
import numpy as np
import csv
#read in stationCarReading file and time command line args
fileName = pd.read_csv(sys.argv[1])
time = int(sys.argv[2])
#variables and the arrays used
#(one array for load in and a new one for updated values)
bigest = 0
xMax = 0
yMax = 0
arr = fileName.values.tolist()
row = len(arr)
N = int(arr[1][3])
newArr = [[1]*N for _ in range(N)]
#load arrays with the grid and calculate
for i in range(time):
    xvalue = float(arr[i][0])
    yvalue = float(arr[i][1])
    for x in range(N):
        for y in range(N):
            newArr[x][y] = norm.pdf(float(arr[i][2]), math.sqrt((x - xvalue)**2 + (y- yvalue)**2), 2/3) * newArr[x][y]

#normalization
    normalize = 0
    for x in range (N):
        for y in range (N):
            normalize += newArr[x][y]
#update new arrary
    for x in range (N):
        for y in range(N):
            newArr[x][y] = newArr[x][y] / normalize
#print the locations
print(newArr)
for x in range (N):
    for y in range (N):
        if (newArr[x][y] > bigest):
            xMax = x
            yMax = y
            bigest = newArr[x][y]

print("\nBigest Number: ")
print(bigest)
print("\nCoordinates X and Y" )
print(xMax, yMax)
