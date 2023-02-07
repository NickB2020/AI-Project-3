#Nick Bischoff
#AI
#Project 3 HMM for car tracking part 2
#This program utilizes the HMM alogrithm, Part 2 Inferencing where a moving car
#is using transition probabilities. The case where the other car is moving
#according to transition probabilities.
#This takes in through command line argument, a  file
#for the sensor readings. E.G. stationaryCarReading10.csv The N(grid-world size)
#is then determined fro this file. and t or the length of the observation
#is taken in through command line argument as well.
#This part of the program will updates the posterior probability about the
#location of the car at a current time t based on the readings from time 1 to
#time t. Then save the map to a file (pMap_atTime20.csv).
#The probability map indicates the posterior
#probabilities of the car at a location.

from scipy.stats import norm
import pandas as pd
import sys
import math

##read in stationCarReading file and time command line args
cVal = pd.read_csv(sys.argv[1])
move = pd.read_csv(sys.argv[2])
time = int(sys.argv[3])

#pre: need to normalize the values
#post: can now normalize the values with this function
def normalize(printArr, N):
    total = 0
    for i in range(N):
        for j in range(N):
            total = total + printArr[i][j]
        for i in range(N):
            for j in range(N):
                printArr[i][j] = printArr [i] [j] / total

#pre: need to print the values as they come in to the array
#post: we can now print values from the array
def display(printArr, N):
    print("Showing Values")
    xPosition = 0
    yPosition = 0
    for i in range(N):
        for j in range(N):
            if (printArr[i][j] >= printArr[xPosition][yPosition]):
                xPosition = i
                yPosition = j
            print("x: ", xPosition)
            print("y: ", yPosition)
            print("value: ", printArr[xPosition][yPosition])

#pre: need to calculate the transiton probability
#post: we can now calculatethe transition probabilty with this function
def probility(i, j, N):
    north = (i -1+N)%N
    east = (j+1+N) %N
    south = (i+1+N)%N
    west = (j-1+N) %N
    return ((printArr[north][j] * move.loc[((north*N)+j)].S)
        + (printArr[i][east] * move.loc[((i*N)+east)].W)
        + (printArr[south][j] * move.loc[((south*N)+j)].N)
        + (printArr[i][west] * move.loc[((i*N)+west)].E))

#pre: need to locate position
#post: we can now locate the position with this function calculating the location
def location(X, Y, distance, N, J):
    for i in range(N):
        for j in range(N):
            if (J==0):
                printArr[i][j] =norm.pdf(distance, math.sqrt((i-X)**2) * printArr [i][j], 2/3)
            else:
                printArr[i][j] = probility(i, j, N) * norm.pdf(distance, math.sqrt((i-X)**2 + (j - Y) **2), 2/3)

printData = cVal.values.tolist()
N = cVal.get('gridSize')[0]
printArr = [[1/N**2] * N for _ in range(N)]

for J in range(time):
    xValue = printData[J][0]
    yValue = printData[J][1]
    distance = printData[J][2]
    location(xValue, yValue, distance, N, J)
    normalize(printArr, N)
display(printArr, N)
finalResult = pd.DataFrame(printArr)
finalResult.to_csv('pMap_atTime'+str(time)+'.csv')
print(printArr)
