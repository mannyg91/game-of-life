import pygame as pg
import numpy as np
from sys import exit
from numpy import random
from numba import njit

#turns on pygame
pg.init()

#creates display surface (game window) can be used later for zooming in/out
# gWidth = 2000
# gHeight = 2000

sWidth = 1000
sHeight = 1000
screen = pg.display.set_mode((sWidth,sHeight),pg.RESIZABLE)
# screen = pg.display.set_mode((0,0),pg.FULLSCREEN)
pg.display.set_caption("Conway's Game of Life")

#setting frame part 1
clock = pg.time.Clock()

#RGB colors
black = (0,0,0)
cellColor = (255,255,255)
gridColor = (20,10,70)



font = pg.font.SysFont('arial', 30)


#for fps
def displayFPS():
	fps = str(int(clock.get_fps()))
	return font.render(fps, 1, (0,255,0))

drawRate = 300
gameRate = 40

cellSize = 5

#determines number of cells
#1000 / 20 = 200 
rows = sWidth // cellSize     
cols = sHeight // cellSize


# activeSurf = pg.Surface((cellSize-1, cellSize-1))
gridSurf = pg.Surface((sWidth,sHeight))
# activeCell = pg.draw.rect(activeSurf, cellColor, (0,0,cellSize-1,cellSize-1))




#creating dead matrix
def createMatrix(rows, cols):
    pg.draw.rect(gridSurf, gridColor, (0,0,sWidth,sHeight))
    for row in range(rows):
        for col in range(cols):
            pg.draw.rect(gridSurf, black, (row*cellSize,col*cellSize,cellSize-1,cellSize-1))
    return np.zeros([rows,cols], dtype = int)


matrix = createMatrix(rows,cols)

#updating matrix
# matrix[1,2] = True
# matrix[2,3] = True
# matrix[3,1] = True
# matrix[3,2] = True
# matrix[3,3] = True

#r-pentomino pattern
matrix[28,30] = True
matrix[28,31] = True
matrix[29,29] = True
matrix[29,30] = True
matrix[30,30] = True

# #r-pentomino pattern
# matrix[8,10] = True
# matrix[8,11] = True
# matrix[9,9] = True
# matrix[9,10] = True
# matrix[10,10] = True

@njit
def getPatterns(matrix):
    return np.array(np.where(matrix == True))

@njit
def liveNeighborCount(matrix, row, col):
    top = max(0, row-1)
    left = max(0, col-1)
    return np.sum(matrix[top:row+2,left:col+2]) - matrix[row,col]

@njit
def cellActivate(matrix, row, col):
        aliveCount = liveNeighborCount(matrix, row, col)
        if aliveCount == 2: 
            return matrix[row,col]
        elif aliveCount == 3:
            return 1
        else: 
            return 0


def randomNoise(matrix):
    x = random.randint(150)
    y = random.randint(150)
    matrix[x,y] = True
    matrix[x+1,y+1] = True
    matrix[x,y+1] = True
    matrix[x+1,y] = True

def updateBoard(matrix):
    newMatrix = np.array(matrix)
    for index, cell in np.ndenumerate(matrix):
        row = index[0]
        col = index[1]
        if cellActivate(matrix, row, col):
            if matrix[row,col] != 1:
                newMatrix[row,col] = 1
            pg.draw.rect(screen,cellColor,(col*cellSize,row*cellSize,cellSize-1,cellSize-1))
        else: 
            if matrix[row,col] != 0:
                newMatrix[row,col] = 0
    return newMatrix



def currentBoard(matrix):
    for index, cell in np.ndenumerate(matrix):
        row = index[0]
        col = index[1]
        if matrix[row,col] == 1:
            pg.draw.rect(screen,cellColor,(col*cellSize,row*cellSize,cellSize-1,cellSize-1))


paused = True

while 1:
    #gets all the events
    for event in pg.event.get():
        #.quit is the x button on window
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.KEYDOWN:
            match event.key:
                case pg.K_SPACE:
                    if paused:
                        paused = False
                    else:
                        paused
                case pg.K_1:
                    gameRate = 3
                case pg.K_2:
                    gameRate = 10
                case pg.K_3:
                    gameRate = 15
                case pg.K_4:
                    gameRate = 25
                case pg.K_5:
                    gameRate = 75
                case pg.K_6:
                    gameRate = 300
       
      

    #background color set
    # screen.fill(gridColor)
    screen.blit(gridSurf,(0,0))
    

    if not paused:
        # randomNoise(matrix)
        newMatrix = updateBoard(matrix)
        matrix = newMatrix
        
    else:
        currentBoard(matrix)
       

    click = pg.mouse.get_pressed()
    mouseX, mouseY = pg.mouse.get_pos()
    # print(click, mousex, mousey)

    #draw
    if click[0]:
        paused = True
        rate = drawRate
        roundedX = mouseX // cellSize
        roundedY = mouseY // cellSize
        matrix[roundedY,roundedX] = True
    elif click[2]:
        paused = True
        rate = drawRate
        roundedX = mouseX // cellSize
        roundedY = mouseY // cellSize
        matrix[roundedY,roundedX] = False
    else:
        rate = gameRate


    screen.blit(displayFPS(), (10,0))
    pg.display.update()
    clock.tick(rate)
