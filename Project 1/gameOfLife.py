import pygame as pg
import numpy as np

from sys import exit
from numpy import random
from numba import njit



#turns on pygame
pg.init()

#creates display surface (game window)



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
randomColor = list(np.random.choice(range(50,256), size=3))
white = (255,255,255)
gridColor = (30,30,30)



#testfont
font = pg.font.Font('font/Pixeltype.ttf', 30)
# text_surface = test_font.render("Conway's Game of Life", False, 'White')


#for fps
def displayFPS():
	fps = str(int(clock.get_fps()))
	return font.render(fps, 1, pg.Color("coral"))

drawRate = 300
gameRate = 40

cellSize = 10

#determines number of cells
#1000 / 20 = 200 
rows = sWidth // cellSize     
cols = sHeight // cellSize


#pulls data
activeCellSurface = pg.Surface((cellSize-1, cellSize-1))
deadCellSurface = pg.Surface((cellSize-1, cellSize-1))
gridSurface = pg.Surface((sWidth,sHeight))

#draws rects onto surface
activeCell = pg.draw.rect(activeCellSurface, white, (0,0,cellSize-1,cellSize-1))
deadCell = pg.draw.rect(deadCellSurface, black, (0,0,cellSize-1,cellSize-1))



#creating dead matrix
def createMatrix(rows, cols):
    pg.draw.rect(gridSurface, gridColor, (0,0,sWidth,sHeight))
    for row in range(rows):
        for col in range(cols):
            pg.draw.rect(gridSurface, black, (row*cellSize,col*cellSize,cellSize-1,cellSize-1))

    return np.zeros([rows,cols], dtype = int)


matrix = createMatrix(rows,cols)
# for index, cell in np.ndenumerate(matrix):
#     row = index[0]
#     col = index[1]
#     screen.blit(deadCellSurface,(col*cellSize,row*cellSize))

#updating matrix
matrix[1,2] = True
matrix[2,3] = True
matrix[3,1] = True
matrix[3,2] = True
matrix[3,3] = True


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

#started work here
def updateBoard(matrix):
    newMatrix = np.array(matrix)
    
    for index, cell in np.ndenumerate(matrix):
        row = index[0]
        col = index[1]
        if cellActivate(matrix, row, col):
            if matrix[row,col] != 1:
                newMatrix[row,col] = 1
            screen.blit(activeCellSurface,(col*cellSize,row*cellSize))
        else: 
            if matrix[row,col] != 0:
                newMatrix[row,col] = 0
                screen.blit(deadCellSurface,(col*cellSize,row*cellSize)) 

    return newMatrix



def currentBoard(matrix):
    for index, cell in np.ndenumerate(matrix):
        row = index[0]
        col = index[1]
        if matrix[row,col] == 1:
            screen.blit(activeCellSurface,(col*cellSize,row*cellSize))

paused = True

#keeps window open
while True:
    #gets all the events
    for event in pg.event.get():
        #.quit is the x button on window
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                if paused == False:
                    paused = True
                else:
                    paused = False
            if event.key == pg.K_1:
                print("1 pressed")
                gameRate = 3
            if event.key == pg.K_2:
                print("2 pressed")
                gameRate = 10
            if event.key == pg.K_3:
                print("3 pressed")
                gameRate = 15
            if event.key == pg.K_4:
                print("4 pressed")
                gameRate = 25
            if event.key == pg.K_5:
                print("5 pressed")
                gameRate = 100
        



    #background color set
    # screen.fill(gridColor)
    screen.blit(gridSurface,(0,0))
    



    if paused == False:
        # randomNoise(matrix)
        newMatrix = updateBoard(matrix)
        matrix = newMatrix
        
    else:
        currentBoard(matrix)
       




    click = pg.mouse.get_pressed()
    mouseX, mouseY = pg.mouse.get_pos()
    # print(click, mousex, mousey)

    #draw
    if click[0] == True:
        paused = True
        rate = drawRate
        roundedX = mouseX // cellSize
        roundedY = mouseY // cellSize
        matrix[roundedY,roundedX] = True
    elif click[2] == True:
        paused = True
        rate = drawRate
        roundedX = mouseX // cellSize
        roundedY = mouseY // cellSize
        matrix[roundedY,roundedX] = False
    else:
        rate = gameRate






    screen.blit(displayFPS(), (10,0))
    #updates the display surface (same as .flip when no arguments passed)
    pg.display.update()
    #setting frame rate part 2, sets max frame rate


    clock.tick(rate)

