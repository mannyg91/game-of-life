import pygame
import numpy as np
from sys import exit

#turns on pygame
pygame.init()

#creates display surface (game window)
width = 1000
height = 1000
screen = pygame.display.set_mode((width,height),pygame.RESIZABLE)
# screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
pygame.display.set_caption("Conway's Game of Life")


#setting frame part 1
clock = pygame.time.Clock()

#RGB colors
black = (0,0,0)
white = (255,255,255)
darkgreen = (0,25,0)

on = pygame.Surface([50,50])
off = pygame.Surface([50,50])

#testfont
test_font = pygame.font.Font('font/Pixeltype.ttf', 50)
# text_surface = test_font.render("Conway's Game of Life", False, 'White')


#creating dead matrix
rows = 100
cols = 100
matrix = np.zeros([rows, cols], dtype = int)


#updating matrix
# matrix[1,2] = True
# matrix[2,3] = True
# matrix[3,1] = True
# matrix[3,2] = True
# matrix[3,3] = True


# #r-pentomino pattern
# matrix[28,30] = True
# matrix[28,31] = True
# matrix[29,29] = True
# matrix[29,30] = True
# matrix[30,30] = True



# #r-pentomino pattern
# matrix[8,10] = True
# matrix[8,11] = True
# matrix[9,9] = True
# matrix[9,10] = True
# matrix[10,10] = True


def liveNeighborCount(matrix, row, col):
    top = max(0, row-1)
    left = max(0, col-1)
    return np.sum(matrix[top:row+2,left:col+2]) - matrix[row,col]


def cellActivate(matrix, row, col):
        aliveCount = liveNeighborCount(matrix, row, col)
        if aliveCount < 2 or aliveCount > 3:
            return False
        if aliveCount == 2: 
            return matrix[row,col]
        if aliveCount == 3:
            return True


def updateBoard(matrix):
    newMatrix = np.array(matrix)
    for index, cell in np.ndenumerate(matrix):
        row = index[0]
        col = index[1]
        if cellActivate(matrix, row, col):
            newMatrix[row,col] = 1
            # randomColor = list(np.random.choice(range(50,256), size=3))
            pygame.draw.rect(screen, white, (col*10,row*10,9,9))
            
        else:
            newMatrix[row,col] = 0
            pygame.draw.rect(screen, black, (col*10,row*10,9,9))

    # print(chr(27) + "[2J")
    # print(newMatrix)
    #pygame stuff
    return newMatrix



def currentBoard(matrix):
    for index, cell in np.ndenumerate(matrix):
        row = index[0]
        col = index[1]
        if matrix[row,col] == 1:
            pygame.draw.rect(screen, white, (col*10,row*10,9,9))
            
        else:
            pygame.draw.rect(screen, black, (col*10,row*10,9,9))

paused = True

#keeps window open
while True:
    #gets all the events
    for event in pygame.event.get():
        #.quit is the x button on window
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.KEYDOWN:
            print("key down")
            if event.key == pygame.K_SPACE:
                print("space down")
                if paused == False:
                    paused = True
                else:
                    paused = False





    #provides mouse position when moved
    # if event.type == pygame.MOUSEMOTION:
    #     print(event.pos)
    # #knows when mouse button is pressed down
    # if event.type == pygame.MOUSEBUTTONDOWN:
    #     paused = True
    #     print('mouse down')

    # if event.type == pygame.MOUSEBUTTONUP:
    #     paused = False
    #     print('mouse up')




    #background color set
    screen.fill(darkgreen)
    
    if paused == False:
        newMatrix = updateBoard(matrix)
        matrix = newMatrix
    else:
        currentBoard(matrix)


    # newMatrix = str(updateBoard(matrix))
    # text_surface = test_font.render(newMatrix, False, 'White')
    # screen.blit(text_surface,(300,50))



    click = pygame.mouse.get_pressed()
    mousex, mousey = pygame.mouse.get_pos()
    # print(click, mousex, mousey)

    if click[0] == True:
        paused = True
        roundedX = mousex // 10
        roundedY = mousey // 10
        matrix[roundedY,roundedX] = True
    # else:
    #     paused = False


    #updates the display surface (same as .flip when no arguments passed)
    pygame.display.update()
    #setting frame rate part 2, sets max frame rate
    clock.tick(60)

