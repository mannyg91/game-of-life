import pygame as pg
import numpy as np
from numpy import random
from pygame.locals import *
from sys import exit
from numba import njit

def run():

    class Game():
        def __init__(self):
            pg.init()
            self.running = True
            self.paused = True
            self.FPS = 60
            self.updateRate = 15
            self.clock = pg.time.Clock()
            displayInfo = pg.display.Info()
            self.sWidth = displayInfo.current_w
            self.sHeight = displayInfo.current_h
            flags = FULLSCREEN | DOUBLEBUF
            self.screen = pg.display.set_mode((self.sWidth, self.sHeight), flags, pg.FULLSCREEN, vsync=1)
            pg.display.set_caption("Conway's Game of Life")
            self.bg = pg.image.load('assets/bgGame7.jpg').convert()
            self.fontMed = pg.font.Font('assets/square_pixel-7.ttf', 27)
            self.fontLarge = pg.font.Font('assets/square_pixel-7.ttf', 32)
            self.xOffset = 0
            self.yOffset = 0

        def gameAudio(self):
            pg.mixer.init()
            pg.mixer.music.load("assets/paradigm.mp3")
            self.volume = .5
            self.muted = False
            pg.mixer.music.set_volume(self.volume)
            pg.mixer.music.play()

        def game_loop(self):
            loop()

        def pause(self, paused: bool):
            if paused:
                self.paused = True
                self.screen.blit(game.fontLarge.render("PAUSED", 1, (0,255,0)), (game.sWidth//2 - 50,game.sHeight - 160))
                pg.mixer.music.pause()
            else:
                self.paused = False
                pg.mixer.music.unpause()
        
        def gameMenu(self):
            mouseX, mouseY = pg.mouse.get_pos()
            if mouseY > game.sHeight * .75:
                menuOpen = True
                gameBar.render(alpha)
                playBtn.render(alpha)
                if alpha < 180:
                    alpha += 60
            else:
                menuOpen = False
                gameBar.render(alpha)
                playBtn.render(alpha)
                if alpha > 0:
                    alpha -= 60
            
        def events(self):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                if event.type == pg.KEYDOWN:
                    match event.key:
                        case pg.K_SPACE:
                            if self.paused:
                                game.pause(False)
                            else:
                                game.pause(True)
                        case pg.K_1: self.updateRate = 60
                        case pg.K_2: self.updateRate = 30
                        case pg.K_3: self.updateRate = 15
                        case pg.K_4: self.updateRate = 10
                        case pg.K_5: self.updateRate = 8
                        case pg.K_6: self.updateRate = 5
                        case pg.K_7: self.updateRate = 3
                        case pg.K_8: self.updateRate = 2
                        case pg.K_9: self.updateRate = 1
                        case pg.K_LEFTBRACKET:
                            gMatrix.currentMatrix = np.rot90(gMatrix.currentMatrix,1)
                            gMatrix.angle += 1
                            if gMatrix.angle > 4:
                                gMatrix.angle = 1
                        case pg.K_RIGHTBRACKET:
                            gMatrix.currentMatrix = np.rot90(gMatrix.currentMatrix,3)
                            gMatrix.angle -= 1
                            if gMatrix.angle < 1:
                                gMatrix.angle = 4
                        case pg.K_m:
                            if self.muted:
                                self.muted = False
                            else:
                                self.muted = True
                        case pg.K_p: print(Matrix.getPatterns(gMatrix))
                        case pg.K_r:
                            Matrix.randomNoise(gMatrix)
                        case pg.K_ESCAPE:
                            pg.quit()
                            exit()

            keys = pg.key.get_pressed()

            yLimit = gMatrix.rows * gMatrix.currentCellSize
            xLimit = gMatrix.cols * gMatrix.currentCellSize


            if keys[pg.K_MINUS]:
                if gMatrix.currentCellSize >= (gMatrix.minCellSize + 6):
                    gMatrix.currentCellSize -= 2
                    gMatrix.renderGrid()
            if keys[pg.K_EQUALS]:
                if gMatrix.currentCellSize <= gMatrix.maxCellSize:
                    gMatrix.currentCellSize += 2
                    gMatrix.renderGrid()
            if keys[pg.K_UP]:
                if self.yOffset <= yLimit:
                    self.yOffset += gMatrix.currentCellSize * 2
                    gMatrix.renderGrid()
            if keys[pg.K_RIGHT]:
                if self.xOffset >= -xLimit:
                    self.xOffset -= gMatrix.currentCellSize * 2
                    gMatrix.renderGrid()
            if keys[pg.K_LEFT]:
                if self.xOffset <= xLimit:
                    self.xOffset += gMatrix.currentCellSize * 2
                    gMatrix.renderGrid()
            if keys[pg.K_DOWN]:
                if self.yOffset >= -yLimit:
                    self.yOffset -= gMatrix.currentCellSize * 2
                    gMatrix.renderGrid()

    game = Game()



    class Btn():
        def __init__(self, imgPath: str, imgHoverPath: str, transparency: bool, scale, x, y, function = 0, alpha = 255):
            if transparency == True:
                img = pg.image.load(imgPath).convert_alpha()
                img.set_alpha(alpha)
                imgHover = pg.image.load(imgHoverPath).convert_alpha()
                imgHover.set_alpha(alpha)
            else:
                img = pg.image.load(imgPath).convert()
                imgHover = pg.image.load(imgHoverPath).convert()
            width = img.get_width()
            height = img.get_height()
            self.img = pg.transform.scale(img, (int(scale*width), int(scale*height)))
            self.imgHover = pg.transform.scale(imgHover, (int(scale*width), int(scale*height)))
            self.rect = self.img.get_rect()
            self.rect.topleft = (x,y)
            self.funct = function

        def render(self, transparency):
            mousePos = pg.mouse.get_pos()
            self.img.set_alpha(transparency)
            self.imgHover.set_alpha(transparency)
            if self.rect.collidepoint(mousePos):
                self.imgRender = self.imgHover
                if pg.mouse.get_pressed()[0]:
                    self.funct()
            else:
                self.imgRender = self.img
            game.screen.blit(self.imgRender, (self.rect.x, self.rect.y))

    gameBar = Btn('assets/gameBar.png', 'assets/gameBar.png', True, 1, 0, game.sHeight - 100)
    playBtn = Btn('assets/play.png', 'assets/playLit.png', True, 1, game.sWidth//2 - 14, game.sHeight - 63)


    class Matrix:
        def __init__(self, startCellSize, minCellSize, maxCellSize):
            self.minCellSize = minCellSize
            self.currentCellSize = startCellSize
            self.maxCellSize = maxCellSize
            self.rows = game.sHeight // minCellSize     
            self.cols = game.sHeight // minCellSize
            self.indices = self.allIndices()
            self.emptyMatrix = self.createMatrix()
            self.currentMatrix = self.emptyMatrix
            self.angle = 1

        def createMatrix(self):
            return np.zeros([self.rows,self.cols], dtype = int)

        def renderGrid(self):
            cellSize = self.currentCellSize
            rows = self.rows
            cols = self.cols
            tile = pg.transform.scale(tileObj.img, (cellSize, cellSize))
            gridSurf.blit(game.bg,(0,0)) 
            if cols > rows:
                for col in range(cols):
                    for row in range(rows): 
                        # gridSurf.blit(tile, ((col*1*cellSize//2)+(row*-1*cellSize//2)-(cellSize//2)+(game.sWidth//2) + game.xOffset,(col*.5*cellSize//2)+(row*.5*cellSize//2) + game.yOffset))
                        gridSurf.blit(tile, ((col*1*cellSize//2)+(row*-1*cellSize//2)-(cellSize//2)+(game.sWidth//2) + game.xOffset,(col*.5*cellSize//2)+(row*.5*cellSize//2) + game.yOffset))
            else:
                for row in range(cols):
                    for col in range(rows):
                        gridSurf.blit(tile, ((col*1*cellSize//2)+(row*-1*cellSize//2)-(cellSize//2)+(game.sWidth//2) + game.xOffset,(col*.5*cellSize//2)+(row*.5*cellSize//2) + game.yOffset))

        @staticmethod
        @njit
        def updateMatrix(matrix,newMatrix,newCells,indices):
            for index in indices:
                row = index[0]
                col = index[1]
                top = max(0, row-1)
                left = max(0, col-1)
                aliveCount = np.sum(matrix[top:row+2,left:col+2]) - matrix[row,col]
                if aliveCount == 2: 
                    newMatrix[row,col] = matrix[row,col]
                elif aliveCount == 3:
                    newMatrix[row,col] = 1
                    if matrix[row,col] == 0:
                        newCells[row,col] = 1
                else:
                    newMatrix[row,col] = 0
            return newMatrix
        
        
        def renderCells(self):
            cellSize = self.currentCellSize
            picture = pg.transform.scale(cellObj.img, (cellSize-1, cellSize-1))
            aliveIndices = np.nonzero(self.currentMatrix)
            # newIndices = np.nonzero(newCells)
            for row, col in zip(*aliveIndices):
                if self.currentMatrix[row,col] == 1:
                    game.screen.blit(picture, ((col*cellSize//2)+(row*-cellSize//2)-(cellSize//2)+(game.sWidth//2) + game.xOffset,(col*.5*cellSize//2)+(row*.5*cellSize//2) + game.yOffset))

        def allIndices(self):
            indices = np.ndindex(self.rows, self.cols)
            placeIndices = []
            for index in indices:
                placeIndices.append(index)
            self.indices = np.array(placeIndices)
            return self.indices

        def randomNoise(self):
            row = random.randint(self.rows - 2)
            col = random.randint(self.cols - 2)
            self.currentMatrix[row-1,col-1] = True
            self.currentMatrix[row-1,col] = True
            self.currentMatrix[row,col] = True
            self.currentMatrix[row,col+1] = True
            self.currentMatrix[row+1,col] = True

        def getPatterns(self):
            return np.array(np.where(self))

        def loadPatternOld(self, indexArr):
            for index in indexArr:
                self.currentMatrix[index[0],index[1]] = True

        def loadPattern(self, pattern):
            for row, col in zip(*pattern):
                self.currentMatrix[row,col] = True
            return
                
    class gameObject:
        def __init__(self, color, imgPath: str, transparency: bool, alpha = 255):
            self.color = color
            if transparency == True:
                self.img = pg.image.load(imgPath).convert_alpha()
                self.img.set_alpha(alpha)
            else:
                self.img = pg.image.load(imgPath).convert()

    cellObj = gameObject((255,255,255),'assets/cube.png',True,250)
    tileObj = gameObject((255,255,255), 'assets/tile.png',True,100)


    gMatrix = Matrix(50, 5, 500)
    rPentomino = np.array([[10,10],[10,11],[11,9],[11,10],[12,10]])
    gMatrix.loadPatternOld(rPentomino)

    gridSurf = pg.Surface((game.sWidth,game.sHeight))


    # =====LOAD PATTERNS HERE=============================
    # need to create another function to load a batch of patterns, not just one at a time
    # elevener = [[34,34,35,35,36,37,37,37,38,39,39],[33,34,32,34,32,30,31,32,29,29,30]]
    # pulsar = [[46,46,46,46,46,46,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,51,51,53,53,53,53,53,53,54,54,54,54,55,55,55,55,56,56,56,56,58,58,58,58,58,58],[84,85,86,90,91,92,82,87,89,94,82,87,89,94,82,87,89,94,84,85,86,90,91,92,84,85,86,90,91,92,82,87,89,94,82,87,89,94,82,87,89,94,84,85,86,90,91,92]]
    # octagon2= [[12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19],[75,76,74,77,73,78,72,79,72,79,73,78,74,77,75,76]]
    # figureEight = [[80,80,80,81,81,81,82,82,82,83,83,83,84,84,84,85,85,85],[151,152,153,151,152,153,151,152,153,154,155,156,154,155,156,154,155,156]]
    pentadecathlon = [[34,34,35,35,35,35,35,35,35,35,36,36],[150,155,148,149,151,152,153,154,156,157,150,155]]
    methuselah = [[76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76],[102,103,104,105,108,109,110,111,115,116,117,118,119,122,123,124,125,126,129,130,131,134,135,136,137,138,139]]
    # gliderGun = [[1,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,8,8,9,9],[27,25,27,15,16,23,24,37,38,14,18,23,24,37,38,3,4,13,19,23,24,3,4,13,17,19,20,25,27,13,19,27,14,18,15,16]]

    # gMatrix.loadPattern(pulsar)
    # gMatrix.loadPattern(octagon2)
    # gMatrix.loadPattern(figureEight)
    # gMatrix.loadPattern(elevener)
    gMatrix.loadPattern(pentadecathlon)
    gMatrix.loadPattern(methuselah)
    # gMatrix.loadPattern(gliderGun)






    gMatrix.renderGrid()
    game.gameAudio()
    game.paused = True
    menuOpen = False


    def loop(count = 0):
        while game.running:
            game.events()
            count += 1
            game.screen.blit(gridSurf,(0,0))

            if count % game.updateRate == 0:
                newCells = np.zeros([gMatrix.rows,gMatrix.cols], dtype = int)
                if not game.paused:
                    # randomNoise(matrix)
                    newMatrix = np.array(gMatrix.currentMatrix)
                    newMatrix = gMatrix.updateMatrix(gMatrix.currentMatrix, newMatrix, newCells, gMatrix.indices)
                    gMatrix.currentMatrix = newMatrix
            if game.paused:
                game.pause(True)
                

            gMatrix.renderCells()
            

            population = str(np.count_nonzero(gMatrix.currentMatrix))
            game.screen.blit(game.fontMed.render("Population: " + population, 1, (0,255,0)), (35,10))

            pg.display.flip()
            game.clock.tick(game.FPS)

    game.game_loop()
