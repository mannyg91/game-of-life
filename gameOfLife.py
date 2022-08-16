import os
import pygame as pg
import numpy as np
import time
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
            self.max_fps = 200
            self.fps_correction = 1
            self.updateDelay = 15
            self.zoomDelay = 1
            self.moveDelay = 1
            self.clock = pg.time.Clock()
            displayInfo = pg.display.Info()
            self.sWidth = displayInfo.current_w
            self.sHeight = displayInfo.current_h
            self.screen = pg.display.set_mode((self.sWidth, self.sHeight), flags = pg.FULLSCREEN | pg.DOUBLEBUF, vsync=1)
            pg.display.set_caption("Conway's Game of Life")
            self.bg = pg.image.load('assets/starbg2.png').convert()
            self.fontMed = pg.font.Font('assets/square_pixel-7.ttf', 27)
            self.fontLarge = pg.font.Font('assets/square_pixel-7.ttf', 32)
            self.xOffset, self.yOffset, self.count = 0, 0, 0
            self.gameAudio()
            self.funct = None

        def gameAudio(self):
            pg.mixer.init()
            pg.mixer.music.load("assets/paradigm.mp3")
            self.volume = 0
            self.muted = False
            pg.mixer.music.set_volume(self.volume)
            pg.mixer.music.play()

        def game_loop(self):
            loop()

        def pause(self, paused: True):
            if paused:
                self.paused = True
                self.screen.blit(self.fontLarge.render("PAUSED", 1, (0,255,0)), (self.sWidth//2 - 50,self.sHeight - 160))
                pg.mixer.music.pause()
            else:
                self.paused = False
                pg.mixer.music.unpause()

        def single_click_events(self):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                if event.type == pg.KEYDOWN:
                    match event.key:
                        case pg.K_SPACE:
                            if self.paused:
                                self.pause(False)
                            else:
                                self.pause(True)
                        case pg.K_1: self.updateDelay = 100
                        case pg.K_2: self.updateDelay = 50
                        case pg.K_3: self.updateDelay = 25
                        case pg.K_4: self.updateDelay = 13
                        case pg.K_5: self.updateDelay = 8
                        case pg.K_6: self.updateDelay = 5
                        case pg.K_7: self.updateDelay = 3
                        case pg.K_8: self.updateDelay = 2
                        case pg.K_9: self.updateDelay = 1
                        case pg.K_c:
                            self.xOffset = 0
                            self.yOffset = 0
                            gMatrix.renderGrid()
                        case pg.K_m:
                            if self.muted:
                                self.muted = False
                            else:
                                self.muted = True
                        case pg.K_p: print(Matrix.getPatterns(gMatrix))
                        case pg.K_r:
                            Matrix.randomNoise(gMatrix)
                        case pg.K_x:
                            gMatrix.currentMatrix = Matrix.createMatrix(gMatrix)
                        case pg.K_ESCAPE:
                            pg.quit()
                            exit()
                if event.type == MOUSEBUTTONDOWN:
                    print("mousebuttondown")
                    if self.funct:
                        self.funct()

        def key_hold_events(self):
            keys = pg.key.get_pressed()

            if self.count % self.zoomDelay == 0:

                if keys[pg.K_MINUS]:

                    if gMatrix.currentCellSize > gMatrix.minCellSize:

                        colCenter1 = ((self.sWidth / gMatrix.currentCellSize) / 2) - (self.xOffset / gMatrix.currentCellSize)
                        rowCenter1 = ((self.sHeight / gMatrix.currentCellSize) / 2) - (self.yOffset / gMatrix.currentCellSize)

                        gMatrix.currentCellSize -= 1

                        colCenter2 = ((self.sWidth / gMatrix.currentCellSize) / 2) - (self.xOffset / gMatrix.currentCellSize)
                        rowCenter2 = ((self.sHeight / gMatrix.currentCellSize) / 2) - (self.yOffset / gMatrix.currentCellSize)

                        differenceX = (colCenter1 - colCenter2) * gMatrix.currentCellSize
                        differenceY = (rowCenter1 - rowCenter2) * gMatrix.currentCellSize

                        self.xOffset -= differenceX
                        self.yOffset -= differenceY

                        gMatrix.renderGrid()

                if keys[pg.K_EQUALS]:

                    if gMatrix.currentCellSize < gMatrix.maxCellSize:

                        colCenter1 = ((self.sWidth / gMatrix.currentCellSize) / 2) - (self.xOffset / gMatrix.currentCellSize)
                        rowCenter1 = ((self.sHeight / gMatrix.currentCellSize) / 2) - (self.yOffset / gMatrix.currentCellSize)


                        gMatrix.currentCellSize += 1

                        colCenter2 = ((self.sWidth / gMatrix.currentCellSize) / 2) - (self.xOffset / gMatrix.currentCellSize)
                        rowCenter2 = ((self.sHeight / gMatrix.currentCellSize) / 2) - (self.yOffset / gMatrix.currentCellSize)

                        differenceX = (colCenter1 - colCenter2) * gMatrix.currentCellSize
                        differenceY = (rowCenter1 - rowCenter2) * gMatrix.currentCellSize
                        self.xOffset -= differenceX
                        self.yOffset -= differenceY

                        gMatrix.renderGrid()


                if keys[pg.K_UP]:
                    if self.yOffset < self.sHeight / 2:
                        self.yOffset += gMatrix.currentCellSize * game.fps_correction
                        gMatrix.renderGrid()

                if keys[pg.K_RIGHT]:
                    # if self.xOffset > gMatrix.xBoundary[gMatrix.currentCellSize]:
                    self.xOffset -= gMatrix.currentCellSize * game.fps_correction
                    gMatrix.renderGrid()

                if keys[pg.K_LEFT]:
                    if self.xOffset < self.sWidth / 2:
                        self.xOffset += gMatrix.currentCellSize * game.fps_correction
                        gMatrix.renderGrid()
                        
                if keys[pg.K_DOWN]:
                    # if self.yOffset > gMatrix.yBoundary[gMatrix.currentCellSize]:
                    self.yOffset -= gMatrix.currentCellSize * game.fps_correction
                    gMatrix.renderGrid()

        def mouse_events(self):
            click = pg.mouse.get_pressed()
            mouseX, mouseY = pg.mouse.get_pos()
            if click[0]:
                if mouseX >= self.xOffset and mouseY >= self.yOffset and mouseY < game.sHeight * .91:
                    try:
                        self.paused = True
                        roundedX = (mouseX - self.xOffset) / gMatrix.currentCellSize
                        roundedY = (mouseY - self.yOffset) / gMatrix.currentCellSize
                        gMatrix.currentMatrix[int(roundedY),int(roundedX)] = True
                    except:
                        print("out of range")
            elif click[2] and mouseX >= self.yOffset:
                if mouseX >= self.xOffset and mouseY >= self.yOffset:
                    try:
                        self.paused = True
                        roundedX = (mouseX - self.xOffset) / gMatrix.currentCellSize
                        roundedY = (mouseY - self.yOffset) / gMatrix.currentCellSize
                        gMatrix.currentMatrix[int(roundedY),int(roundedX)] = False
                    except:
                        print("out of range")

    game = Game()

    class Gui():
        def __init__(self):
            self.opaque_level = 0
            self.menuOpen = False
        def gameMenu(self):
            mouseX, mouseY = pg.mouse.get_pos()
            if mouseY > game.sHeight * .75:
                self.menuOpen = True
                if self.opaque_level < 245:
                    self.opaque_level += 5 * game.fps_correction
            else:
                self.menuOpen = False
                if self.opaque_level > 0:
                    self.opaque_level -= 10 * game.fps_correction
            gameBar.render(self.opaque_level)
            playBtn.render(self.opaque_level)
            speedSlider.render(self.opaque_level)
            speedSlider2.render(self.opaque_level)
            speedSliderBall.render(self.opaque_level)
            speedSliderBall2.render(self.opaque_level)

        def pauseSwitch():
            if game.paused:
                game.paused = False
            else:
                game.paused = True

    game_gui = Gui()


    class Btn():
        def __init__(self, imgPath: str, imgHoverPath: str, transparency: bool, scale, x, y, function, alpha = 255):
            if transparency == True:
                img = pg.image.load(imgPath).convert_alpha()
                img.set_alpha(alpha)
                imgHover = pg.image.load(imgHoverPath).convert_alpha()
                imgHover.set_alpha(alpha)
            else:
                img = pg.image.load(imgPath).convert()
                imgHover = pg.image.load(imgHoverPath).convert()
            self.width = img.get_width()
            self.height = img.get_height()
            self.img = pg.transform.scale(img, (int(scale*self.width), int(scale*self.height)))
            self.imgHover = pg.transform.scale(imgHover, (int(scale*self.width), int(scale*self.height)))
            self.rect = self.img.get_rect()
            self.rect.topleft = (x,y)
            self.btn_funct = function

        def render(self, transparency):
            mousePos = pg.mouse.get_pos()
            self.img.set_alpha(transparency)
            self.imgHover.set_alpha(transparency)
            if self.rect.collidepoint(mousePos):
                self.imgRender = self.imgHover
                game.funct = self.btn_funct
            else:
                self.imgRender = self.img
            game.screen.blit(self.imgRender, (self.rect.x, self.rect.y))

        def test():
            game.screen.blit(game.fontMed.render("test succeeded", 1, (0,255,0)), (550,300))


    gameBar = Btn('assets/gameBar.png', 'assets/gameBar.png', True, 1, 0, game.sHeight - 100, Btn.test)
    
    playBtn = Btn('assets/play.png', 'assets/playLit.png', True, 1, game.sWidth//2 - 14, game.sHeight - 63, Gui.pauseSwitch)
    playBtn2 = Btn('assets/play.png', 'assets/playLit.png', True, 1, game.sWidth//2 - 64, game.sHeight//2, Gui.pauseSwitch)
    speedIcon = Btn('assets/speed.png','assets/speed.png', True, 1, game.sWidth//2 + 50, game.sHeight//2, Gui.pauseSwitch)
    speedSlider = Btn('assets/slider.png','assets/slider.png', True, .25, game.sWidth//2 + 50, game.sHeight -52, Btn.test)
    speedSlider2 = Btn('assets/slider.png','assets/slider.png', True, .25, game.sWidth//2 - 250, game.sHeight-52, Btn.test)
    speedSliderBall = Btn('assets/sliderBall.png','assets/sliderBallLit.png', True, .3, game.sWidth//2 + 200, game.sHeight - 57, Btn.test)
    speedSliderBall2 = Btn('assets/sliderBall.png','assets/sliderBallLit.png', True, .3, game.sWidth//2 - 200, game.sHeight - 57, Btn.test)


    class Matrix:
        def __init__(self, startCellSize, minCellSize, maxCellSize):
            self.minCellSize = minCellSize
            self.currentCellSize = startCellSize
            self.initialCellSize = startCellSize
            self.maxCellSize = maxCellSize
            self.rows = game.sHeight // minCellSize    
            self.cols = game.sWidth // minCellSize
            self.indices = self.allIndices()
            self.emptyMatrix = self.createMatrix()
            self.currentMatrix = self.emptyMatrix
            self.angle = 1
            self.bgColor = [0,0,0]
            self.initGridColor = [120,15,195]
            self.cellColor = (255,255,255)
            self.xBoundary = self.cameraBounds(game.sWidth)
            self.yBoundary = self.cameraBounds(game.sHeight)
            self.gridShades = self.createGridShades(self.minCellSize, .965, self.initGridColor)
            self.bgSurf = pg.Surface((game.sWidth, game.sHeight))

        def createMatrix(self):
            return np.zeros([self.rows,self.cols], dtype = int)

        def createGridShades(self, minCellSize, change, colors)->dict:
            gridShades = {}
            for i in range(self.maxCellSize,minCellSize - 1, -1):
                r = colors[0] * change
                g = colors[1] * change
                b = colors[2] * change
                colors = [r,g,b]
                gridShades[i] = colors
            return gridShades

        def renderGrid(self):
            gridColor = self.gridShades[self.currentCellSize]
            self.bgSurf.blit(game.bg,(0,0)) 
            pg.draw.rect(self.bgSurf, gMatrix.bgColor, (game.xOffset,game.yOffset,self.currentCellSize * self.cols ,self.currentCellSize * self.rows))
            for row in range(self.rows):
                pg.draw.line(self.bgSurf, gridColor, (game.xOffset,row * self.currentCellSize + game.yOffset), (self.cols * self.currentCellSize + game.xOffset,row * self.currentCellSize + game.yOffset))
            for col in range(self.cols):
                pg.draw.line(self.bgSurf, gridColor, (col * self.currentCellSize + game.xOffset,game.yOffset), (col * self.currentCellSize + game.xOffset, self.rows * self.currentCellSize + game.yOffset))

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
            # picture = pg.transform.scale(cellObj.img, (cellSize-1, cellSize-1))
            aliveIndices = np.nonzero(self.currentMatrix)
            # newIndices = np.nonzero(newCells)
            for row, col in zip(*aliveIndices):
                if self.currentMatrix[row,col] == 1:
                    pg.draw.rect(game.screen,self.cellColor,(col * cellSize + game.xOffset, row * cellSize + game.yOffset,cellSize-1,cellSize-1))

        def allIndices(self):
            indices = np.ndindex(self.rows, self.cols)
            placeIndices = []
            for index in indices:
                placeIndices.append(index)
            self.indices = np.array(placeIndices)
            return self.indices

        def cameraBounds(self, dimension):
            leadingOffset = dimension // self.minCellSize
            offsets = {self.minCellSize : 0}
            for count, size in enumerate(range(self.minCellSize + 1, self.maxCellSize)):
                offsets[size] = leadingOffset * -(count + 1)
            return offsets

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
        def __init__(self, imgPath: str, transparency: bool):
            self.opaque_level = 255
            if transparency == True:
                self.img = pg.image.load(imgPath).convert_alpha()
                self.img.set_alpha(self.opaque_level)
            else:
                self.img = pg.image.load(imgPath).convert()

    cellObj = gameObject('assets/cube.png',True)
    tileObj = gameObject('assets/tile.png',True)

    gMatrix = Matrix(10, 3, 60)
    rPentomino = np.array([[10,10],[10,11],[11,9],[11,10],[12,10]])
    gMatrix.loadPatternOld(rPentomino)



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
    

    def loop():
        while game.running:

            game.single_click_events()
            game.key_hold_events()
            game.mouse_events()
            
            game.screen.blit(gMatrix.bgSurf,(0,0))

            if game.count % game.updateDelay == 0:
                newCells = np.zeros([gMatrix.rows,gMatrix.cols], dtype = int)
                if not game.paused:
                    newMatrix = np.array(gMatrix.currentMatrix)
                    newMatrix = gMatrix.updateMatrix(gMatrix.currentMatrix, newMatrix, newCells, gMatrix.indices)
                    gMatrix.currentMatrix = newMatrix
            if game.paused:
                game.pause(True)
                

            gMatrix.renderCells()
            

            population = str(np.count_nonzero(gMatrix.currentMatrix))
            game.screen.blit(game.fontMed.render("Population: " + population, 1, (0,255,0)), (35,10))


            display_fps = game.clock.get_fps()
            if display_fps > 0:
                game.fps_correction = game.max_fps / display_fps
          


            #---FOR TESTING:---#
            # game.screen.blit(game.fontMed.render("FPS: " + str(display_fps), 1, (0,255,0)), (550,10))
            # game.screen.blit(game.fontMed.render("X/Y Offset: " + str(game.xOffset) + "," + str(game.yOffset), 1, (0,255,0)), (350,10))
            # game.screen.blit(game.fontMed.render("number of cols: " + str(game.sWidth // gMatrix.currentCellSize), 1, (0,255,0)), (350,50))
            # game.screen.blit(game.fontMed.render("number of rows: " + str(game.sHeight // gMatrix.currentCellSize), 1, (0,255,0)), (650,50))
            # game.screen.blit(game.fontMed.render("cell size: " + str(gMatrix.currentCellSize), 1, (0,255,0)), (950,50))
            # game.screen.blit(game.fontMed.render("center col/row:" + str(((game.sWidth / gMatrix.currentCellSize) // 2) - (game.xOffset // gMatrix.currentCellSize)) + "," + str(((game.sHeight / gMatrix.currentCellSize) // 2)  - (game.yOffset // gMatrix.currentCellSize)), 1, (0,255,0)), (650,80))
            # game.screen.blit(game.fontMed.render("dict offset: " + str(gMatrix.offsets[gMatrix.currentCellSize]), 1, (0,255,0)), (950,100))

            Gui.gameMenu(game_gui)
            game.count += 1
            pg.display.flip()
            game.clock.tick(game.max_fps)


    game.game_loop()

run()