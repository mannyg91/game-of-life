import pygame as pg
import numpy as np
import time
from numpy import random
from pygame.locals import *
from sys import exit
from numba import njit

class Game():
    def __init__(self):
        pg.init()
        self.running = True
        self.paused = True
        self.FPS = 60
        self.updateRate = 3
        displayInfo = pg.display.Info()
        self.sWidth = displayInfo.current_w
        self.sHeight = displayInfo.current_h
        flags = FULLSCREEN | DOUBLEBUF
        self.screen = pg.display.set_mode((self.sWidth, self.sHeight), flags, pg.FULLSCREEN, vsync=1)
        pg.display.set_caption("Conway's Game of Life")
        self.bg = pg.image.load('assets/bgGame.png').convert()
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
        self.clock = pg.time.Clock()
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
                    case pg.K_m:
                        if self.muted:
                            self.muted = False
                        else:
                            self.muted = True
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

gameBar = Btn('assets/bgGame.png', 'assets/bgGame.png', True, 1, 0, game.sHeight - 100)
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

    def getPatterns(self):
        return np.array(np.where(self))

    def loadPattern(self, indexArr):
        for index in indexArr:
            self.currentMatrix[index[0],index[1]] = True
            
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


gMatrix = Matrix(10, 5, 500)
rPentomino = np.array([[28,30],[28,31],[29,29],[29,30],[30,30]])
gMatrix.loadPattern(rPentomino)

gridSurf = pg.Surface((game.sWidth,game.sHeight))



gMatrix.renderGrid()
game.gameAudio()
game.paused = False
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
        
        pg.display.flip()
        game.clock.tick(game.FPS)

game.game_loop()