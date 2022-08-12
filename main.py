from matplotlib import scale
import pygame as pg
import time
import sys
from button import Button
from pygame.locals import *
import gameOfLife_MIDI
import gameOfLife__working__no_classes_
import gameOfLife_isometric
import time

pg.init()
clock = pg.time.Clock()
FPS = 60
displayInfo = pg.display.Info()
sWidth = displayInfo.current_w
sHeight = displayInfo.current_h

flags = FULLSCREEN | DOUBLEBUF
SCREEN = pg.display.set_mode((sWidth, sHeight), flags, pg.FULLSCREEN)
pg.display.set_caption("Conway's Game of Life")

centerX = sWidth//2
centerY = sHeight//2

def get_font(size):
    return pg.font.Font("assets/square_pixel-7.ttf", size)

def empty():
    pass

def quit():
    pg.quit()
    exit()

def main_menu():

    TRANSITION = pg.image.load("assets/transition.png").convert_alpha()
    BG = pg.image.load("assets/bgmenu2.jpg").convert_alpha()
    startImg = pg.image.load("assets/startBtn.png").convert_alpha()
    musicImg = pg.image.load("assets/musicBtn.png").convert_alpha()
    cubeImg = pg.image.load("assets/cubeBtn.png").convert_alpha()
    exitImg = pg.image.load("assets/exitBtn.png").convert_alpha()
    startLitImg = pg.image.load("assets/startBtnLit.png").convert_alpha()
    musicLitImg = pg.image.load("assets/musicBtnLit.png").convert_alpha()
    cubeLitImg = pg.image.load("assets/cubeBtnLit.png").convert_alpha()
    exitLitImg = pg.image.load("assets/exitBtnLit.png").convert_alpha()

    def openFade(img, transAlpha):
        if transAlpha > 0 and transAlpha <= 255:
            SCREEN.blit(img, (0,0))
            img.set_alpha(transAlpha)
            transAlpha -= 2   
        return transAlpha

    def closeFade(img, transAlpha, func):
        if transAlpha < 255:
            SCREEN.blit(TRANSITION, (0,0))
            img.set_alpha(transAlpha)
            transAlpha += 2  
        if transAlpha >= 255:
            pass
        return transAlpha


    class Btn():
        def __init__(self, img, imgHover, scale, x, y, function):
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
            SCREEN.blit(self.imgRender, (self.rect.x, self.rect.y))


    alpha = 0
    alpha2 = 0
    transAlpha = 255
    titleSize = 500

    startBtn = Btn(startImg, startLitImg, .6, centerX-150, centerY+20, gameOfLife__working__no_classes_.runBasicGame)
    musicBtn = Btn(musicImg, musicLitImg, .6, centerX-520, centerY+105, gameOfLife_MIDI.runMusicGame)
    cubeBtn = Btn(cubeImg, cubeLitImg, .6, centerX+240, centerY+95, gameOfLife_isometric.run)
    exitBtn = Btn(exitImg, exitLitImg, .6, centerX-150, centerY+200, quit)

    
# def fade(img, alpha):
#     if alpha == 0:
#         while alpha < 255:
#             SCREEN.blit(img, (0,0)) 
#             img.set_alpha
#             alpha += 1
#     else:     
#         while alpha > 0:
#             SCREEN.blit(img, (0,0))
#             img.set_alpha
#             alpha += 1


    while True:
        
        SCREEN.blit(BG, (0, 0))

        
        MENU_TEXT = get_font(int(titleSize * .38)).render("Conway's", True, "#00ff00")
        MENU_TEXT2 = get_font(titleSize).render("Game of Life", True, "#00ff00")
        MENU_TEXT.set_alpha(alpha)
        MENU_TEXT2.set_alpha(alpha)

        SCREEN.blit(MENU_TEXT, MENU_TEXT.get_rect(center=(centerX-350,centerY-265)))
        SCREEN.blit(MENU_TEXT2, MENU_TEXT2.get_rect(center=(centerX,centerY-180)))

        if titleSize <= 180:
            startBtn.render(alpha2)
            musicBtn.render(alpha2)
            cubeBtn.render(alpha2)
            exitBtn.render(alpha2)
            alpha2 += 3


        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        if titleSize > 180:
            titleSize -= 3
        alpha += 2

        transAlpha = openFade(TRANSITION,transAlpha)
        pg.display.update()
        clock.tick(FPS)

main_menu()