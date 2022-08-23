import pygame as pg
import sys
from pygame.locals import *
import gameOfLife_musicMode
import gameOfLife
import gameOfLife_isometric


class Game():
    def __init__(self):
        pg.init()
        self.FPS = 60
        self.updateRate = 15
        self.clock = pg.time.Clock()
        displayInfo = pg.display.Info()
        self.sWidth = displayInfo.current_w
        self.sHeight = displayInfo.current_h
        flags = FULLSCREEN | DOUBLEBUF
        self.screen = pg.display.set_mode((self.sWidth, self.sHeight), flags, pg.FULLSCREEN, vsync=1)
        pg.display.set_caption("Conway's Game of Life")
        self.bg = pg.image.load("assets/bgmenu2.jpg").convert()
        self.centerX = self.sWidth//2
        self.centerY = self.sHeight//2

    def gameAudio(self):
                pg.mixer.init()
                pg.mixer.music.load("assets/paradigm.mp3")
                self.volume = .5
                self.muted = False
                pg.mixer.music.set_volume(self.volume)
                pg.mixer.music.play()
                
    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit()
            if event.type == pg.KEYDOWN:
                match event.key:
                    case pg.K_m:
                        if self.muted:
                            self.muted = False
                        else:
                            self.muted = True
                    case pg.K_ESCAPE:
                        pg.quit()
                        exit()

    def gameAudio(self):
        pg.mixer.init()
        pg.mixer.music.load("assets/intro.mp3")
        self.btn = pg.mixer.Sound("assets/btnsound.mp3")
        self.volume = .9
        self.muted = False
        self.sfx_playing = False
        pg.mixer.music.set_volume(self.volume)
        pg.mixer.music.play()



game = Game()
game.gameAudio()



def get_font(size):
    return pg.font.Font("assets/square_pixel-7.ttf", size)

def empty():
    pass

def quit():
    pg.quit()
    exit()

def main_menu():

    TRANSITION = pg.image.load("assets/transition.png").convert_alpha()
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
            game.screen.blit(img, (0,0))
            img.set_alpha(transAlpha)
            transAlpha -= 1  
        return transAlpha

    def closeFade(img, transAlpha, func):
        if transAlpha < 255:
            game.screen.blit(TRANSITION, (0,0))
            img.set_alpha(transAlpha)
            transAlpha += 1  
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
                # if game.sfx_playing == False:
                #     # pg.mixer.Channel(0).play(game.btn)
                #     game.sfx_playing = True
                if pg.mouse.get_pressed()[0]:
                    self.funct()
            else:
                self.imgRender = self.img
            game.screen.blit(self.imgRender, (self.rect.x, self.rect.y))


    alpha = 0
    alpha2 = 0
    transAlpha = 255
    titleSize = 500

    startBtn = Btn(startImg, startLitImg, .6, game.centerX-150, game.centerY+20, gameOfLife.run)
    musicBtn = Btn(musicImg, musicLitImg, .6, game.centerX-520, game.centerY+105, gameOfLife_musicMode.run)
    cubeBtn = Btn(cubeImg, cubeLitImg, .6, game.centerX+240, game.centerY+95, gameOfLife_isometric.run)
    exitBtn = Btn(exitImg, exitLitImg, .6, game.centerX-150, game.centerY+200, quit)

    
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
    count = 0

    while True:
        game.events()
        game.screen.blit(game.bg, (0, 0))

        
        MENU_TEXT = get_font(int(titleSize * .38)).render("Conway's", True, "#00ff00")
        MENU_TEXT2 = get_font(titleSize).render("Game of Life", True, "#00ff00")
        MENU_TEXT.set_alpha(alpha)
        MENU_TEXT2.set_alpha(alpha)

        game.screen.blit(MENU_TEXT, MENU_TEXT.get_rect(center=(game.centerX-350,game.centerY-265)))
        game.screen.blit(MENU_TEXT2, MENU_TEXT2.get_rect(center=(game.centerX,game.centerY-180)))

        if titleSize <= 180:
            startBtn.render(alpha2)
            musicBtn.render(alpha2)
            cubeBtn.render(alpha2)
            exitBtn.render(alpha2)
            alpha2 += 1


        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        if titleSize > 180:
            titleSize -= 1

        if count % 2 == 0:
            alpha += 1

        count += 1
        



        transAlpha = openFade(TRANSITION,transAlpha)
        pg.display.update()
        game.clock.tick(game.FPS)

main_menu()