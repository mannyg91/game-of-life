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
            self.zoomDelay = 2
            self.moveDelay = 1
            self.clock = pg.time.Clock()
            displayInfo = pg.display.Info()
            self.sWidth = displayInfo.current_w
            self.sHeight = displayInfo.current_h
            self.screen = pg.display.set_mode((self.sWidth, self.sHeight), flags = pg.FULLSCREEN | pg.DOUBLEBUF, vsync=1)
            # self.screen = pg.display.set_mode((1200,1000), flags = pg.RESIZABLE)
            pg.display.set_caption("Conway's Game of Life")
            # self.bg = pg.image.load('assets/bgGame2.png').convert()
            self.bg = pg.image.load('assets/a (4).png').convert()
            self.font_small = pg.font.Font('assets/square_pixel-7.ttf', 22)
            self.fontMed = pg.font.Font('assets/square_pixel-7.ttf', 27)
            self.fontLarge = pg.font.Font('assets/square_pixel-7.ttf', 32)
            self.xOffset, self.yOffset, self.count = 0, 0, 0
            self.gameAudio()
            self.drawing = True
            self.capturing = False
            self.placing = False
            self.funct = None

        def gameAudio(self):
            pg.mixer.init()
            pg.mixer.music.load("assets/paradigm.mp3")
            self.flash = pg.mixer.Sound("assets/camera_flash.mp3")
            self.volume = .8
            self.muted = False
            pg.mixer.music.set_volume(self.volume)
            pg.mixer.music.play()

        def game_loop(self):
            loop()

        def pause(self, paused: True):
            if paused:
                self.paused = True
                self.screen.blit(self.fontLarge.render("PAUSED", 1, (0,255,0)), (self.sWidth//2 - 57, 35))
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
                    if event.key == pg.K_ESCAPE:
                        pg.quit()
                        exit()

                if TextInput.input_active == False:
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
                            
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.funct:
                            self.funct()

                        if game.placing:
                            mouseX, mouseY = pg.mouse.get_pos()
                            if mouseX >= self.xOffset and mouseY >= self.yOffset and mouseY < game.sHeight * .91 and BtnImg.sliding == False:
                                # try:
                                #     self.paused = True
                                #     roundedX = (mouseX - self.xOffset) / gMatrix.currentCellSize
                                #     roundedY = (mouseY - self.yOffset) / gMatrix.currentCellSize
                                #     # gMatrix.place_pos = [int(roundedY),int(roundedX)]
                                #     Submatrix.place(int(roundedX,int(roundedY)))
                                # except:
                                #     print("out of range - place")

                            
                                self.paused = True
                                roundedX = int((mouseX - self.xOffset) / gMatrix.currentCellSize)
                                roundedY = int((mouseY - self.yOffset) / gMatrix.currentCellSize)
                                # gMatrix.place_pos = [int(roundedY),int(roundedX)]
                                Submatrix.place(roundedX,roundedY)
            
                    



                    if name_input.input_rect.collidepoint(event.pos):
                        TextInput.input_active = True
                    else:
                        TextInput.input_active = False
                        name_input.color = name_input.color_passive

                    
     




                if game_gui.menuOpen:
                    new_volume = volumeSliderBall.slider(volumeSlider, event, game.volume)
                    new_speed = speedSliderBall.slider(speedSlider, event, game.updateDelay)
                    if new_volume != None:
                        pg.mixer.music.set_volume(new_volume)
                    if new_speed != None:
                        if new_speed == 0:
                            new_speed += 1
                        self.updateDelay = 100 - new_speed

                if self.capturing == True:
                    gMatrix.capture_patterns(event)



                if TextInput.input_active == True:
                    name_input.color = name_input.color_active
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_BACKSPACE:
        
                            name_input.user_text = name_input.user_text[:-1]
        
                        else:
                            name_input.user_text += event.unicode
                            print(name_input.user_text)




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

            if game.drawing:
                if click[0]:
                    if mouseX >= self.xOffset and mouseY >= self.yOffset and mouseY < game.sHeight * .91 and BtnImg.sliding == False:
                        try:
                            self.paused = True
                            roundedX = (mouseX - self.xOffset) / gMatrix.currentCellSize
                            roundedY = (mouseY - self.yOffset) / gMatrix.currentCellSize
                            gMatrix.currentMatrix[int(roundedY),int(roundedX)] = True
                        except:
                            print("out of range - draw")
                elif click[2] and mouseX >= self.yOffset and mouseY >= self.yOffset and mouseY < game.sHeight * .91:
                    if mouseX >= self.xOffset and mouseY >= self.yOffset:
                        try:
                            self.paused = True
                            roundedX = (mouseX - self.xOffset) / gMatrix.currentCellSize
                            roundedY = (mouseY - self.yOffset) / gMatrix.currentCellSize
                            gMatrix.currentMatrix[int(roundedY),int(roundedX)] = False
                        except:
                            print("out of range - draw")



    game = Game()

    class Gui():
        def __init__(self):
            self.opaque_level = 0
            self.opaque_level_fast = 0
            self.menuOpen = False
            self.library_open = False


        def gameMenu(self):
            mouseX, mouseY = pg.mouse.get_pos()
            if mouseY > game.sHeight * .75:
                self.menuOpen = True
                if self.opaque_level < 252:
                    self.opaque_level += 3 * game.fps_correction
            else:
                self.menuOpen = False
                if self.opaque_level > 0:
                    self.opaque_level -= 15 * game.fps_correction

            gameBar.render(self.opaque_level)
            # playArea.render(self.opaque_level_fast)
            playBtn.render(self.opaque_level)
            speedIcon.render(self.opaque_level)
            speedSlider.render(self.opaque_level)

            soundIcon.render(self.opaque_level)
            volumeSlider.render(self.opaque_level)
            speedSliderBall.render(self.opaque_level)
            volumeSliderBall.render(self.opaque_level)

            captureIcon.render(self.opaque_level)

            libraryIcon.render(500,500)



        
            population = str(np.count_nonzero(gMatrix.currentMatrix))
            population_surf = game.fontMed.render("Population: " + population, 1, (0,255,0))
            population_surf.set_alpha(self.opaque_level)
            game.screen.blit(population_surf, (game.sWidth - 300, 20))

            if gMatrix.flashing == True:
                gMatrix.screen_flash()


        def pauseSwitch():
            if game.paused:
                game.paused = False
            else:
                game.paused = True
            game.pause(game.paused)

        
        def capture_switch():
            if game.capturing == True:
                game.capturing = False
                game.drawing = True
                gMatrix.saved = True
            else:
                game.capturing = True
                game.drawing = False
                gMatrix.saved = False


        # def horizontal_slider(self, line, circle):
        #     min = 0
        #     max = line.width
        #     click = pg.mouse.get_pressed()
        #     mouseX, mouseY = pg.mouse.get_pos()
        #     if click[0]:
        #         if mouseX > self.x:

        def library_switch(self):
            if self.library_open == False:
                self.library_open = True
                print("library trynng to render")
                # Submatrix.render_library()
            else:
                self.library_open = False
            
        def place_switch():
            if game.placing == False:
                game.placing = True
            else:
                game.placing = False


         
            


    game_gui = Gui()

    class BtnRect():
        def __init__(self, text, function, extra_width = 25, extra_height = 5, fill_color = (0,0,0), border_color = (255,255,255), font_color = (255,255,255), font_color_hover = (0,255,0)):
            self.text = text
            self.btn_funct = function
            self.fill_color = fill_color
            self.border_color = border_color
            self.border_width = 2
            self.border_radius = 5
            self.font_color = font_color
            self.font_color_hover = font_color_hover
            self.text_width, self.text_height = game.font_small.size(text)
            self.extra_width = extra_width
            self.extra_height = extra_height
            self.padding = 0
            self.min_width = self.text_width + self.padding
            self.width = self.min_width + extra_width
            self.height = self.text_height + self.padding + self.extra_height

            self.surf = pg.Surface((self.width,self.height))
            pg.draw.rect(self.surf, self.border_color, (0,0, self.width,self.height), border_radius= self.border_radius)
            pg.draw.rect(self.surf, self.fill_color, (self.border_width,self.border_width,self.width-(self.border_width*2),self.height-(self.border_width*2)))
            

        def render(self, screen_offset_x, screen_offset_y):
            self.x_pos = (game.sWidth // 2) + screen_offset_x
            self.y_pos = (game.sHeight // 2) + screen_offset_y
            self.hover_rect = pg.draw.rect(game.screen, self.border_color, (self.x_pos,self.y_pos, self.width,self.height), border_radius= self.border_radius)
            game.screen.blit(self.surf, (self.x_pos, self.y_pos))
            mouse_pos = pg.mouse.get_pos()
            if self.hover_rect.collidepoint(mouse_pos):
                self.current_color = self.font_color_hover
                game.funct = self.btn_funct
            else:
                self.current_color = self.font_color
            game.screen.blit(game.font_small.render(self.text, 1, self.current_color), (self.x_pos + self.padding + (self.extra_width//2), self.y_pos + self.padding + (self.extra_height//2)))

    libraryIcon = BtnRect("Library", game_gui.library_switch)


    class BtnImg():
        sliding = False

        def __init__(self, imgPath: str, imgHoverPath: str, transparency: bool, scale, x, y, function, alpha = 255):
            if transparency == True:
                img = pg.image.load(imgPath).convert_alpha()
                img.set_alpha(alpha)
                imgHover = pg.image.load(imgHoverPath).convert_alpha()
                imgHover.set_alpha(alpha)
            else:
                img = pg.image.load(imgPath).convert()
                imgHover = pg.image.load(imgHoverPath).convert()
            self.img = pg.transform.scale(img, (int(scale*img.get_width()), int(scale*img.get_height())))
            self.imgHover = pg.transform.scale(imgHover, (int(scale*img.get_width()), int(scale*img.get_height())))
            self.width = img.get_width() * scale
            self.height = img.get_height() * scale
            self.x = x
            self.y = y 
            self.slider_min = x
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
            game.screen.blit(self.imgRender, (self.x, self.y))

        def test():
            print("pressed")


        def slider(self, line, event, toChange):
            offset_x = 0
            min = line.slider_min - self.width // 2
            max =  min + line.width
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:            
                    if self.rect.collidepoint(event.pos):
                        BtnImg.sliding = True
                        offset_x = self.rect.x - event.pos[0]
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:            
                    BtnImg.sliding = False
            if event.type == pg.MOUSEMOTION:
                if BtnImg.sliding:
                    if event.pos[0] < max and event.pos[0] > min:
                        self.rect.x = event.pos[0] + offset_x
                        self.x = self.rect.x
                        slider_position = self.x - min
                        old_range = max - min
                        new_value = (((slider_position - 0) * 100) / old_range)
                        if toChange < 1:
                            toChange = new_value / 100
                        else:
                            toChange = int(new_value)
                        return toChange

                        


    gameBar = BtnImg('assets/gameBar.png', 'assets/gameBar.png', True, 1, 0, game.sHeight - 120, BtnImg.test)
    
    playBtn = BtnImg('assets/play.png', 'assets/playLit.png', True, 1, game.sWidth//2 - 16, game.sHeight - 62, Gui.pauseSwitch)
    playBtn2 = BtnImg('assets/play.png', 'assets/playLit.png', True, 1, game.sWidth//2 - 64, game.sHeight//2, Gui.pauseSwitch)
    playArea = BtnImg('assets/playArea.png','assets/playArea.png', True, .8, game.sWidth//2 -135, game.sHeight - 111, Gui.pauseSwitch)

    speedIcon = BtnImg('assets/speed.png','assets/speed.png', True, .11, game.sWidth//2 - 467, game.sHeight-60, Gui.pauseSwitch)
    speedSlider = BtnImg('assets/sliderLit.png','assets/sliderLit.png', True, .3, game.sWidth//2 - 420, game.sHeight -49, BtnImg.test)
    speedSliderBall = BtnImg('assets/sliderBall.png','assets/sliderBallLit.png', True, .4, game.sWidth//2 - 370, game.sHeight - 58, BtnImg.test)

    soundIcon = BtnImg('assets/soundLoud.png','assets/soundLoud.png', True, .25, game.sWidth//2 - 735, game.sHeight - 61, BtnImg.test)
    volumeSlider = BtnImg('assets/sliderLit.png','assets/sliderLit.png', True, .3, game.sWidth//2 - 700, game.sHeight-49, BtnImg.test)
    volumeSliderBall = BtnImg('assets/sliderBall.png','assets/sliderBallLit.png', True, .4, game.sWidth//2 - 650, game.sHeight - 58, BtnImg.test)

    captureIcon = BtnImg('assets/captureIcon.png', 'assets/captureIconLit.png', True, .6, game.sWidth//2 + 350, game.sHeight - 68, Gui.capture_switch)
    


    class Matrix:
        def __init__(self, startCellSize, minCellSize, maxCellSize):
            self.minCellSize = minCellSize
            self.currentCellSize = startCellSize
            self.initialCellSize = startCellSize
            self.maxCellSize = maxCellSize
            self.rows = game.sHeight // minCellSize    
            self.cols = game.sHeight // minCellSize
            self.indices = self.allIndices()
            self.emptyMatrix = self.createMatrix()
            self.currentMatrix = self.emptyMatrix
            self.angle = 1
            self.bgColor = [0,0,0]
            self.initGridColor = [120,15,195]
            self.cellColor = (255,255,255)
            self.xBoundary = self.cameraBounds(game.sWidth)
            self.yBoundary = self.cameraBounds(game.sHeight)
            self.gridShades = self.createGridShades(self.minCellSize, .973, self.initGridColor)
            self.bgSurf = pg.Surface((game.sWidth, game.sHeight))
            self.boxSurf = pg.Surface((game.sWidth, game.sHeight))
            # self.submatrix_surf = pg.Surface((game.sWidth//2, game.sHeight//2))
            # self.submatrix_cellsize = 10

            self.saved_patterns = {
                1 : [Submatrix(np.array([[1,1,1 ,1 ,0 ,0 ,1 ,1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]])),"Test"],
                2 : [Submatrix(np.array([[0,0,0,0,1,1], [0,0,0,1,0,1], [0,0,0,1,0,0], [0,1,1,1,0,0], [1,0,0,0,0,0], [1,1,0,0,0,0]])), "Elevener"],
                3 : [Submatrix(np.array([[1,1,1,0,0,0], [1,1,1,0,0,0], [1,1,1,0,0,0], [0,0,0,1,1,1], [0,0,0,1,1,1], [0,0,0,1,1,1]])), "Figure Eight"],
                4 : [Submatrix(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1], [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1], [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])), "Glider Gun"],
                5 : [Submatrix(np.array([[1,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,  1,1]])), "Methuselah"],
                6 : [Submatrix(np.array([[0,0,1,0,0,0,0,1,0,0], [1,1,0,1,1,1,1,0,1,1], [0,0,1,0,0,0,0,1,0,0]])), "Pentadecathlon"],
                7 : [Submatrix(np.array([[0,0,0,1,1,0,0,0], [0,0,1,0,0,1,0,0], [0,1,0,0,0,0,1,0], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,0,0,0,0,1,0], [0,0,1,0,0,1,0,0], [0,0,0,1,1,0,0,0]])), "Octagon 2"],
                8 : [Submatrix(np.array([[0,0,1,1,1,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [0,0,1,1,1,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,1,1,0,0,0,1,1,1,0,0], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,1,1,0,0,0,1,1,1,0,0]])), "Pulsar"],
                
                }

            self.current_submatrix = None
            self.capture_x = 0
            self.capture_y = 0
            self.capture_display_x = 0
            self.capture_display_y = 0
            self.capture_display_w = 0
            self.capture_display_h = 0
            self.capture_count = len(self.saved_patterns)
            self.start_row = 0
            self.start_col = 0
            self.end_row = 0
            self.end_col = 0
            self.confirmed = False
            self.confirm_surf = pg.Surface((300,300))
            self.saved = False
            self.flashing = False
            self.flash_alpha = 150
            self.mouse_down = False

            self.place_pos = None

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


        def capture_patterns(self, event):

            if event.type == pg.MOUSEBUTTONDOWN:
                
                if event.button == 1:     
                    self.mouse_down = True
                    #cell selection logic must differ from display logic
                    self.start_col = int((event.pos[0] - game.xOffset) / gMatrix.currentCellSize)
                    self.start_row = int((event.pos[1] - game.yOffset) / gMatrix.currentCellSize)

                    # print("start_col and start_row" + str(self.start_col) + "," + str(self.start_row))

                    print("mouse button down")
                    if self.start_col < 0:
                        self.start_col = 0
                    if self.start_row < 0:
                        self.start_row = 0
                    self.capture_x = event.pos[0]
                    self.capture_y = event.pos[1]
                    print("first xy pos:")
                    print(self.capture_x,self.capture_y)


            if event.type == pg.MOUSEBUTTONUP:
                #don't allow matrices smaller than 4x4
                #need start, end


                if event.button == 1:
                    self.mouse_down = False
                    game.capturing == False
                    #RECORD MATRIX, POP UP ANOTHER WINDOW, ASKING IF THEY'D LIKE TO SAVE IT WITH A MINI MATRIX REPRESENTAITON.

                    #the way these are calculated are probably troublesome
                    self.end_col = int((event.pos[0] - game.xOffset) / gMatrix.currentCellSize)
                    self.end_row = int((event.pos[1] - game.yOffset) / gMatrix.currentCellSize)

                    # print("end_col and end_row" + str(self.end_col) + "," + str(self.end_row))

                    if self.end_col < 0:
                        self.end_col = 0
                    if self.end_row < 0:
                        self.end_row = 0

                    

                    row_range =  abs(self.end_row - self.start_row) + 1
                    col_range =  abs(self.end_col - self.start_col) + 1

                    if row_range > 2 and col_range > 2:


                        #started editing here
                        #Quadrant 1
                        if self.start_col > self.end_col and self.start_row > self.end_row:
                            self.current_submatrix = Submatrix(gMatrix.trimMatrix(gMatrix.currentMatrix[self.end_row:self.start_row:+1,self.end_col:self.start_col+1]))
                        #Quadrant 2
                        elif self.start_col < self.end_col and self.start_row > self.end_row:
                            print("quadrant2")
                            self.current_submatrix = Submatrix(gMatrix.trimMatrix(gMatrix.currentMatrix[self.end_row:self.start_row+1, self.start_col:self.end_col+1]))
                        #Quadrant 3
                        elif self.start_col > self.end_col and self.start_row < self.end_row:
                            self.current_submatrix = Submatrix(gMatrix.trimMatrix(gMatrix.currentMatrix[self.start_row:self.end_row+1,self.end_col:self.start_col+1]))
                        #Quadrant 4
                        else:
                            print("quadrant 4")
                            self.current_submatrix = Submatrix(gMatrix.trimMatrix(gMatrix.currentMatrix[self.start_row:self.end_row+1,self.start_col:self.end_col+1]))

                        # self.current_submatrix = Submatrix(gMatrix.trimMatrix())


                        self.flashing = True
                        



                #JUST FOR TESTING: CONFIRMS THAT PATTERN EXISTS
                # for i in saved_patterns:
                #     print("pattern:")
                #     print(saved_patterns[i].submatrix)                       


            if event.type == pg.MOUSEMOTION and self.mouse_down:
                print("mouse motion reached")

                self.capture_display_w = abs(event.pos[0] - self.capture_x)
                self.capture_display_h = abs(event.pos[1] - self.capture_y)    
            
                if event.pos[0] < self.capture_x:
                    self.capture_display_x = event.pos[0]
                else:
                    self.capture_display_x = self.capture_x

                if event.pos[1] < self.capture_y:
                    self.capture_display_y = event.pos[1]
                else:
                    self.capture_display_y = self.capture_y



                    



                
                    



        def trimMatrix(self, matrix):
 
            alive_indices = np.nonzero(matrix)
            min_indices = [np.subtract(alive_indices[0],np.amin(alive_indices[0])),np.subtract(alive_indices[1],np.amin(alive_indices[1]))]

            new_rows = np.amax(min_indices[0]) +1
            new_cols = np.amax(min_indices[1]) +1


            trimmed_matrix = np.zeros(shape=(new_rows,new_cols))
            for row, col in zip(*min_indices):
                trimmed_matrix[row,col] = True
            trimmed_matrix = np.array(trimmed_matrix, dtype='int16')

            print("trimmed_matrix")
            print(trimmed_matrix)

            return trimmed_matrix


        def accepted(self):
            print("Accepted")
            # self.saved_patterns[self.capture_count] = self.current_submatrix
            self.confirmed = True
            # for i in self.saved_patterns:
            #     print("pattern:")
            #     print(self.saved_patterns[i].submatrix)                       


        def rejected(self):
            print("rejected")
            self.confirmed = False
            self.capturing = False
            # for i in self.saved_patterns:
            #     print("pattern:")
            #     print(self.saved_patterns[i].submatrix)              

        def screen_flash(self):
            pg.mixer.Channel(0).play(game.flash)

            # if self.flash_alpha > 10:
            #     self.flash_alpha -= 10
            # flash_surf = pg.Surface((game.sWidth, game.sHeight)).convert_alpha()
            # flash_surf.fill((255, 255, 255, self.flash_alpha))
            # game.screen.blit(flash_surf, (0,0))
            
            # if game.count % 10 == 0:
            self.flashing = False
                # self.flash_alpha = 150




        def getPatterns(self):
            return np.array(np.where(self))

        def loadPatternOld(self, indexArr):
            for index in indexArr:
                self.currentMatrix[index[0],index[1]] = True

        def loadPattern(self, pattern):
            for row, col in zip(*pattern):
                self.currentMatrix[row,col] = True
            return


    class Submatrix:
        saved_patterns = {}
        page_num = 1

        def __init__(self, np_submatrix, width_center=game.sWidth//2, height_center=game.sHeight//2):
            self.submatrix = np_submatrix
            print(self.submatrix.shape)
            self.rows, self.cols = self.submatrix.shape
            self.grid_color = [80,80,80]
            self.bg_color = (100,100,100)
            self.cell_color = (0,255,0)
            self.width_center = width_center
            self.height_center = height_center

            #error happens here
            if (self.width_center) // self.cols > (self.height_center) // self.rows:
                self.cell_size = self.height_center // self.rows
            else:
                self.cell_size = self.width_center // self.cols
            self.submatrix_width = self.cell_size * self.cols
            self.submatrix_height = self.cell_size * self.rows
            self.submatrix_surf = pg.Surface((self.submatrix_width, self.submatrix_height))
            self.border_width = 3
            self.submatrix_border = pg.Surface((self.submatrix_width + (self.border_width * 2), self.submatrix_height + (self.border_width * 2)))
            self.alive_indices = np.nonzero(self.submatrix)
            self.end_pos_y = self.height_center + (self.submatrix_height // 2)


            

            self.lib_grid_color = (20,20,20)
            self.lib_bg_color = (0,0,0)
            self.lib_cell_color = (0,255,0)

            self.lib_border_width = 1

            self.library_width = library_window.width
            self.library_height = library_window.height

            self.library_surf = pg.Surface((library_window.width, library_window.height *.6))
            self.library_padding = 10



            self.section_width = self.library_width - 25
            self.section_height = self.library_height *.6


            if self.section_width // self.cols > self.section_height // self.rows:
                self.lib_cell_size = self.section_height // self.rows
                print("libcellsize")
                print(self.lib_cell_size)
            else:
                self.lib_cell_size = self.section_width // self.cols
                print("libcellsize")
                print(self.lib_cell_size)
            
            if self.lib_cell_size < 1:
                self.lib_cell_size = 1

            self.sect_submatrix_w = self.lib_cell_size * self.cols
            self.sect_submatrix_h = self.lib_cell_size * self.rows

            #PUT EDGE LIMIT HERE, IF THE ABOVE IS GREATER THAN WIDTH OF SECTION WITH, THEN SET TO SECTION_WIDTH
            if self.sect_submatrix_w > self.section_width:
                self.sect_submatrix_w = self.section_width
            
            self.section_border_surf = pg.Surface((self.sect_submatrix_w + (self.lib_border_width*2),self.section_height))
            self.section_surf = pg.Surface((self.sect_submatrix_w,self.sect_submatrix_h))

            # self.section_width = self.lib_cell_size * self.cols
            # self.section_height = self.lib_cell_size * self.rows

            self.max_cols = self.sect_submatrix_w // self.lib_cell_size
            self.max_rows = self.sect_submatrix_h // self.lib_cell_size
            self.x_offset = (self.library_width - self.sect_submatrix_w) // 2
            self.y_offset = (self.section_height - self.sect_submatrix_h) // 2

        def render_submatrix(self):
            pg.draw.rect(self.submatrix_border, (255,255,255), (0,0,self.submatrix_width + (self.border_width * 2),self.submatrix_height + (self.border_width * 2)), border_radius=6 )
            pg.draw.rect(self.submatrix_surf, (100,100,100), (0,0,self.submatrix_width,self.submatrix_height))


            for row, col in zip(*self.alive_indices):
                if self.submatrix[row,col] == 1:
                    pg.draw.rect(self.submatrix_surf,self.cell_color,(col * self.cell_size + 1, row * self.cell_size + 1,self.cell_size,self.cell_size))
            if self.cell_size > 1:
                for row in range(self.rows):

                                    #surf                color               surf pos                          size
                    pg.draw.line(self.submatrix_surf, self.grid_color, (0,row * self.cell_size), (self.submatrix_width,row * self.cell_size))
                for col in range(self.cols):
                    pg.draw.line(self.submatrix_surf, self.grid_color, (col * self.cell_size,0), (col * self.cell_size,self.submatrix_height))
           
            
            game.screen.blit(self.submatrix_border, (self.width_center - (self.submatrix_width // 2) - self.border_width,self.height_center - (self.submatrix_height // 2) - self.border_width - 50))
            game.screen.blit(self.submatrix_surf, (self.width_center - (self.submatrix_width // 2),self.height_center - (self.submatrix_height // 2)- 50))


        def library_submatrix(self):


                                                    #pos relative to surf               x                                               y
            pg.draw.rect(self.section_border_surf, (0,0,0), (0,0,self.sect_submatrix_w + (self.lib_border_width * 2),self.section_height), border_radius= 2 )
            pg.draw.rect(self.section_surf, (0,0,0), (0,0,self.sect_submatrix_w,self.sect_submatrix_h))

            # if self.cell_size > 4:
            #     for row, col in zip(*self.alive_indices):
            #         if self.submatrix[row,col] == 1:
            #             pg.draw.rect(self.section_surf,self.lib_cell_color,(col * self.lib_cell_size + 1 , row * self.lib_cell_size + 1,self.lib_cell_size,self.lib_cell_size))
            #     for row in range(self.max_rows):
            #         pg.draw.line(self.section_surf, self.lib_grid_color, (0,row * self.lib_cell_size), (self.sect_submatrix_w,row * self.lib_cell_size))
            #     for col in range(self.max_cols):
            #         pg.draw.line(self.section_surf, self.lib_grid_color, (col * self.lib_cell_size,0), (col * self.lib_cell_size,self.sect_submatrix_h))
            # else:
            for row, col in zip(*self.alive_indices):
                if self.submatrix[row,col] == 1:
                    pg.draw.rect(self.section_surf,self.cell_color,(col * self.lib_cell_size , row * self.lib_cell_size,self.lib_cell_size,self.lib_cell_size))

                                    #surf                     x,  y
            game.screen.blit(self.section_border_surf, ((library_window.x_pos + self.x_offset) - self.lib_border_width, (library_window.y_pos + 20) - self.lib_border_width))
            game.screen.blit(self.section_surf, (library_window.x_pos + self.x_offset,library_window.y_pos + self.y_offset + 20))



        def save():
            gMatrix.capture_count += 1
            gMatrix.saved_patterns[gMatrix.capture_count] = [gMatrix.current_submatrix, name_input.user_text]
            name_input.user_text = ""
            gMatrix.saved = True
            gMatrix.confirmed = False
            gMatrix.current_submatrix = None
            
            submatrices = []
            for i in gMatrix.saved_patterns:
                submatrices.append(gMatrix.saved_patterns[i][0])

            #testing:
            # for i in gMatrix.saved_patterns:
            #     print("pattern:")
            #     print(gMatrix.saved_patterns[i][0].submatrix)  
            

        
        def render_library():
            gMatrix.saved_patterns[Submatrix.page_num][0].library_submatrix()



            # for item in cycle(gMatrix.saved_patterns.items()):
            #     # print("pattern:")
            #     # print(gMatrix.saved_patterns[i][0].submatrix)
            #     # #should return object

            #     #tries to call submatrix on the object and render it
            #     gMatrix.saved_patterns[i][0].library_submatrix()


    # submatrixtest = np.array([[0,1],[2,3]])
    # test = Submatrix(submatrixtest)


        def next_page():
            total_pages = len(gMatrix.saved_patterns)
            print(Submatrix.page_num)
            if Submatrix.page_num < total_pages:
                Submatrix.page_num += 1
            elif Submatrix.page_num == total_pages:
                Submatrix.page_num = 1

        def prev_page():
            total_pages = len(gMatrix.saved_patterns)
            print(Submatrix.page_num)
            if Submatrix.page_num > 1:
                Submatrix.page_num -= 1
            elif Submatrix.page_num == 1:
                Submatrix.page_num = total_pages

        def place(x,y):
            game.drawing = False
            sub_obj = gMatrix.saved_patterns[Submatrix.page_num][0]
            sub = sub_obj.submatrix
            sub_height, sub_width = sub.shape
            print(f"subwidth and height: {sub_width} , {sub_height}")
            adjusted_pos = np.subtract((x,y), (sub_width // 2,sub_height // 2))
            # adjusted_pos = (x,y)
            print(f"adjustedpos {adjusted_pos}")
            col, row = adjusted_pos
            print(f"row,col {row,col}")
            gMatrix.currentMatrix[row:row+sub.shape[0], col:col+ sub.shape[1]] += sub




    next_icon = BtnRect("Next", Submatrix.next_page)
    prev_icon = BtnRect("Prev", Submatrix.prev_page)
    place_icon = BtnRect("Place", Gui.place_switch)


    class Window:
        def __init__(self, width, height, border, border_color = (255,255,255)):
            self.height = height
            self.width = width
            self.surf = pg.Surface((self.width,self.height))
            self.fill_color = (0,0,0)
            self.border_color = border_color
            self.border_width = border
            self.border_radius = 7
            self.font_color = (0,255,0)
            self.x_pos = 0
            self.y_pos = 0
            pg.draw.rect(self.surf, self.border_color, (0,0, self.width,self.height), border_radius= self.border_radius)
            pg.draw.rect(self.surf, self.fill_color, (self.border_width,self.border_width,self.width-(self.border_width*2),self.height-(self.border_width*2)))

        def edge_render(self, screen_offset_x, screen_offset_y):
            self.x_pos = game.sWidth + screen_offset_x
            self.y_pos = game.sHeight + screen_offset_y
            game.screen.blit(self.surf, (self.x_pos, self.y_pos))

        def center_render(self, screen_offset_x, screen_offset_y):
            self.x_pos = (game.sWidth // 2) + screen_offset_x
            self.y_pos = (game.sHeight // 2) + screen_offset_y
            game.screen.blit(self.surf, (self.x_pos, self.y_pos))

        def render_text(self, text, offset_x, offset_y, color):
            game.screen.blit(game.font_small.render(text, 1, color), (self.x_pos + offset_x, self.y_pos + offset_y))

            # game.screen.blit(game.fontMed.render("FPS: " + str(display_fps), 1, (0,255,0)), (550,10))

    library_window = Window(350,450,3, (200,200,200))

    class TextInput:
        input_active = False

        def __init__(self, x, y, length, width):
            self.input_rect = pg.Rect(x,y,length,width)
            self.color_active = (215,215,215)
            self.color_passive = (100,100,100)
            self.color = None
            self.text_color = (0,0,0)
            self.user_text = ""
            self.text_surf = game.font_small.render(self.user_text, True, (0,255,0))
            

    name_input = TextInput(810,867,250,25)




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

    gMatrix = Matrix(10, 2, 60)

    rPentomino = np.array([[10,10],[10,11],[11,9],[11,10],[12,10]])
    gMatrix.loadPatternOld(rPentomino)



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
            
            if game.capturing and gMatrix.mouse_down:
                pg.draw.rect(game.screen, (0, 255, 0), (gMatrix.capture_display_x,gMatrix.capture_display_y, gMatrix.capture_display_w, gMatrix.capture_display_h), 1)



            if gMatrix.current_submatrix != None and gMatrix.saved == False:

                if gMatrix.confirmed == False:
                #CHANGE AFTER
                    game.capturing = False
                    confirm = Window(250,60,3)
                    # testwindow.center_render(-175,gMatrix.current_submatrix.submatrix_height//2)
                    confirm.center_render(-128,275)
                    confirm.render_text("Confirm Capture?",34,17, (0,255,0))

                    yes1 = BtnRect("Yes",gMatrix.accepted)
                    no1 = BtnRect("No",gMatrix.rejected,39)
                    yes1.render(-85,360)
                    no1.render(15,360)

                if gMatrix.confirmed == True:
                    save_btn = BtnRect("Save",Submatrix.save)
                    name_prompt = Window(470,100,3)
                    name_prompt.center_render(-235,275)
                    save_btn.render(115,325)
                    name_prompt.render_text("Confirmed. Please enter a name:",47,17,(0,255,0))
                    name_input.text_surf = game.font_small.render(name_input.user_text, True, name_input.text_color)
                    pg.draw.rect(game.screen, name_input.color, name_input.input_rect)
                    game.screen.blit(name_input.text_surf, (name_input.input_rect.x+10, name_input.input_rect.y+1))

               
                # print("main loop matrix:")
                # print(gMatrix.current_submatrix)
                # print(gMatrix.current_submatrix)
        
                gMatrix.current_submatrix.render_submatrix()
                
            


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
            if game_gui.library_open == True:
                library_window.edge_render(-370,-1000)
                next_icon.render(774,-110)
                prev_icon.render(674,-110)
                place_icon.render(720,-60)

   
                gMatrix.saved_patterns[Submatrix.page_num][0].library_submatrix()

                name_text = game.font_small.render(gMatrix.saved_patterns[Submatrix.page_num][1], 1, (0,255,0))
                center_text = (350 - name_text.get_width()) // 2
                game.screen.blit(name_text, (1550 + center_text,375))





            game.count += 1
            pg.display.flip()
            game.clock.tick(game.max_fps)



    game.game_loop()


