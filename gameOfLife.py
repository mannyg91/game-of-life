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
            self.paused = False
            self.max_fps = 300
            self.fps_correction = 1
            self.update_delay = 15
            self.zoom_delay = 3
            self.move_delay = 3
            self.clock = pg.time.Clock()
            display_info = pg.display.Info()
            self.s_width = display_info.current_w
            self.s_height = display_info.current_h
            self.screen = pg.display.set_mode((self.s_width, self.s_height), flags = pg.FULLSCREEN | pg.DOUBLEBUF, vsync=1)
            # self.screen = pg.display.set_mode((1200,1000), flags = pg.RESIZABLE)
            pg.display.set_caption("Conway's Game of Life")
            self.bg = pg.image.load('assets/a (4).png').convert()
            self.font_small = pg.font.Font('assets/square_pixel-7.ttf', 22)
            self.font_med = pg.font.Font('assets/square_pixel-7.ttf', 27)
            self.font_large = pg.font.Font('assets/square_pixel-7.ttf', 32)
            self.x_offset, self.y_offset, self.count = -1650, -2160, 0
            self.game_audio()
            self.drawing = False
            self.capturing = False
            self.placing = False
            self.funct = None

        def game_audio(self):
            pg.mixer.init()
            pg.mixer.music.load("assets/sb_aurora.mp3")
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
                self.screen.blit(self.font_large.render("PAUSED", 1, (0,255,0)), (self.s_width//2 - 46, 55))
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
                            case pg.K_1: self.update_delay = 100
                            case pg.K_2: self.update_delay = 50
                            case pg.K_3: self.update_delay = 25
                            case pg.K_4: self.update_delay = 13
                            case pg.K_5: self.update_delay = 8
                            case pg.K_6: self.update_delay = 5
                            case pg.K_7: self.update_delay = 3
                            case pg.K_8: self.update_delay = 2
                            case pg.K_9: self.update_delay = 1
                            case pg.K_c:
                                self.x_offset = 0
                                self.y_offset = 0
                                game_matrix.render_grid()
                            case pg.K_m:
                                if self.muted:
                                    self.muted = False
                                else:
                                    self.muted = True
                            case pg.K_p: 
                                # print(Matrix.get_patterns(game_matrix))
                                print(game_matrix.saved_patterns[Submatrix.page_num][0].submatrix)
                            case pg.K_r:
                                Matrix.random_noise(game_matrix)
                            case pg.K_x:
                                Matrix.clear_board()
                            
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.funct:
                            self.funct()

                        if game.placing:
                            game.drawing = False
                            mouse_x, mouse_y = pg.mouse.get_pos()
                            if mouse_x <= 1600 or mouse_y>= 600:

                                if mouse_x >= self.x_offset and mouse_y >= self.y_offset and mouse_y < game.s_height * .91 and BtnImg.sliding == False:
                                    # try:
                                    #     self.paused = True
                                    #     x_rounded = (mouse_x - self.x_offset) / game_matrix.cell_size_current
                                    #     y_rounded = (mouse_y - self.y_offset) / game_matrix.cell_size_current
                                    #     # game_matrix.place_pos = [int(y_rounded),int(x_rounded)]
                                    #     Submatrix.place(int(x_rounded,int(y_rounded)))
                                    # except:
                                    #     print("out of range - place")

                                    self.paused = True
                                    x_rounded = int((mouse_x - self.x_offset) / game_matrix.cell_size_current)
                                    y_rounded = int((mouse_y - self.y_offset) / game_matrix.cell_size_current)
                                    # game_matrix.place_pos = [int(y_rounded),int(x_rounded)]
                                    Submatrix.place(x_rounded,y_rounded)
                
                    if name_input.input_rect.collidepoint(event.pos):
                        TextInput.input_active = True
                    else:
                        TextInput.input_active = False
                        name_input.color = name_input.color_passive

                if game_gui.menu_open:
                    new_volume = slider_volume_ball.slider(slider_volume, event, game.volume)
                    new_speed = slider_speed_ball.slider(slider_speed, event, game.update_delay)
                    if new_volume != None:
                        pg.mixer.music.set_volume(new_volume)
                    if new_speed != None:
                        if new_speed == 0:
                            new_speed += 1
                        self.update_delay = 100 - new_speed

                if self.capturing == True:
                    game_matrix.capture_patterns(event)

                if TextInput.input_active == True:
                    name_input.color = name_input.color_active
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_BACKSPACE:
                            name_input.user_text = name_input.user_text[:-1]
                        else:
                            name_input.user_text += event.unicode




        def key_hold_events(self):
            keys = pg.key.get_pressed()
            
            if self.count % self.zoom_delay == 0:

                if keys[pg.K_MINUS]:
                    
                    #move adjustment
                    self.move_delay = int(game_matrix.cell_size_current // 4)
                    if self.move_delay < 1: self.move_delay = 1

                    if game_matrix.cell_size_current > game_matrix.cell_size_min:

                        col_center_1 = ((self.s_width / game_matrix.cell_size_current) / 2) - (self.x_offset / game_matrix.cell_size_current)
                        row_center_1 = ((self.s_height / game_matrix.cell_size_current) / 2) - (self.y_offset / game_matrix.cell_size_current)

                        game_matrix.cell_size_current -= 1

                        col_center_2 = ((self.s_width / game_matrix.cell_size_current) / 2) - (self.x_offset / game_matrix.cell_size_current)
                        row_center_2 = ((self.s_height / game_matrix.cell_size_current) / 2) - (self.y_offset / game_matrix.cell_size_current)

                        difference_x = (col_center_1 - col_center_2) * game_matrix.cell_size_current
                        difference_y = (row_center_1 - row_center_2) * game_matrix.cell_size_current

                        self.x_offset -= difference_x
                        self.y_offset -= difference_y

                        game_matrix.render_grid()

                if keys[pg.K_EQUALS]:

                    #move adjustment
                    self.move_delay = int(game_matrix.cell_size_current // 4)
                    if self.move_delay < 1: self.move_delay = 1

                    if game_matrix.cell_size_current < game_matrix.cell_size_max:

                        col_center_1 = ((self.s_width / game_matrix.cell_size_current) / 2) - (self.x_offset / game_matrix.cell_size_current)
                        row_center_1 = ((self.s_height / game_matrix.cell_size_current) / 2) - (self.y_offset / game_matrix.cell_size_current)


                        game_matrix.cell_size_current += 1

                        col_center_2 = ((self.s_width / game_matrix.cell_size_current) / 2) - (self.x_offset / game_matrix.cell_size_current)
                        row_center_2 = ((self.s_height / game_matrix.cell_size_current) / 2) - (self.y_offset / game_matrix.cell_size_current)

                        difference_x = (col_center_1 - col_center_2) * game_matrix.cell_size_current
                        difference_y = (row_center_1 - row_center_2) * game_matrix.cell_size_current
                        self.x_offset -= difference_x
                        self.y_offset -= difference_y

                        game_matrix.render_grid()

            if self.count % self.move_delay == 0:

                if keys[pg.K_UP]:
                    if self.y_offset < self.s_height / 2:
                        self.y_offset += game_matrix.cell_size_current * game.fps_correction
                        game_matrix.render_grid()

                if keys[pg.K_RIGHT]:
                    # if self.x_offset > game_matrix.boundary_x[game_matrix.cell_size_current]:
                    self.x_offset -= game_matrix.cell_size_current * game.fps_correction
                    game_matrix.render_grid()

                if keys[pg.K_LEFT]:
                    if self.x_offset < self.s_width / 2:
                        self.x_offset += game_matrix.cell_size_current * game.fps_correction
                        game_matrix.render_grid()
                        
                if keys[pg.K_DOWN]:
                    # if self.y_offset > game_matrix.boundary_y[game_matrix.cell_size_current]:
                    self.y_offset -= game_matrix.cell_size_current * game.fps_correction
                    game_matrix.render_grid()

        def mouse_events(self):
            click = pg.mouse.get_pressed()
            mouse_x, mouse_y = pg.mouse.get_pos()

            if game.drawing:
                if click[0]:
                    if mouse_x >= self.x_offset and mouse_y >= self.y_offset and mouse_y < game.s_height * .91 and BtnImg.sliding == False:
                        try:
                            self.paused = True
                            x_rounded = (mouse_x - self.x_offset) / game_matrix.cell_size_current
                            y_rounded = (mouse_y - self.y_offset) / game_matrix.cell_size_current
                            game_matrix.matrix_current[int(y_rounded),int(x_rounded)] = True
                        except:
                            print("out of range - draw")
                elif click[2] and mouse_x >= self.y_offset and mouse_y >= self.y_offset and mouse_y < game.s_height * .91:
                    if mouse_x >= self.x_offset and mouse_y >= self.y_offset:
                        try:
                            self.paused = True
                            x_rounded = (mouse_x - self.x_offset) / game_matrix.cell_size_current
                            y_rounded = (mouse_y - self.y_offset) / game_matrix.cell_size_current
                            game_matrix.matrix_current[int(y_rounded),int(x_rounded)] = False
                        except:
                            print("out of range - draw")



    game = Game()

    class Gui():
        def __init__(self):
            self.opaque_level = 0
            self.opaque_level_fast = 0
            self.menu_open = False
            self.library_open = False

        def game_menu(self):
            mouse_x, mouse_y = pg.mouse.get_pos()
            if mouse_y > game.s_height * .75:
                self.menu_open = True
                if self.opaque_level < 252:
                    self.opaque_level += 3 * game.fps_correction
            else:
                self.menu_open = False
                if self.opaque_level > 0:
                    self.opaque_level -= 15 * game.fps_correction

            game_bar.render(self.opaque_level)
            btn_play.render(self.opaque_level)
            icon_speed.render(self.opaque_level)
            slider_speed.render(self.opaque_level)

            icon_sound.render(self.opaque_level)
            slider_volume.render(self.opaque_level)
            slider_speed_ball.render(self.opaque_level)
            slider_volume_ball.render(self.opaque_level)

            icon_capture.render(self.opaque_level)
            icon_draw.render(self.opaque_level)
            icon_wipe.render(self.opaque_level)
            icon_library.render(self.opaque_level)

        
            population = str(np.count_nonzero(game_matrix.matrix_current))
            population_surf = game.font_med.render("Population: " + population, 1, (0,255,0))
            population_surf.set_alpha(self.opaque_level)
            game.screen.blit(population_surf, (50, 20))

            if game_matrix.flashing == True:
                game_matrix.screen_flash()

        def pauseSwitch():
            if game.paused:
                game.paused = False
            else:
                game.paused = True
            game.pause(game.paused)

        def drawSwitch():
            if game.drawing:
                game.drawing = False
                icon_draw.img_path = 'assets/draw.png'
            else:
                game.drawing = True
                game.placing = False
                game_gui.library_open = False
                icon_draw.img_path = 'assets/drawLit.png'
            game.pause(game.paused)

        def capture_switch():
            if game.capturing == True:
                game.capturing = False
                game.drawing = True
                game_matrix.saved = True
            else:
                game.capturing = True
                game.drawing = False
                game_matrix.saved = False

        # def horizontal_slider(self, line, circle):
        #     min = 0
        #     max = line.width
        #     click = pg.mouse.get_pressed()
        #     mouse_x, mouse_y = pg.mouse.get_pos()
        #     if click[0]:
        #         if mouse_x > self.x:

        def library_switch(self):
            if self.library_open == False:
                self.library_open = True
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
            self.x_pos = (game.s_width // 2) + screen_offset_x
            self.y_pos = (game.s_height // 2) + screen_offset_y
            self.hover_rect = pg.draw.rect(game.screen, self.border_color, (self.x_pos,self.y_pos, self.width,self.height), border_radius= self.border_radius)
            game.screen.blit(self.surf, (self.x_pos, self.y_pos))
            mouse_pos = pg.mouse.get_pos()

            if self.hover_rect.collidepoint(mouse_pos):
                self.current_color = self.font_color_hover
                game.funct = self.btn_funct
            else:
                self.current_color = self.font_color
            game.screen.blit(game.font_small.render(self.text, 1, self.current_color), (self.x_pos + self.padding + (self.extra_width//2), self.y_pos + self.padding + (self.extra_height//2)))


    class BtnImg():
        sliding = False

        def __init__(self, img_path: str, img_hover_path: str, transparency: bool, scale, x, y, function, alpha = 255):
            if transparency == True:
                img = pg.image.load(img_path).convert_alpha()
                img.set_alpha(alpha)
                img_hover = pg.image.load(img_hover_path).convert_alpha()
                img_hover.set_alpha(alpha)
            else:
                img = pg.image.load(img_path).convert()
                img_hover = pg.image.load(img_hover_path).convert()
            self.img = pg.transform.scale(img, (int(scale*img.get_width()), int(scale*img.get_height())))
            self.img_hover = pg.transform.scale(img_hover, (int(scale*img.get_width()), int(scale*img.get_height())))
            self.width = img.get_width() * scale
            self.height = img.get_height() * scale
            self.x = x
            self.y = y 
            self.slider_min = x
            self.rect = self.img.get_rect()
            self.rect.topleft = (x, y)
            self.btn_funct = function
            
        def render(self, transparency):
            mouse_pos = pg.mouse.get_pos()
            self.img.set_alpha(transparency)
            self.img_hover.set_alpha(transparency)
            if self.rect.collidepoint(mouse_pos):
                self.img_render = self.img_hover
                game.funct = self.btn_funct
            else:
                self.img_render = self.img
            game.screen.blit(self.img_render, (self.x, self.y))

        def test():
            print("pressed")

        def slider(self, line, event, to_change):
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
                        if to_change < 1:
                            to_change = new_value / 100
                        else:
                            to_change = int(new_value)
                        return to_change
              

    game_bar = BtnImg('assets/gameBar.png', 'assets/gameBar.png', True, 1, 0, game.s_height - 120, BtnImg.test)
    
    btn_play = BtnImg('assets/play.png', 'assets/playLit.png', True, .42, game.s_width//2 - 28, game.s_height - 72, Gui.pauseSwitch)
    # btn_play2 = BtnImg('assets/play.png', 'assets/playLit.png', True, 1, game.s_width//2 - 64, game.s_height//2, Gui.pauseSwitch)

    icon_speed = BtnImg('assets/speed.png','assets/speed.png', True, .11, game.s_width//2 - 467, game.s_height-60, Gui.pauseSwitch)
    slider_speed = BtnImg('assets/sliderLit.png','assets/sliderLit.png', True, .3, game.s_width//2 - 420, game.s_height -49, BtnImg.test)
    slider_speed_ball = BtnImg('assets/sliderBall.png','assets/sliderBallLit.png', True, .4, game.s_width//2 - 370, game.s_height - 58, BtnImg.test)

    icon_sound = BtnImg('assets/soundLoud.png','assets/soundLoud.png', True, .25, game.s_width//2 - 720, game.s_height - 61, BtnImg.test)
    slider_volume = BtnImg('assets/sliderLit.png','assets/sliderLit.png', True, .3, game.s_width//2 - 685, game.s_height-49, BtnImg.test)
    slider_volume_ball = BtnImg('assets/sliderBall.png','assets/sliderBallLit.png', True, .4, game.s_width//2 - 635, game.s_height - 58, BtnImg.test)

    icon_draw = BtnImg('assets/draw.png','assets/drawLit.png', True, .28,  game.s_width//2 + 310, game.s_height - 68, Gui.drawSwitch)
    icon_capture = BtnImg('assets/captureIcon.png', 'assets/captureIconLit.png', True, .57, game.s_width//2 + 412, game.s_height - 68, Gui.capture_switch)
    icon_library = BtnImg('assets/library.png','assets/libraryLit.png', True, .42, game.s_width//2 + 518, game.s_height - 71, game_gui.library_switch)


    class Matrix:
        def __init__(self, startCellSize, cell_size_min, cell_size_max):
            self.cell_size_min = cell_size_min
            self.cell_size_current = startCellSize
            self.initialCellSize = startCellSize
            self.cell_size_max = cell_size_max
            self.rows = game.s_height // cell_size_min    
            self.cols = game.s_height // cell_size_min
            self.indices = self.all_indices()
            self.matrix_blank = self.create_matrix()
            self.matrix_current = self.matrix_blank
            self.angle = 1
            self.color_bg = [0,0,0]
            self.color_grid_init = [120,15,195]
            self.color_cell = (255,255,255)
            self.boundary_x = self.camera_bounds(game.s_width)
            self.boundary_y = self.camera_bounds(game.s_height)
            self.shades_grid = self.create_grid_shades(self.cell_size_min, .973, self.color_grid_init)
            self.surf_bg = pg.Surface((game.s_width, game.s_height))
            self.surf_box = pg.Surface((game.s_width, game.s_height))
            # self.submatrix_surf = pg.Surface((game.s_width//2, game.s_height//2))
            # self.submatrix_cellsize = 10

            self.saved_patterns = {
                1 : [Submatrix(np.array([[0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0],[0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1],[1,1,0,1,0,1,0,0,1,1,0,0,1,0,1,0,1,1],[0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,0],[0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,0],[1,1,0,1,0,1,0,0,1,1,0,0,1,0,1,0,1,1],[1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0],[0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0]])), "101"],
                2 : [Submatrix(np.array([[0,0,0,1,0,0],[0,1,0,1,0,0],[0,0,0,0,0,1],[1,1,1,1,1,0],[0,0,0,0,0,1],[0,1,0,1,0,0],[0,0,0,1,0,0]])), "Butterfly"],
                3 : [Submatrix(np.array([[0,0,0,0,1,1], [0,0,0,1,0,1], [0,0,0,1,0,0], [0,1,1,1,0,0], [1,0,0,0,0,0], [1,1,0,0,0,0]])), "Elevener"],
                4 : [Submatrix(np.array([[1,1,1,0,0,0], [1,1,1,0,0,0], [1,1,1,0,0,0], [0,0,0,1,1,1], [0,0,0,1,1,1], [0,0,0,1,1,1]])), "Figure Eight"],
                5 : [Submatrix(np.array([[0,0,0,0,0,0,1],[0,0,1,0,1,0,1],[0,0,0,0,1,0,1],[1,1,0,0,0,0,0],[0,0,1,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,0,1,0,0]])), "Fox"],
                6 : [Submatrix(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1], [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1], [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])), "Glider Gun"],
                7 : [Submatrix(np.array([[0,0,0,0,1,0,0],[0,0,1,0,1,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,1,0,1,0,0],[0,0,1,0,0,0,0]])), "Lei"],
                8 : [Submatrix(np.array([[0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]])), "Lightweight Spaceship"],
                9 : [Submatrix(np.array([[0,0,0,0,1,0,0],[0,0,1,0,1,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,1,0,1,0,0],[0,0,1,0,0,0,0]])), "Mazing"],
                10 : [Submatrix(np.array([[1,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,  1,1]])), "Methuselah"],
                11 : [Submatrix(np.array([[0,0,0,1,1,0,0,0], [0,0,1,0,0,1,0,0], [0,1,0,0,0,0,1,0], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,0,0,0,0,1,0], [0,0,1,0,0,1,0,0], [0,0,0,1,1,0,0,0]])), "Octagon 2"],
                12 : [Submatrix(np.array([[0,0,1,0,0,0,0,1,0,0], [1,1,0,1,1,1,1,0,1,1], [0,0,1,0,0,0,0,1,0,0]])), "Pentadecathlon"],                
                13 : [Submatrix(np.array([[0,0,1,1,1,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [0,0,1,1,1,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,1,1,0,0,0,1,1,1,0,0], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,1,1,0,0,0,1,1,1,0,0]])), "Pulsar"],
                14 : [Submatrix(np.array([[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1],[1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1],[1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])),"Queen Bee Shuttle"],
                15 : [Submatrix(np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,], [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,], [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,], [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,], [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,], [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0,], [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,], [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1,], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,], [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,], [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0,], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,], [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0,], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,], [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,], [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,], [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,], [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,]])),"Super Mario"]
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

        def create_matrix(self):
            return np.zeros([self.rows,self.cols], dtype = int)

        def create_grid_shades(self, cell_size_min, change, colors)->dict:
            shades_grid = {}
            for i in range(self.cell_size_max,cell_size_min - 1, -1):
                r = colors[0] * change
                g = colors[1] * change
                b = colors[2] * change
                colors = [r,g,b]
                shades_grid[i] = colors
            return shades_grid

        def render_grid(self):
            color_grid = self.shades_grid[self.cell_size_current]
            self.surf_bg.blit(game.bg,(0,0)) 
            pg.draw.rect(self.surf_bg, game_matrix.color_bg, (game.x_offset,game.y_offset,self.cell_size_current * self.cols ,self.cell_size_current * self.rows))
            for row in range(self.rows + 1):
                pg.draw.line(self.surf_bg, color_grid, (game.x_offset,row * self.cell_size_current + game.y_offset), (self.cols * self.cell_size_current + game.x_offset,row * self.cell_size_current + game.y_offset))
            for col in range(self.cols + 1):
                pg.draw.line(self.surf_bg, color_grid, (col * self.cell_size_current + game.x_offset,game.y_offset), (col * self.cell_size_current + game.x_offset, self.rows * self.cell_size_current + game.y_offset))

        @staticmethod
        @njit
        def update_matrix(matrix,matrix_new,cells_new,indices):
            for index in indices:
                row = index[0]
                col = index[1]
                top = max(0, row-1)
                left = max(0, col-1)
                alive_count = np.sum(matrix[top:row+2,left:col+2]) - matrix[row,col]
                if alive_count == 2: 
                    matrix_new[row,col] = matrix[row,col]
                elif alive_count == 3:
                    matrix_new[row,col] = 1
                    if matrix[row,col] == 0:
                        cells_new[row,col] = 1
                else:
                    matrix_new[row,col] = 0
            return matrix_new
        
        def render_cells(self):
            cell_size = self.cell_size_current
            # picture = pg.transform.scale(cellObj.img, (cell_size-1, cell_size-1))
            alive_indices = np.nonzero(self.matrix_current)
            # newIndices = np.nonzero(cells_new)
            for row, col in zip(*alive_indices):
                if self.matrix_current[row,col] == 1:
                    pg.draw.rect(game.screen,self.color_cell,(col * cell_size + game.x_offset, row * cell_size + game.y_offset,cell_size-1,cell_size-1))

        def all_indices(self):
            indices = np.ndindex(self.rows, self.cols)
            indices_placed = []
            for index in indices:
                indices_placed.append(index)
            self.indices = np.array(indices_placed)
            return self.indices

        def camera_bounds(self, dimension):
            leading_offset = dimension // self.cell_size_min
            offsets = {self.cell_size_min : 0}
            for count, size in enumerate(range(self.cell_size_min + 1, self.cell_size_max)):
                offsets[size] = leading_offset * -(count + 1)
            return offsets

        def random_noise(self):
            row = random.randint(self.rows - 2)
            col = random.randint(self.cols - 2)
            self.matrix_current[row-1,col-1] = True
            self.matrix_current[row-1,col] = True
            self.matrix_current[row,col] = True
            self.matrix_current[row,col+1] = True
            self.matrix_current[row+1,col] = True


        def capture_patterns(self, event):

            if event.type == pg.MOUSEBUTTONDOWN:
                
                if event.button == 1:     
                    self.mouse_down = True
                    #cell selection logic must differ from display logic
                    self.start_col = int((event.pos[0] - game.x_offset) / game_matrix.cell_size_current)
                    self.start_row = int((event.pos[1] - game.y_offset) / game_matrix.cell_size_current)

                    if self.start_col < 0:
                        self.start_col = 0
                    if self.start_row < 0:
                        self.start_row = 0
                    self.capture_x = event.pos[0]
                    self.capture_y = event.pos[1]


            if event.type == pg.MOUSEBUTTONUP:
                #don't allow matrices smaller than 4x4
                #need start, end


                if event.button == 1:
                    self.mouse_down = False
                    game.capturing == False

                    self.end_col = int((event.pos[0] - game.x_offset) / game_matrix.cell_size_current)
                    self.end_row = int((event.pos[1] - game.y_offset) / game_matrix.cell_size_current)

                    if self.end_col < 0:
                        self.end_col = 0
                    if self.end_row < 0:
                        self.end_row = 0

                    row_range =  abs(self.end_row - self.start_row) + 1
                    col_range =  abs(self.end_col - self.start_col) + 1

                    if row_range > 2 and col_range > 2:

                        #Quadrant 1
                        if self.start_col > self.end_col and self.start_row > self.end_row:
                            submatrix = game_matrix.matrix_current[self.end_row:self.start_row:+1,self.end_col:self.start_col+1]
                        #Quadrant 2
                        elif self.start_col < self.end_col and self.start_row > self.end_row:
                            submatrix = game_matrix.matrix_current[self.end_row:self.start_row+1, self.start_col:self.end_col+1]
                        #Quadrant 3
                        elif self.start_col > self.end_col and self.start_row < self.end_row:
                            submatrix = game_matrix.matrix_current[self.start_row:self.end_row+1,self.end_col:self.start_col+1]
                        #Quadrant 4
                        else:
                            submatrix = game_matrix.matrix_current[self.start_row:self.end_row+1,self.start_col:self.end_col+1]

                        if np.count_nonzero(submatrix) != 0:
                                self.current_submatrix = Submatrix(game_matrix.matrix_trim(submatrix))
        

                        self.flashing = True
                                         


            if event.type == pg.MOUSEMOTION and self.mouse_down:

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


        def matrix_trim(self, matrix):
            alive_indices = np.nonzero(matrix)
            min_indices = [np.subtract(alive_indices[0],np.amin(alive_indices[0])),np.subtract(alive_indices[1],np.amin(alive_indices[1]))]

            new_rows = np.amax(min_indices[0]) +1
            new_cols = np.amax(min_indices[1]) +1


            trimmed_matrix = np.zeros(shape=(new_rows,new_cols))
            for row, col in zip(*min_indices):
                trimmed_matrix[row,col] = True
            trimmed_matrix = np.array(trimmed_matrix, dtype='int16')


            return trimmed_matrix

        def accepted(self):
            self.confirmed = True
                   
        def rejected(self):
            self.confirmed = False
            self.capturing = False

        def screen_flash(self):
            pg.mixer.Channel(0).play(game.flash)

            # if self.flash_alpha > 10:
            #     self.flash_alpha -= 10
            # flash_surf = pg.Surface((game.s_width, game.s_height)).convert_alpha()
            # flash_surf.fill((255, 255, 255, self.flash_alpha))
            # game.screen.blit(flash_surf, (0,0))
            
            # if game.count % 10 == 0:
            self.flashing = False
                # self.flash_alpha = 150

        def clear_board():
            game_matrix.matrix_current = Matrix.create_matrix(game_matrix)

        def get_patterns(self):
            return np.array(np.where(self))

        def load_pattern_old(self, indexArr):
            for index in indexArr:
                self.matrix_current[index[0],index[1]] = True

        def load_pattern(self, pattern):
            for row, col in zip(*pattern):
                self.matrix_current[row,col] = True
            return


    class Submatrix:
        saved_patterns = {}
        page_num = 1

        def __init__(self, np_submatrix, width_center=game.s_width//2, height_center=game.s_height//2):
            self.submatrix = np_submatrix
            self.rows, self.cols = self.submatrix.shape
            self.grid_color = [80,80,80]
            self.bg_color = (100,100,100)
            self.cell_color = (0,255,0)
            self.width_center = width_center
            self.height_center = height_center

            #CHECK LATER
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
            else:
                self.lib_cell_size = self.section_width // self.cols

            if self.lib_cell_size < 1:
                self.lib_cell_size = 1

            self.sect_submatrix_w = self.lib_cell_size * self.cols
            self.sect_submatrix_h = self.lib_cell_size * self.rows

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


        #EDIT LATER
        def refresh(self, np_submatrix, width_center=game.s_width//2, height_center=game.s_height//2):
            self.rows, self.cols = self.submatrix.shape

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
            else:
                self.lib_cell_size = self.section_width // self.cols

            
            if self.lib_cell_size < 1:
                self.lib_cell_size = 1

            self.sect_submatrix_w = self.lib_cell_size * self.cols
            self.sect_submatrix_h = self.lib_cell_size * self.rows

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
                    pg.draw.line(self.submatrix_surf, self.grid_color, (0,row * self.cell_size), (self.submatrix_width,row * self.cell_size))
                for col in range(self.cols):
                    pg.draw.line(self.submatrix_surf, self.grid_color, (col * self.cell_size,0), (col * self.cell_size,self.submatrix_height))
           
            game.screen.blit(self.submatrix_border, (self.width_center - (self.submatrix_width // 2) - self.border_width,self.height_center - (self.submatrix_height // 2) - self.border_width - 50))
            game.screen.blit(self.submatrix_surf, (self.width_center - (self.submatrix_width // 2),self.height_center - (self.submatrix_height // 2)- 50))


        def library_submatrix(self):
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

            game.screen.blit(self.section_border_surf, ((library_window.x_pos + self.x_offset) - self.lib_border_width, (library_window.y_pos + 20) - self.lib_border_width))
            game.screen.blit(self.section_surf, (library_window.x_pos + self.x_offset,library_window.y_pos + self.y_offset + 20))



        def save():
            game_matrix.capture_count += 1
            game_matrix.saved_patterns[game_matrix.capture_count] = [game_matrix.current_submatrix, name_input.user_text]
            name_input.user_text = ""
            game_matrix.saved = True
            game_matrix.confirmed = False
            game_matrix.current_submatrix = None
            
            submatrices = []
            for i in game_matrix.saved_patterns:
                submatrices.append(game_matrix.saved_patterns[i][0])

            #testing:
            # for i in game_matrix.saved_patterns:
            #     print("pattern:")
            #     print(game_matrix.saved_patterns[i][0].submatrix)  
            

        
        def render_library():
            game_matrix.saved_patterns[Submatrix.page_num][0].library_submatrix()

        def next_page():
            total_pages = len(game_matrix.saved_patterns)

            if Submatrix.page_num < total_pages:
                Submatrix.page_num += 1
            elif Submatrix.page_num == total_pages:
                Submatrix.page_num = 1

        def prev_page():
            total_pages = len(game_matrix.saved_patterns)
            if Submatrix.page_num > 1:
                Submatrix.page_num -= 1
            elif Submatrix.page_num == 1:
                Submatrix.page_num = total_pages

        def rotate():
            matrix = game_matrix.saved_patterns[Submatrix.page_num][0].submatrix
            game_matrix.saved_patterns[Submatrix.page_num][0].submatrix  = np.rot90(matrix,1)
            Submatrix.refresh(game_matrix.saved_patterns[Submatrix.page_num][0],game_matrix.saved_patterns[Submatrix.page_num][0].submatrix)

        def flip():
            matrix = game_matrix.saved_patterns[Submatrix.page_num][0].submatrix
            game_matrix.saved_patterns[Submatrix.page_num][0].submatrix  = np.fliplr(matrix)
            Submatrix.refresh(game_matrix.saved_patterns[Submatrix.page_num][0],game_matrix.saved_patterns[Submatrix.page_num][0].submatrix)


        def place(x,y):
            game.drawing = False
            sub_obj = game_matrix.saved_patterns[Submatrix.page_num][0]
            sub = sub_obj.submatrix
            sub_height, sub_width = sub.shape
            adjusted_pos = np.subtract((x,y), (sub_width // 2,sub_height // 2))
            col, row = adjusted_pos

            #top, and left edge cases:
            if col < 0:
                sub = sub[:,abs(col):]
                col = 0
            if row < 0:
                sub = sub[abs(row):,:]
                row = 0

            game_matrix.matrix_current[row:row+sub.shape[0], col:col+ sub.shape[1]] += sub


    next_icon = BtnRect(" Next ", Submatrix.next_page)
    prev_icon = BtnRect(" Prev ", Submatrix.prev_page)
    rotate_btn = BtnRect("Rotate", Submatrix.rotate)
    flip_btn = BtnRect(" Flip ", Submatrix.flip)
    place_icon = BtnRect("   Place   ", Gui.place_switch)



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
            self.border_rect = pg.draw.rect(self.surf, self.border_color, (0,0, self.width,self.height), border_radius= self.border_radius)
            self.fill_rect = pg.draw.rect(self.surf, self.fill_color, (self.border_width,self.border_width,self.width-(self.border_width*2),self.height-(self.border_width*2)))

        def edge_render(self, screen_offset_x, screen_offset_y):
            self.x_pos = game.s_width + screen_offset_x
            self.y_pos = game.s_height + screen_offset_y
            game.screen.blit(self.surf, (self.x_pos, self.y_pos))

        def center_render(self, screen_offset_x, screen_offset_y):
            self.x_pos = (game.s_width // 2) + screen_offset_x
            self.y_pos = (game.s_height // 2) + screen_offset_y
            game.screen.blit(self.surf, (self.x_pos, self.y_pos))

        def render_text(self, text, offset_x, offset_y, color):
            game.screen.blit(game.font_small.render(text, 1, color), (self.x_pos + offset_x, self.y_pos + offset_y))

            # game.screen.blit(game.font_med.render("FPS: " + str(display_fps), 1, (0,255,0)), (550,10))

    library_window = Window(350,510,3, (200,200,200))

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



    # class gameObject:
    #     def __init__(self, img_path: str, transparency: bool):
    #         self.opaque_level = 255
    #         if transparency == True:
    #             self.img = pg.image.load(img_path).convert_alpha()
    #             self.img.set_alpha(self.opaque_level)
    #         else:
    #             self.img = pg.image.load(img_path).convert()

    # cellObj = gameObject('assets/cube.png',True)
    # tileObj = gameObject('assets/tile.png',True)

    game_matrix = Matrix(10, 2, 60)


    icon_wipe = BtnImg('assets/wipe.png','assets/wipeLit.png', True, .28, game.s_width//2 + 618, game.s_height - 71,Matrix.clear_board)

    r_pentomino = np.array([[261,260],[261,261],[262,259],[262,260],[263,260]])
    game_matrix.load_pattern_old(r_pentomino)

    game_matrix.render_grid()


    def loop():

        while game.running:
            game.single_click_events()
            game.key_hold_events()
            game.mouse_events()

            game.screen.blit(game_matrix.surf_bg,(0,0))

            if game.count % game.update_delay == 0:
                cells_new = np.zeros([game_matrix.rows,game_matrix.cols], dtype = int)
                if not game.paused:
                    matrix_new = np.array(game_matrix.matrix_current)
                    matrix_new = game_matrix.update_matrix(game_matrix.matrix_current, matrix_new, cells_new, game_matrix.indices)
                    game_matrix.matrix_current = matrix_new
            if game.paused:
                game.pause(True)
                
            game_matrix.render_cells()
            
            if game.capturing and game_matrix.mouse_down:
                pg.draw.rect(game.screen, (0, 255, 0), (game_matrix.capture_display_x,game_matrix.capture_display_y, game_matrix.capture_display_w, game_matrix.capture_display_h), 1)

            if game_matrix.current_submatrix != None and game_matrix.saved == False:

                if game_matrix.confirmed == False:
                #CHANGE AFTER
                    game.capturing = False
                    confirm = Window(250,60,3)
                    # testwindow.center_render(-175,game_matrix.current_submatrix.submatrix_height//2)
                    confirm.center_render(-128,275)
                    confirm.render_text("Confirm Capture?",34,17, (0,255,0))

                    yes1 = BtnRect("Yes",game_matrix.accepted)
                    no1 = BtnRect("No",game_matrix.rejected,39)
                    yes1.render(-85,360)
                    no1.render(15,360)

                if game_matrix.confirmed == True:
                    save_btn = BtnRect("Save",Submatrix.save)
                    name_prompt = Window(470,100,3)
                    name_prompt.center_render(-235,275)
                    save_btn.render(115,325)
                    name_prompt.render_text("Confirmed. Please enter a name:",47,17,(0,255,0))
                    name_input.text_surf = game.font_small.render(name_input.user_text, True, name_input.text_color)
                    pg.draw.rect(game.screen, name_input.color, name_input.input_rect)
                    game.screen.blit(name_input.text_surf, (name_input.input_rect.x+10, name_input.input_rect.y+1))

                game_matrix.current_submatrix.render_submatrix()
                
        

            display_fps = game.clock.get_fps()
            if display_fps > 0:
                game.fps_correction = game.max_fps / display_fps
        

            #---FOR TESTING:---#
            # game.screen.blit(game.font_med.render("FPS: " + str(display_fps), 1, (0,255,0)), (250,300))
            # game.screen.blit(game.font_med.render("X/Y Offset: " + str(game.x_offset) + "," + str(game.y_offset), 1, (0,255,0)), (350,10))
            # game.screen.blit(game.font_med.render("number of cols: " + str(game.s_width // game_matrix.cell_size_current), 1, (0,255,0)), (350,50))
            # game.screen.blit(game.font_med.render("number of rows: " + str(game.s_height // game_matrix.cell_size_current), 1, (0,255,0)), (650,50))
            # game.screen.blit(game.font_med.render("cell size: " + str(game_matrix.cell_size_current), 1, (0,255,0)), (950,50))
            # game.screen.blit(game.font_med.render("center col/row:" + str(((game.s_width / game_matrix.cell_size_current) // 2) - (game.x_offset // game_matrix.cell_size_current)) + "," + str(((game.s_height / game_matrix.cell_size_current) // 2)  - (game.y_offset // game_matrix.cell_size_current)), 1, (0,255,0)), (650,80))
            # # game.screen.blit(game.font_med.render("dict offset: " + str(game_matrix.offsets[game_matrix.cell_size_current]), 1, (0,255,0)), (950,100))


            Gui.game_menu(game_gui)
            if game_gui.library_open == True:
                library_window.edge_render(-370,-1000)
                prev_icon.render(670,-95)
                next_icon.render(770,-95)
                rotate_btn.render(770,-58)
                flip_btn.render(670,-58)
                place_icon.render(693,-20)

                game_matrix.saved_patterns[Submatrix.page_num][0].library_submatrix()

                name_text = game.font_small.render(game_matrix.saved_patterns[Submatrix.page_num][1], 1, (0,255,0))
                center_text = (350 - name_text.get_width()) // 2
                game.screen.blit(name_text, (1550 + center_text,405))


            game.count += 1
            pg.display.flip()
            game.clock.tick(game.max_fps)



    game.game_loop()
