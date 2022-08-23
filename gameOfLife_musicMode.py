import pygame as pg
from pygame import midi

import itertools

import numpy as np
from numpy import random

from sys import exit
from numba import njit
from time import sleep
from pygame.locals import *

def run():
    #turns on pygame
    pg.init()
    pg.midi.init()
    instrument = 1
    musicKey = 1
    port = pg.midi.get_default_output_id()
    midi_out = pg.midi.Output(port, 0)

    #needs device id
    midi_in = pg.midi.Input(1)


    midiBank = {1:(4,"Electric Piano 1"),2:(12,"Marimba"),3:(14,"Tubular Bells"),4:(16,"Drawbar Organ"),5:(24,"Acoustic Guitar (nylon)"),6:(26,"Electric Guitar (jazz)"),7:(33,"Electric Bass (finger)"),8:(93,"Pad 6 (metallic)"),9:(101,"FX 6 (goblins)"),10:(108, "Kalimba")}

    #USED TO SET TO MONITORS RESOLUTION WHEN FULLSCREEN IS ENABLED
    displayInfo = pg.display.Info()
    sWidth = displayInfo.current_w
    sHeight = displayInfo.current_h



    flags = FULLSCREEN | DOUBLEBUF
    screen = pg.display.set_mode((sWidth, sHeight), flags, pg.FULLSCREEN)

    # sWidth = 996
    # sHeight = 996

    screen = pg.display.set_mode((sWidth,sHeight))
    pg.display.set_caption("Conway's Game of Life")

    #setting frame part 1
    clock = pg.time.Clock()

    #RGB colors
    black = (0,0,0)
    cellColor = (255,255,255)
    gridColor = [40,5,65]
    musicColor = (0,200,0)
    musicColor2 = (20,10,120)

    font = pg.font.Font('assets/square_pixel-7.ttf', 24)
    # gridImg = pg.image.load('bg.png')
    # cellImg = pg.image.load('cell2.png')


    drawRate = 10000
    gameRate = 10

    cellSize = 7
    minCellSize = 7
    maxCellSize = 100

    #determines number of cells
    #1000 / 20 = 200 
    rows = sHeight // minCellSize     
    cols = sWidth // minCellSize


    # activeSurf = pg.Surface((cellSize-1, cellSize-1))
    gridSurf = pg.Surface((sWidth,sHeight))
    # activeCell = pg.draw.rect(activeSurf, cellColor, (0,0,cellSize-1,cellSize-1))

    musicSurf = pg.Surface((cellSize-1, cellSize-1))
    musicSurf2 = pg.Surface((cellSize, cellSize))
    musicCell = pg.draw.rect(musicSurf, musicColor, (0,0,cellSize-1,cellSize-1))
    musicCell2 = pg.draw.rect(musicSurf2, musicColor2, (0,0,cellSize,cellSize))
    volume = 100

    def displayFPS():
        fps = str(int(clock.get_fps()))
        return font.render("FPS: " + fps, 1, (0,255,0))

    def createMatrix(rows, cols):
        return np.zeros([rows,cols], dtype = int)

    def renderGrid(rows, cols):
        pg.draw.rect(gridSurf, gridColor, (0,0,sWidth,sHeight))
        if cols > rows:
            for col in range(cols):
                for row in range(rows):
                    pg.draw.rect(gridSurf, black, (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
        else:
            for row in range(cols):
                for col in range(rows):
                    pg.draw.rect(gridSurf, black, (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
        return

    #design pattern maker
    @njit
    def getPatterns(matrix):
        return np.array(np.where(matrix))


    def randomNoise(matrix, rows, cols):
        row = random.randint(rows-2)
        col = random.randint(cols-2)
        matrix[row-1,col-1] = True
        matrix[row-1,col] = True
        matrix[row,col] = True
        matrix[row,col+1] = True
        matrix[row+1,col] = True

    
    indices = np.ndindex(rows, cols)
    xindices = []
    for index in indices:
        xindices.append(index)
    xindices = np.array(xindices)


    @njit
    def aliveNeighbors(indices):
        neighbors = []
        aliveIndices = np.nonzero(matrix)
        for row, col in zip(*aliveIndices):
            top = max(0, row-1)
            left = max(0, col-1)
            neighbors.append(indices[top:row+2,left:col+2])
        return
        

    @njit
    def updateMatrix(matrix,newMatrix,newCells,indices):
        for index in indices:
            row = index[0]
            col = index[1]
            top = max(0, row-1)
            left = max(0, col-1)

            #figures out how many live neighbors each cell has
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


    #may need to modify to be adaptable to screens with vertical resolutions
    def renderCells(matrix, muted, newCells, noteDict,noteColor):
        # picture = pg.transform.scale(cellImg, (cellSize, cellSize))
        aliveIndices = np.nonzero(matrix)
        newIndices = np.nonzero(newCells)
        for row, col in zip(*aliveIndices):
            if matrix[row,col] == 1:
                pg.draw.rect(screen,cellColor,(col*cellSize,row*cellSize,cellSize-1,cellSize-1))
                # screen.blit(picture,(col*cellSize,row*cellSize))
        if sequencer == False and muted == False:
            for row, col in zip(*newIndices):
                notesCell(row, col, cellSize, volume, noteDict, noteColor)



    # def wand(matrix, cellSize)):
    #     random.randint()
    #     click = pg.mouse.get_pressed()
    #     mouseX, mouseY = pg.mouse.get_pos()
    #     if click[0]:
    #         paused = True
    #         rate = drawRate
    #         roundedX = mouseX // cellSize
    #         roundedY = mouseY // cellSize
    #         matrix[roundedY,roundedX] = True
    #     elif click[2]:
    #         paused = True
    #         rate = drawRate
    #         roundedX = mouseX // cellSize
    #         roundedY = mouseY // cellSize
    #         matrix[roundedY,roundedX] = False
    #     else:
    #         rate = gameRate

    #know where its been clicked
    #generate an array of random numbers within a range of 



    matrix = createMatrix(rows,cols)

    # rPentomino = np.array([[28,30],[28,31],[29,29],[29,30],[30,30]])



    #r-pentomino pattern
    matrix[28,30] = True
    matrix[28,31] = True
    matrix[29,29] = True
    matrix[29,30] = True
    matrix[30,30] = True



    #MUSIC STUFF:
    #====================================================
    #pentatonic, diatonic, hirajoshi, harmonic minor, chromatic, blues, melodic minor, whole note, metalydian, doubleharmonic, minor thirds, major thirds, alt thirds
    scaleDict = {
        1:[127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21],
        2:[126, 124, 122, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 98, 97, 95, 93, 92, 90, 88, 86, 85, 83, 81, 80, 78, 76, 74, 73, 71, 69, 68, 66, 64, 62, 61, 59, 57, 56, 54, 52, 50, 49, 47, 45, 44, 42, 40, 38, 37, 35, 33, 32, 30, 28, 26, 25, 23, 21],
        3:[125, 124, 120, 119, 117, 113, 112, 108, 107, 105, 101, 100, 96, 95, 93, 89, 88, 84, 83, 81, 77, 76, 72, 71, 69, 65, 64, 60, 59, 57, 53, 52, 48, 47, 45, 41, 40, 36, 35, 33, 29, 28, 24, 23, 21],
        4:[125, 124, 122, 121, 119, 117, 116, 113, 112, 110, 109, 107, 105, 104, 101, 100, 98, 97, 95, 93, 92, 89, 88, 86, 85, 83, 81, 80, 77, 76, 74, 73, 71, 69, 68, 65, 64, 62, 61, 59, 57, 56, 53, 52, 50, 49, 47, 45, 44, 41, 40, 38, 37, 35, 33, 32, 29, 28, 26, 25, 23, 21],
        5:[127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21],
        6:[127, 124, 123, 122, 120, 117, 115, 112, 111, 110, 108, 105, 103, 100, 99, 98, 96, 93, 91, 88, 87, 86, 84, 81, 79, 76, 75, 74, 72, 69, 67, 64, 63, 62, 60, 57, 55, 52, 51, 50, 48, 45, 43, 40, 39, 38, 36, 33, 31, 28, 27, 26, 24, 21],
        7:[126, 124, 122, 120, 119, 117, 116, 114, 112, 110, 108, 107, 105, 104, 102, 100, 98, 96, 95, 93, 92, 90, 88, 86, 84, 83, 81, 80, 78, 76, 74, 72, 71, 69, 68, 66, 64, 62, 60, 59, 57, 56, 54, 52, 50, 48, 47, 45, 44, 42, 40, 38, 36, 35, 33, 32, 30, 28, 26, 24, 23, 21],
        8:[127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21],
        9:[126, 125, 123, 121, 119, 118, 116, 114, 112, 111, 109, 107, 105, 104, 102, 100, 98, 97, 95, 93, 91, 90, 88, 86, 84, 83, 81, 79, 77, 76, 74, 72, 70, 69, 67, 65, 63, 62, 60, 58, 56, 55, 53, 51, 49, 48, 46, 44, 42, 41, 39, 37, 35, 34, 32, 30, 28, 27, 25, 23, 21],
        10:[125, 124, 122, 121, 118, 117, 116, 113, 112, 110, 109, 106, 105, 104, 101, 100, 98, 97, 94, 93, 92, 89, 88, 86, 85, 82, 81, 80, 77, 76, 74, 73, 70, 69, 68, 65, 64, 62, 61, 58, 57, 56, 53, 52, 50, 49, 46, 45, 44, 41, 40, 38, 37, 34, 33, 32, 29, 28, 26, 25, 22, 21],
        11:[126, 123, 120, 117, 114, 111, 108, 105, 102, 99, 96, 93, 90, 87, 84, 81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30, 27, 24, 21],
        12:[125, 121, 117, 113, 109, 105, 101, 97, 93, 89, 85, 81, 77, 73, 69, 65, 61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21],
        13:[126, 123, 119, 116, 112, 109, 105, 102, 98, 95, 91, 88, 84, 81, 77, 74, 70, 67, 63, 60, 56, 53, 49, 46, 42, 39, 35, 32, 28, 25, 21]
        }

    scaleNames = {1:"Major Pentatonic",2:"Major Diatonic",3:"Hirajoshi",4:"Harmonic Minor",5:"Chromatic",6:"Blues",7:"Melodic Minor",8:"Whole Note",9:"Meta-lydian",10:"Double Harmonic Minor",11:"Minor Thirds",12:"Major Thirds",13:"Alt Thirds"}
    noteColors = {21: (255, 0, 0), 22: (103, 78, 167), 23: (255, 255, 0), 24: (166, 77, 121), 25: (207, 226, 243), 26: (177, 47, 69), 27: (123, 132, 255), 28: (255, 165, 0), 29: (209, 138, 221), 30: (0, 255, 0), 31: (186, 128, 128), 32: (111, 168, 220), 33: (255, 0, 0), 34: (103, 78, 167), 35: (255, 255, 0), 36: (166, 77, 121), 37: (207, 226, 243), 38: (177, 47, 69), 39: (123, 132, 255), 40: (255, 165, 0), 41: (209, 138, 221), 42: (0, 255, 0), 43: (186, 128, 128), 44: (111, 168, 220), 45: (255, 0, 0), 46: (103, 78, 167), 47: (255, 255, 0), 48: (166, 77, 121), 49: (207, 226, 243), 50: (177, 47, 69), 51: (123, 132, 255), 52: (255, 165, 0), 53: (209, 138, 221), 54: (0, 255, 0), 55: (186, 128, 128), 56: (111, 168, 220), 57: (255, 0, 0), 58: (103, 78, 167), 59: (255, 255, 0), 60: (166, 77, 121), 61: (207, 226, 243), 62: (177, 47, 69), 63: (123, 132, 255), 64: (255, 165, 0), 65: (209, 138, 221), 66: (0, 255, 0), 67: (186, 128, 128), 68: (111, 168, 220), 69: (255, 0, 0), 70: (103, 78, 167), 71: (255, 255, 0), 72: (166, 77, 121), 73: (207, 226, 243), 74: (177, 47, 69), 75: (123, 132, 255), 76: (255, 165, 0), 77: (209, 138, 221), 78: (0, 255, 0), 79: (186, 128, 128), 80: (111, 168, 220), 81: (255, 0, 0), 82: (103, 78, 167), 83: (255, 255, 0), 84: (166, 77, 121), 85: (207, 226, 243), 86: (177, 47, 69), 87: (123, 132, 255), 88: (255, 165, 0), 89: (209, 138, 221), 90: (0, 255, 0), 91: (186, 128, 128), 92: (111, 168, 220), 93: (255, 0, 0), 94: (103, 78, 167), 95: (255, 255, 0), 96: (166, 77, 121), 97: (207, 226, 243), 98: (177, 47, 69), 99: (123, 132, 255), 100: (255, 165, 0), 101: (209, 138, 221), 102: (0, 255, 0), 103: (186, 128, 128), 104: (111, 168, 220), 105: (255, 0, 0), 106: (103, 78, 167), 107: (255, 255, 0), 108: (166, 77, 121), 109: (207, 226, 243), 110: (177, 47, 69), 111: (123, 132, 255), 112: (255, 165, 0), 113: (209, 138, 221), 114: (0, 255, 0), 115: (186, 128, 128), 116: (111, 168, 220), 117: (255, 0, 0), 118: (103, 78, 167), 119: (255, 255, 0), 120: (166, 77, 121), 121: (207, 226, 243), 122: (177, 47, 69), 123: (123, 132, 255), 124: (255, 165, 0), 125: (209, 138, 221), 126: (0, 255, 0), 127: (186, 128, 128)}
    flashColors = itertools.cycle(['red','orange','yellow','green','cyan','blue','purple','pink'])

    baseColor = next(flashColors)
    nextColor = next(flashColors)
    currentColor = baseColor
    colorRate = .5
    colorSteps = colorRate * gameRate
    step = 1

    def midi(notes, volume):
        for n in notes:
            midi_out.note_on(n, volume) # 74 is middle C, 127 is "how loud" - max is 127
        # if paused == True:
        #this keeps speed steady while drawing, but is undesirable 
        #it stops frames from jumping up
        sleep(.05)
        for n in notes:
            midi_out.note_off(n, volume)




    #takes active notes from matrix in the corresponding column, pushes to notes function with note dictionary, 
    #col-1 corrects an offset
    def playColumn(matrix, col, noteDict, volume, cellSize):
        playCol = matrix[:, col-1]
        return notes(playCol, noteDict, col-1, volume, cellSize)

    #generates list of notes to be played
    #takes values in the playCol, takes row from index for rectangle generation, 
    # if the rowCell is active, rect is drawn and the correct note grabbed from the noteDict is appended to notes list
    # these notes are passed into midi for playback  
    def notes(playCol, noteDict, col, volume, cellSize):
        notes = []
        for index, rowCell in np.ndenumerate(playCol):
            row = index[0]
            if rowCell == 1:
                # screen.blit(musicSurf,(col*cellSize,row*cellSize))
                pg.draw.rect(screen, musicColor, (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
                note = noteDict[row]
                notes.append(note)
            else: 
                screen.blit(musicSurf2,(col*cellSize,row*cellSize))
        return midi(notes, volume)



    #creates notesDict. Uses number of rows.
    def noteList(cols, scaleNum):
        notes = scaleDict[scaleNum]
        noteDict = {}

        #NEED TO BUILD VIA FOR LOOP, CREATE SO MUSIC ONLY PLAYS ON AREA IN ZOOMED CAMERA POSITION
        times = int(np.ceil((cols // len(notes)))) 
        notes = np.tile(notes,times+1)

        for row in range(cols):
            noteDict[row] = notes[row]
        return noteDict

    noteDict = noteList(cols, musicKey)


    #for music without sequencer
    def midi2(note, volume, length=1):
        midi_out.note_on(note, volume) # 74 is middle C, 127 is "how loud" - max is 127
        # sleep(length)
        # for n in notes: 
        #     midi_out.note_off(n, volume)
        if paused == True:
            print("pausedmidi2")
            for n in notes:
                midi_out.note_off(n, volume)

    def notesCell(row, col, cellSize, volume, noteDict, noteColor):
        note = noteDict[col]
        # # screen.blit(musicSurf2,(col*cellSize,row*cellSize))
        # pg.draw.rect(screen, noteColors[note], (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
        pg.draw.rect(screen, noteColor, (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
        midi2(note, volume)
        return

    # def colLight(col):
    #     playCol = matrix[:, col-1]
    #     for cell in col:
    #         if cell == 1:
    #             screen.blit(activeSurf,(col*cellSize,row*cellSize))

    def setDictionary(dict, keys, value):
        for key in keys:
            dict[key] = value



    sizeVol = {1:5, 2:20, 3:34, 4:46, 5:58, 6:69, 7:79,8:88, 9:96, 10:103,11:109,12:114,13:118,14:121,15:123,16:124,17:125,18:126,19:127} 
    setDictionary(sizeVol, range(20,101), 127)

    colNum = 0
    #====================================================

    muted = False
    sequencer = False
    seqStatus = "Off"
    drawColor = cellColor
    renderGrid(rows, cols)
    paused = True

    while 1:
        musicSurf2 = pg.Surface((cellSize, cellSize))
        musicCell2 = pg.draw.rect(musicSurf2, musicColor2, (0,0,cellSize,cellSize))
        newCells = createMatrix(rows,cols)
        midi_out.set_instrument(midiBank[instrument][0])

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
                            pg.midi.Output.close
                            paused = True
                    case pg.K_1:
                        gameRate = 1
                    case pg.K_2:
                        gameRate = 2
                    case pg.K_3:
                        gameRate = 3
                    case pg.K_4:
                        gameRate = 6
                    case pg.K_5:
                        gameRate = 10
                    case pg.K_6:
                        gameRate = 15
                    case pg.K_7:
                        gameRate = 22
                    case pg.K_8:
                        gameRate = 40
                    case pg.K_9:
                        gameRate = 100
                    # case pg.K_c:
                    #     myColor = colorchooser.askcolor()
                    #     cellColor = myColor[0]
                    case pg.K_m:
                        if muted:
                            muted = False
                        else:
                            muted = True
                    case pg.K_b:
                        if musicKey == 1:
                            musicKey = 14
                        musicKey -=1
                        noteDict = noteList(cols, musicKey)
                    case pg.K_n:
                        if musicKey == 13:
                            musicKey = 0
                        musicKey += 1
                        noteDict = noteList(cols, musicKey)
                    case pg.K_r:
                        randomNoise(matrix,rows,cols)
                    case pg.K_s:
                        if sequencer:
                            sequencer = False
                            seqStatus = "Off"
                        else:
                            sequencer = True
                            seqStatus = "On"
                    case pg.K_COMMA:
                        if instrument == 1:
                            instrument = 11
                        instrument -= 1
                    case pg.K_PERIOD:
                        if instrument == 10:
                            instrument = 0
                        instrument += 1
                    case pg.K_MINUS:
                        if cellSize > minCellSize:
                            cellSize -= 1
                            if cellSize < 10:
                                gridColor[0] = gridColor[0] * .9
                                gridColor[1] = gridColor[1] * .9
                                gridColor[2] = gridColor[2] * .9
                            volume = sizeVol[cellSize]
                            renderGrid(rows, cols)
                    case pg.K_EQUALS:
                        if cellSize < maxCellSize:
                            cellSize += 1
                            if cellSize < 10:
                                gridColor[0] = gridColor[0] * 1.1
                                gridColor[1] = gridColor[1] * 1.1
                                gridColor[2] = gridColor[2] * 1.1
                            volume = sizeVol[cellSize]
                            renderGrid(rows, cols)
                    case pg.K_ESCAPE:
                        pg.quit()
                        exit()
                
        


        #background color set
        # screen.fill(gridColor)
        screen.blit(gridSurf,(0,0))
        

        if not paused:
            # randomNoise(matrix)
            newMatrix = np.array(matrix)
            newMatrix = updateMatrix(matrix, newMatrix, newCells, xindices)
            matrix = newMatrix
        else:
            screen.blit(font.render("PAUSED", 1, (0,255,0)), (sWidth//2 - 20,sHeight - 100))
            muted = True
            pg.midi.Output.close
            pg.midi.Output.abort
            


        step += 1
        if step < colorSteps:
            currentColor = [x + (((y-x)/colorSteps)*step) for x, y in zip(pg.color.Color(baseColor),pg.color.Color(nextColor))]
        else:
            baseColor = nextColor
            nextColor = next(flashColors)
            step = 1

        renderCells(matrix, muted, newCells, noteDict,currentColor)
        

        click = pg.mouse.get_pressed()
        mouseX, mouseY = pg.mouse.get_pos()



        
        #draw
        #do not add cells that exceed board size
        #to speed up: turn into function, add blit
        if click[0]:
            paused = True
            # rate = drawRate
            roundedX = mouseX // cellSize
            roundedY = mouseY // cellSize
            matrix[roundedY,roundedX] = True
        elif click[2]:
            paused = True
            # rate = drawRate
            roundedX = mouseX // cellSize
            roundedY = mouseY // cellSize 
            matrix[roundedY,roundedX] = False
        else:
            rate = gameRate

        #music
        #NEED TO PROGRAM SEQUENCER TO ONLY LOOP OVER VISIBLE COLUMNS
        #enables column to cycle through screen
        if sequencer == True:
            playColumn(matrix, colNum, noteDict, volume, cellSize)
            colNum += 1
            if colNum == cols:
                colNum = 0


        aliveNeighbors(xindices)

        # screen.blit(displayFPS(), (10,30))
        population = str(np.count_nonzero(matrix))
        screen.blit(font.render("Population: " + population, 1, (0,255,0)), (25,10))
        # screen.blit(font.render("CellSize: " + str(cellSize), 1, (0,255,0)), (700,5))
        # screen.blit(font.render("Cells: " + str((sWidth / cellSize) * (sHeight / cellSize)), 1, (0,255,0)), (900,5))
        screen.blit(font.render("Instrument " + str(instrument), 1, (0,255,0)), (sWidth - 200,10))
        screen.blit(font.render("Sequencer: " + str(seqStatus), 1, (0,255,0)), (sWidth-1000,10))
        screen.blit(font.render("Musical Key " + str(musicKey) + ": " + scaleNames[musicKey], 1, (0,255,0)), (sWidth - 735,10))
        pg.display.update()
        clock.tick(rate)
