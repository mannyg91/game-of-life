import pygame as pg
from pygame import midi

import numpy as np
from numpy import random

from tkinter import *
from tkinter import colorchooser

from sys import exit
from numba import njit
from time import sleep

#turns on pygame
pg.init()
pg.midi.init()
instrument = 46
port = pg.midi.get_default_output_id()
midi_out = pg.midi.Output(port, 0)

midiBank = { 0 : "Acoustic Grand Piano", 1 : "Bright Acoustic Piano",2 : "Electric Grand Piano",3 : "Honky-tonk Piano",4 : "Electric Piano 1", 5 : "Electric Piano 2",6  : "Harpsichord",7 : "Clavinet",8 :  "Celesta",9 : "Glockenspiel",10 : "Music Box",11 : "Vibraphone",12 : "Marimba",13 : "Xylophone", 14 : "Tubular Bells", 15 : "Dulcimer", 16 : "Drawbar Organ", 17 : "Percussive Organ", 18 : "Rock Organ", 19 : "Church Organ", 20: "Reed Organ", 21: "Accordion", 22: "Harmonica", 23 : "Tango Accordion", 24 : "Acoustic Guitar (nylon)", 25 : "Acoustic Guitar (steel)", 26 : "Electric Guitar (jazz)", 27 : "Electric Guitar (clean)", 28 : "Electric Guitar (muted)", 29: "Overdriven Guitar", 30 : "Distortion Guitar", 31 : "Guitar harmonics", 32 : "Acoustic Bass", 33 : "Electric Bass (finger)", 34 : "Electric Bass (pick)", 35 : "Fretless Bass", 36 : "Slap Bass 1", 37 : "Slap Bass 2", 38 : "Synth Bass 1", 39: "Synth Bass 2", 40 : "Violin", 41 : "Viola", 42 : "Cello", 43 : "Contrabass", 44 : "Tremolo Strings", 45 : "Pizzicato Strings", 46 : "Orchestral Harp", 47 : "Timpani", 48: "String Ensemble 1", 49 : "String Ensemble 2", 50 : "Synth Strings 1", 51 : "Synth Strings 2", 52 : "Choir Aahs", 53 : "Voice Oohs", 54 : "Synth Voice", 55 : "Orchestra Hit", 56 : "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet", 60: "French Horn", 61: "Brass Section", 62 : "Synth Brass 1", 63 : "Synth Brass 2", 64: "Soprano Sax", 65: "Alto Sax", 66 : "Tenor Sax", 67: "Baritone Sax", 68 : "Oboe", 69 :  "English Horn", 70 : "Bassoon", 71 : "Clarinet", 72 : "Piccolo", 73: "Flute",74: "Recorder",75: "Pan Flute",76: "Blown Bottle",77: "Shakuhachi",78: "Whistle",79: "Ocarina",80: "Lead 1 (square)",81: "Lead 2 (sawtooth)",82: "Lead 3 (calliope)",83: "Lead 4 (chiff)",84: "Lead 5 (charang)",85: "Lead 6 (voice)",86: "Lead 7 (fifths)",87: "Lead 8 (bass + lead)",88: "Pad 1 (new age)",89: "Pad 2 (warm)",90: "Pad 3 (polysynth)",91: "Pad 4 (choir)",92: "Pad 5 (bowed)",93: "Pad 6 (metallic)",94: "Pad 7 (halo)",95: "Pad 8 (sweep)",96: "FX 1 (rain)",97: "FX 2 (soundtrack)",98: "FX 3 (crystal)",99: "FX 4 (atmosphere)",100: "FX 5 (brightness)",101: "FX 6 (goblins)",102: "FX 7 (echoes)",103: "FX 8 (sci-fi)",104: "Sitar",105: "Banjo",106: "Shamisen",107: "Koto",108: "Kalimba",109: "Bag pipe",110: "Fiddle",111: "Shanai",112: "Tinkle Bell",113: "Agogo",114: "Steel Drums",115: "Woodblock",116: "Taiko Drum",117: "Melodic Tom",118: "Synth Drum",119: "Reverse Cymbal",120: "Guitar Fret Noise",121: "Breath Noise",122: "Seashore",123: "Bird Tweet",124: "Telephone Ring", 125: "Helicopter",126: "Applause",127: "Gunshot" }

#USED TO SET TO MONITORS RESOLUTION WHEN FULLSCREEN IS ENABLED
displayInfo = pg.display.Info()
sWidth = displayInfo.current_w
sHeight = displayInfo.current_h


from pygame.locals import *
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
musicColor2 = (20,10,90)

font = pg.font.Font('square_pixel-7.ttf', 24)
gridImg = pg.image.load('bg.png')
cellImg = pg.image.load('cell2.png')


drawRate = 10000
gameRate = 10

cellSize = 10
minCellSize = 2
maxCellSize = 500

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


#may need to modify to be adaptable to screens with vertical resolutions
def renderCells(matrix, muted, newCells):
    # picture = pg.transform.scale(cellImg, (cellSize, cellSize))
    aliveIndices = np.nonzero(matrix)
    newIndices = np.nonzero(newCells)
    for row, col in zip(*aliveIndices):
        if matrix[row,col] == 1:
            pg.draw.rect(screen,cellColor,(col*cellSize,row*cellSize,cellSize-1,cellSize-1))
            # screen.blit(picture,(col*cellSize,row*cellSize))
    if sequencer == False and muted == False:
        for row, col in zip(*newIndices):
            notesCell(row, col, cellSize)



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


#updating matrix
# matrix[1,2] = True
# matrix[2,3] = True
# matrix[3,1] = True
# matrix[3,2] = True
# matrix[3,3] = True

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

#MUSIC STUFF:
#====================================================
def midi(notes, volume=127, length=.1):
    for n in notes:
        midi_out.note_on(n, volume) # 74 is middle C, 127 is "how loud" - max is 127

    #this keeps speed steady while drawing, but is undesirable 
    #it stops frames from jumping up
    sleep(length)
    for n in notes: 
        midi_out.note_off(n, volume)


def playColumn(matrix, col, noteDict):
    playCol = matrix[:, col-1]
    return notes(playCol, noteDict, col-1)

def notes(playCol, noteDict, col):
    notes = []
    for index, rowCell in np.ndenumerate(playCol):
        row = index[0]
        if rowCell == 1:
            # screen.blit(musicSurf,(col*cellSize,row*cellSize))
            pg.draw.rect(screen, musicColor, (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
            note = noteDict[index[0]]
            notes.append(note)
        else: 
            screen.blit(musicSurf2,(col*cellSize,row*cellSize))
    midi(notes)
    return

def noteList(cols):
    noteDict = {}
    #NEED TO BUILD VIA FOR LOOP, CREATE SO MUSIC ONLY PLAYS ON AREA IN ZOOMED CAMERA POSITION
    notes = [127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21,127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21, 127, 124, 122, 120, 117, 115, 112, 110, 108, 105, 103, 100, 98, 96, 93, 91, 88, 86, 84, 81, 79, 76, 74, 72, 69, 67, 64, 62, 60, 57, 55, 52, 50, 48, 45, 43, 40, 38, 36, 33, 31, 28, 26, 24, 21]
    for row in range(cols):
        noteDict[row] = notes[row]
    return noteDict

noteDict = noteList(cols)

def midi2(note, volume=100, length=.5):
    midi_out.note_on(note, volume) # 74 is middle C, 127 is "how loud" - max is 127
    # sleep(length)
    # for n in notes: 
    #     midi_out.note_off(n, volume)

def notesCell(row, col, cellSize, noteDict = noteDict):
    note = noteDict[col]
    # screen.blit(musicSurf2,(col*cellSize,row*cellSize))
    pg.draw.rect(screen, (0,230,0), (col*cellSize,row*cellSize,cellSize-1,cellSize-1))
    midi2(note)
    return

# def colLight(col):
#     playCol = matrix[:, col-1]
#     for cell in col:
#         if cell == 1:
#             screen.blit(activeSurf,(col*cellSize,row*cellSize))





colNum = 0
#====================================================

muted = False
sequencer = False
drawColor = cellColor
renderGrid(rows, cols)
paused = True

while 1:
    newCells = createMatrix(rows,cols)
    midi_out.set_instrument(instrument)
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
                        paused = True
                case pg.K_1:
                    gameRate = 3
                case pg.K_2:
                    gameRate = 10
                case pg.K_3:
                    gameRate = 15
                case pg.K_4:
                    gameRate = 25
                case pg.K_5:
                    gameRate = 75
                case pg.K_6:
                    gameRate = 300
                case pg.K_c:
                    myColor = colorchooser.askcolor()
                    cellColor = myColor[0]
                case pg.K_m:
                    if muted:
                        muted = False
                    else:
                        muted = True
                case pg.K_r:
                    randomNoise(matrix,rows,cols)
                case pg.K_s:
                    if sequencer:
                        sequencer = False
                    else:
                        sequencer = True
                case pg.K_COMMA:
                    if instrument == 0:
                        instrument = 128
                    instrument -= 1
                case pg.K_PERIOD:
                    if instrument == 127:
                        instrument = -1
                    instrument += 1
                case pg.K_MINUS:
                    if cellSize > minCellSize:
                        cellSize -= 1
                        if cellSize < 10:
                            gridColor[0] = gridColor[0] * .9
                            gridColor[1] = gridColor[1] * .9
                            gridColor[2] = gridColor[2] * .9
                        print(gridColor)
                        renderGrid(rows, cols)
                case pg.K_EQUALS:
                    if cellSize < maxCellSize:
                        cellSize += 1
                        if cellSize < 10:
                            gridColor[0] = gridColor[0] * 1.1
                            gridColor[1] = gridColor[1] * 1.1
                            gridColor[2] = gridColor[2] * 1.1
                            print(gridColor)
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

  
    renderCells(matrix, muted, newCells)
       

    click = pg.mouse.get_pressed()
    mouseX, mouseY = pg.mouse.get_pos()

    #draw
    #do not add cells that exceed board size
    #to speed up: turn into function, add blit
    if click[0]:
        paused = True
        rate = drawRate
        roundedX = mouseX // cellSize
        roundedY = mouseY // cellSize
        matrix[roundedY,roundedX] = True
    elif click[2]:
        paused = True
        rate = drawRate
        roundedX = mouseX // cellSize
        roundedY = mouseY // cellSize
        matrix[roundedY,roundedX] = False
    else:
        rate = gameRate

    #music
    #NEED TO PROGRAM SEQUENCER TO ONLY LOOP OVER VISIBLE COLUMNS
    if sequencer == True and paused == False:
        playColumn(matrix, colNum, noteDict)
        colNum += 1
        if colNum == cols:
            colNum = 0

    screen.blit(displayFPS(), (10,5))
    population = str(np.count_nonzero(matrix))
    screen.blit(font.render("Population: " + population, 1, (0,255,0)), (400,5))
    screen.blit(font.render("Instrument " + str(instrument) + ": "  + str(midiBank.get(instrument)), 1, (0,255,0)), (sWidth - 500,5))
    screen.blit(font.render("Sequencer: " + str(sequencer), 1, (0,255,0)), (sWidth-800,5))
    pg.display.update()
    clock.tick(rate)
