# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread("cell.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversion de l'image en niveaux de gris

L = [] # liste qui contiendra les coordonnées des minima locaux
Marque = np.zeros(img.shape) # matrice qui indique si les pixels sont traités


for i in range(1, np.size(img, 0) - 1):
    for j in range(1, np.size(img, 1) - 1):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img[i-1:i+1, j-1:j+1])
        a = int(min_loc[0] + i - 1)
        b = int(min_loc[1] + j - 1)
        L.append((a,b))
        Marque[a,b] = 1


fifos = [[] for i in range(256)] # liste des piles correspondant aux intensités des pixels


# dans la boucle suivante, on considère les 4 plus proches voisins du minimum local considéré
for marqueur in L:
    x, y = marqueur[0], marqueur[1]
    if ((x-1,y) not in L):
        fifos[img[x-1,y]].append((x-1,y))
    if ((x,y-1) not in L):
        fifos[img[x,y-1]].append((x,y-1))
    if ((x,y+1) not in L):
        fifos[img[x,y+1]].append((x,y+1))
    if ((x+1,y) not in L):
        fifos[img[x+1,y]].append((x+1,y))
        

while (Marque != np.ones(img.shape)):
    x, y = -1, -1 # coordonnées du premier point d'intensité minimale dans les piles
    n = 0
    
    while (x == - 1):
        if (not fifos[n] == False):
            z = fifos[n].popleft()
            x, y = z[0], z[1]
        n += 1
        
    Marque[x,y] = 1        
    if (Marque[x-1,y] + Marque[x,y-1] + Marque[x,y+1] + Marque[x+1,y] < 4):
        if ((x-1,y) not in fifos[img[x-1,y]] and Marque[x-1,y] == 0):
            fifos[img[x-1,y]].append((x-1,y))
        if ((x,y-1) not in fifos[img[x,y-1]] and Marque[x,y-1] == 0):
            fifos[img[x,y-1]].append((x,y-1))
        if ((x,y+1) not in fifos[img[x,y+1]] and Marque[x,y+1] == 0):
            fifos[img[x,y+1]].append((x,y+1))
        if ((x+1,y) not in fifos[img[x+1,y]] and Marque[x+1,y] == 0):
            fifos[img[x+1,y]].append((x+1,y))
            


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()