# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque # deque permet d'avoir une structure de piles

img = cv2.imread("cell.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversion de l'image en niveaux de gris

L = [] # liste qui contiendra les coordonnées des minima locaux
Marque = np.zeros(img.shape) # matrice qui indique si les pixels sont traités
labels = np.zeros(img.shape) # matrice contenant les labels, les pixels ayant un label valant 0 à la fin du programme sont des points frontière

k = 10 # taille du carré considéré pour calculer les minima locaux
p = 1 # compteur qui sert d'indicateur de label

for i in range(0, np.size(img, 0) - k, k):
    for j in range(0, np.size(img, 1) - k, k):
        # minMaxLoc retourne les valeurs et les positions des min et max de l'image considérée
        # k permet de choisir la taille de la sous-image
        # pour un carré de taille 3 (k=2), beaucoup trop de minima locaux
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img[i:i+k, j:j+k])
        a = int(min_loc[0] + i)
        b = int(min_loc[1] + j)
        L.append((a,b))
        Marque[a,b] = 1
        labels[a,b] = p
        p += 1


fifos = [deque() for i in range(256)] # liste des piles correspondant aux intensités des pixels


# dans la boucle suivante, on considère les 4 plus proches voisins du minimum local considéré
for marqueur in L:
    x, y = marqueur[0], marqueur[1]
    if ((x-1,y) not in L):
        fifos[img[x-1,y]].append((x-1,y))
    if ((x,y-1) not in L):
        fifos[img[x,y-1]].append((x,y-1))
    if (((x,y+1) not in L) and (y+1 < np.size(img,1))):
        fifos[img[x,y+1]].append((x,y+1))
    if (((x+1,y) not in L) and (x+1 < np.size(img,0))):
        fifos[img[x+1,y]].append((x+1,y))
        

while (np.array_equal(Marque, np.ones(img.shape)) == False):
    x, y = -1, -1 # coordonnées du premier point d'intensité minimale dans les piles
    n = 0
    
    while (x == - 1):
        if ((not fifos[n]) == False): # not L est égal à True lorsque L est vide
            z = fifos[n].popleft() # popleft() retourne et enlève le premier élément de la pile
            x, y = z[0], z[1]
        n += 1
        
        
########################################################################################
    Marque[x,y] = 1
    if (((x+1) < np.size(img,0)) and ((y+1) < np.size(img,1))):
        if (Marque[x-1,y] + Marque[x,y-1] + Marque[x,y+1] + Marque[x+1,y] < 4):
            if (((x-1,y) not in fifos[img[x-1,y]]) and (Marque[x-1,y] == 0)):
                fifos[img[x-1,y]].append((x-1,y))
            if (((x,y-1) not in fifos[img[x,y-1]]) and (Marque[x,y-1] == 0)):
                fifos[img[x,y-1]].append((x,y-1))
            if (((x,y+1) not in fifos[img[x,y+1]]) and (Marque[x,y+1] == 0)):
                fifos[img[x,y+1]].append((x,y+1))
            if (((x+1,y) not in fifos[img[x+1,y]]) and (Marque[x+1,y] == 0)):
                fifos[img[x+1,y]].append((x+1,y))
                
    elif ((x+1) < np.size(img,0)):
        if (Marque[x-1,y] + Marque[x,y-1] +Marque[x+1,y] < 3):
            if (((x-1,y) not in fifos[img[x-1,y]]) and (Marque[x-1,y] == 0)):
                fifos[img[x-1,y]].append((x-1,y))
            if (((x,y-1) not in fifos[img[x,y-1]]) and (Marque[x,y-1] == 0)):
                fifos[img[x,y-1]].append((x,y-1))
            if (((x+1,y) not in fifos[img[x+1,y]]) and (Marque[x+1,y] == 0)):
                fifos[img[x+1,y]].append((x+1,y))
                
    elif ((y+1) < np.size(img,1)):
        if (Marque[x-1,y] + Marque[x,y-1] +Marque[x,y+1] < 3):
            if (((x-1,y) not in fifos[img[x-1,y]]) and (Marque[x-1,y] == 0)):
                fifos[img[x-1,y]].append((x-1,y))
            if (((x,y-1) not in fifos[img[x,y-1]]) and (Marque[x,y-1] == 0)):
                fifos[img[x,y-1]].append((x,y-1))
            if (((x,y+1) not in fifos[img[x,y+1]]) and (Marque[x,y+1] == 0)):
                fifos[img[x,y+1]].append((x,y+1))
                
    else:
        if 
        
########################################################################################
            


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()