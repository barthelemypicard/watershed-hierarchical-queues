# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque # deque permet d'avoir une structure de piles
from scipy.signal import argrelextrema
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


# Permet d'obtenir les voisins de (x,y) en 4 ou 8 connexités
def voisins(x,y,nb_connex):
    voisin = np.zeros((nb_connex, 2))
    voisin[0] = [ x-1, y   ]
    voisin[1] = [ x  , y-1 ]
    voisin[2] = [ x+1, y   ]
    voisin[3] = [ x  , y+1 ]
    if nb_connex == 8:
        voisin[4] = [ x-1, y-1 ]
        voisin[5] = [ x+1, y-1 ]
        voisin[6] = [ x-1, y+1 ]
        voisin[7] = [ x+1, y+1 ]
        
    return voisin


def hqWatershed(img, nb_connex = 4, mask = None):

    L = [] # liste qui contiendra les coordonnées des minima locaux
    Marque = np.zeros(img.shape) # matrice qui indique si les pixels sont traités
    labels = np.zeros(img.shape) # matrice contenant les labels, les pixels ayant un label valant 0 à la fin du programme sont des points frontière

    k = 50 # taille du carré considéré pour calculer les minima locaux
    p = 1 # compteur qui sert d'indicateur de label
    
    if mask == None:
        for i in range(0, np.size(img, 0) - k, k):
            for j in range(0, np.size(img, 1) - k, k):
                # minMaxLoc retourne les valeurs et les positions des min et max de l'image considérée
                # k permet de choisir la taille de la sous-image
                # pour un carré de taille 3 (k=2), beaucoup trop de minima locaux
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img[i:i+k, j:j+k])
                if (min_val != max_val):
                    a = int(min_loc[0] + i)
                    b = int(min_loc[1] + j)
                    L.append((a,b))
                    Marque[a,b] = 1
                    labels[a,b] = p
                    p += 1
    else:
        (x_mask, y_mask) = mask
        len = x_mask.shape
        for i in range(len[0]):
            L.append((x_mask[i],y_mask[i]))
            Marque[x_mask[i],y_mask[i]] = 1
            labels[x_mask[i],y_mask[i]] = p
            p += 1
        

    #cv2.namedWindow('labels', cv2.WINDOW_NORMAL)
    #cv2.imshow('labels', labels)
    #cv2.waitKey(0)
    
    #L.append((60,80)); Marque[60,80] = 1; labels[60,80] = 1
    #L.append((140,100)); Marque[140,100] = 1; labels[140,100] = 2


    fifos = [deque() for i in range(256)] # liste des piles correspondant aux intensités des pixels


    # dans la boucle suivante, on considère les 4 plus proches voisins du minimum local considéré
    for marqueur in L:
        x, y = marqueur[0], marqueur[1]
        all_voisins = voisins(x,y,nb_connex)
        for vois in all_voisins:
            X = vois[0]
            Y = vois[1]
            if ((X,Y) not in L) and X >= 0 and Y >= 0 and X < np.size(img,0) and Y < np.size(img,1):
                if ((X,Y) not in fifos[img[X,Y]]):
                    fifos[img[X,Y]].append((X,Y))
        #if ((x-1,y) not in L):
        #    fifos[img[x-1,y]].append((x-1,y))
        #if ((x,y-1) not in L):
        #    fifos[img[x,y-1]].append((x,y-1))
        #if (((x,y+1) not in L) and (y+1 < np.size(img,1))):
        #    fifos[img[x,y+1]].append((x,y+1))
        #if (((x+1,y) not in L) and (x+1 < np.size(img,0))):
        #    fifos[img[x+1,y]].append((x+1,y))
        

    while (np.array_equal(Marque, np.ones(img.shape)) == False):
        x, y = -1, -1 # coordonnées du premier point d'intensité minimale dans les piles
        n = 0
    
        while (x == - 1):
            if ((not fifos[n]) == False): # not L est égal à True lorsque L est vide
                z = fifos[n].popleft() # popleft() retourne et enlève le premier élément de la pile
                if not Marque[z[0], z[1]]:
                    x, y = z[0], z[1]
            n += 1
            if (n > 256):
                print "All queues empty"
                break
        
        
########################################################################################
        Marque[x,y] = 1
        
        # On cherche à savoir si le pixel courant est un point frontière
        # C'est le cas si il est entouré par deux pixels marqués par des
        # des labels différents
        labels[x,y] = -1
        all_voisins = voisins(x,y,nb_connex)
        for vois in all_voisins:
            xv = vois[0]
            yv = vois[1]
            if xv >= 0 and yv >= 0 and xv < np.size(img,0) and yv < np.size(img,1):
                if Marque[xv,yv] and labels[x,y] != 0:
                    if labels[xv,yv] == 0 and labels[x,y] == -2:
                        labels[x,y] = -1
                    elif labels[xv, yv] > 0 and labels[x,y] <= -1:
                        labels[x,y] = labels[xv,yv]
                    elif labels[xv, yv] > 0 and labels[xv,yv] > 0 and labels[x,y] != labels[xv,yv]:
                        labels[x,y] = 0
                if not Marque[xv,yv] and ((xv,yv) not in fifos[img[xv,yv]]):
                    fifos[img[xv,yv]].append((xv,yv))
        if labels[x,y] == -1:
            labels[x,y] = 1000
        #if labels[x,y] >= 0:
        #    Marque[x,y] = 1
        #    for vois in all_voisins:
        #        if xv >= 0 and yv >= 0 and xv < np.size(img,0) and yv < np.size(img,1):
        #            if not Marque[xv,yv] and ((xv,yv) not in fifos[img[xv,yv]]):
        #                fifos[img[xv,yv]].append((xv,yv))
        #else:
        #    if ((x,y) not in fifos[img[xv,yv]]):
        #        fifos[img[x,y]].append((x,y))
    
    #    if (((x+1) < np.size(img,0)) and ((y+1) < np.size(img,1))):
    #        if (Marque[x-1,y] + Marque[x,y-1] + Marque[x,y+1] + Marque[x+1,y] < 4):
    #            if (((x-1,y) not in fifos[img[x-1,y]]) and (Marque[x-1,y] == 0)):
    #                fifos[img[x-1,y]].append((x-1,y))
    #            if (((x,y-1) not in fifos[img[x,y-1]]) and (Marque[x,y-1] == 0)):
    #                fifos[img[x,y-1]].append((x,y-1))
    #            if (((x,y+1) not in fifos[img[x,y+1]]) and (Marque[x,y+1] == 0)):
    #                fifos[img[x,y+1]].append((x,y+1))
    #            if (((x+1,y) not in fifos[img[x+1,y]]) and (Marque[x+1,y] == 0)):
    #                fifos[img[x+1,y]].append((x+1,y))
    #                
    #    elif ((x+1) < np.size(img,0)):
    #        if (Marque[x-1,y] + Marque[x,y-1] +Marque[x+1,y] < 3):
    #            if (((x-1,y) not in fifos[img[x-1,y]]) and (Marque[x-1,y] == 0)):
    #                fifos[img[x-1,y]].append((x-1,y))
    #            if (((x,y-1) not in fifos[img[x,y-1]]) and (Marque[x,y-1] == 0)):
    #                fifos[img[x,y-1]].append((x,y-1))
    #            if (((x+1,y) not in fifos[img[x+1,y]]) and (Marque[x+1,y] == 0)):
    #                fifos[img[x+1,y]].append((x+1,y))
    #                
    #    elif ((y+1) < np.size(img,1)):
    #        if (Marque[x-1,y] + Marque[x,y-1] +Marque[x,y+1] < 3):
    #            if (((x-1,y) not in fifos[img[x-1,y]]) and (Marque[x-1,y] == 0)):
    #                fifos[img[x-1,y]].append((x-1,y))
    #            if (((x,y-1) not in fifos[img[x,y-1]]) and (Marque[x,y-1] == 0)):
    #                fifos[img[x,y-1]].append((x,y-1))
    #            if (((x,y+1) not in fifos[img[x,y+1]]) and (Marque[x,y+1] == 0)):
    #                fifos[img[x,y+1]].append((x,y+1))
    #                
    #    else:
    #        if 
            
    ########################################################################################
            


    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return labels

if __name__ == "__main__":
    nb_connex = 4
    img = cv2.imread("coins.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversion de l'image en niveaux de gris
    cv2.namedWindow('Base image', cv2.WINDOW_NORMAL)
    cv2.imshow('Base image',img)
    cv2.waitKey(1)
    
    radius = 15
    kernel = np.ones((radius,radius),np.uint8)
    img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.namedWindow('Open image', cv2.WINDOW_NORMAL)
    cv2.imshow('Open image',img_open)
    cv2.waitKey(1)
    
    kernel = np.ones((2,2),np.uint8)
    img_grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    Mint_arg = argrelextrema(img_open, np.less)
    Mint = np.zeros(img.shape)
    Mint[Mint_arg] = 1
    
    Mext = hqWatershed(255-img_open, nb_connex);
    Mext = (Mext == 0)
    cv2.namedWindow('Mext image', cv2.WINDOW_NORMAL)
    cv2.imshow('Mext image', Mext.astype(np.int64))
    cv2.waitKey(1)
    
    msk = (Mint + Mext)/2;
    msk = (msk > 0);
    cv2.namedWindow('mask image', cv2.WINDOW_NORMAL)
    cv2.imshow('mask image', msk.astype(np.int64))
    cv2.waitKey(1)
    
    #Impose min
    grad_modif = img_grad - msk
    result = hqWatershed(grad_modif)
    result = 255*(result - result.min()) / (result.max() - result.min())
    cv2.namedWindow('result image', cv2.WINDOW_NORMAL)
    cv2.imshow('result image', result)
    cv2.waitKey(1)
    
   # cv2.destroyAllWindows()