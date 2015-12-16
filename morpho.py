# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque # deque permet d'avoir une structure de piles
from scipy.signal import argrelextrema
import scipy.misc
import pylab as pl


#Ajoute du bruit à une image
def addBlur(im, std = -1):
    im_max = np.max(im)
    im_min = np.min(im)
    h, w = im.shape
    if std == -1:
        std = (im_max-im_min) / 4
    result = im + std*(np.random.rand(h, w))
    return result
    


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

#Calcule le watershed en utilisant des files d'attentes hierarchiques
#Entrées :
# - img : image à traiter
# - nb_connex : nombre de connexités (4 ou 8)
# - mask : masque des marqueurs (si rien n'est donné, on calcule les maxima locaux)
# Sortie :
# - labels : image labelisée (0 pour les fronitères)
def hqWatershed(img, nb_connex = 8, mask = None):

    L = [] # liste qui contiendra les coordonnées des minima locaux
    tmp_L = deque() #Liste d'initialisation des marqueurs
    Marque = np.zeros(img.shape) # matrice qui indique si les pixels sont traités
    labels = np.zeros(img.shape) # matrice contenant les labels, les pixels ayant un label valant 0 à la fin du programme sont des points frontière

    k = 50 # taille du carré considéré pour calculer les minima locaux
    p = 1 # compteur qui sert d'indicateur de label
    
    # Initialisation des marqueurs
    if mask == None:
        #Si aps de masque en entrée, on calcule les minima locaux
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
        print "Coucou"
        mask_max = mask.max()
        for i in range(np.size(img, 0)):
            for j in range(np.size(img, 1)):
                current_label = -1
                if mask[i,j] == mask_max and Marque[i,j] == 0:
                    current_label = p
                    p += 1
                    L.append((i,j))
                    Marque[i,j] = 1
                    labels[i,j] = current_label
                    all_voisins = voisins(i,j,8)
                    for vois in all_voisins:
                        X = vois[0]
                        Y = vois[1]
                        # On parcourt les voisins de même label de proche en proche
                        if tmp_L.count((X,Y)) == 0 and X >= 0 and Y >= 0 and X < np.size(img,0) and Y < np.size(img,1):
                            if Marque[X,Y] == 0 and mask[X,Y] == mask_max:
                                tmp_L.append((X,Y))
                # Tant que toute la zone connexe n'est pas labelisée, on continue
                while tmp_L:
                    x, y = tmp_L.popleft()
                    L.append((x,y))
                    Marque[x,y] = 1
                    labels[x,y] = current_label
                    all_voisins = voisins(x,y,nb_connex)
                    for vois in all_voisins:
                        X = vois[0]
                        Y = vois[1]
                        if tmp_L.count((X,Y)) == 0 and X >= 0 and Y >= 0 and X < np.size(img,0) and Y < np.size(img,1):
                            if Marque[X,Y] == 0 and mask[X,Y] == mask_max:
                                tmp_L.append((X,Y))
    print "Markers initialized"
    
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
        
    h, w = Marque.shape
    n_px = h*w
    while (np.array_equal(Marque, np.ones(img.shape)) == False):
        # print "%d%% achieved." % (np.sum(Marque)*100/n_px)
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
            # Cas particulier où le point est indéterminé. On lui donne alors 
            # un label arbitraire
            labels[x,y] = 1000

    return labels



if __name__ == "__main__":
    nb_connex = 8
    ext_mask = None
    img = cv2.imread("coins.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversion de l'image en niveaux de gris
    cv2.namedWindow('Base image', cv2.WINDOW_NORMAL)
    cv2.imshow('Base image',img)
    cv2.waitKey(1)
    
    radius = 20
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
    
    ext_mask = cv2.imread("coins_mask.png")
    ext_mask = cv2.cvtColor(ext_mask, cv2.COLOR_BGR2GRAY)
    ext_mask = scipy.misc.imresize(ext_mask, img.shape)
    cv2.namedWindow('Mask image', cv2.WINDOW_NORMAL)
    cv2.imshow('Mask image',ext_mask)
    cv2.waitKey(1)
    
    if ext_mask == None:
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
    else:
        msk = ext_mask
        
    #Impose min
    # grad_modif = img_grad - msk
    result = hqWatershed(img_grad, mask = msk)
    result = 255*((result>0) - result.min()) / (result.max() - result.min())
    cv2.namedWindow('result image', cv2.WINDOW_NORMAL)
    cv2.imshow('result image', result)
    
    print "Done."
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()