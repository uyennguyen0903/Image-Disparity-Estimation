#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Petit code d'évaluation des disparités, avec gestion des occultations.
MàJ 20/03/2022
"""

import sys
import numpy as np
#import matplotlib.pyplot as plt
from skimage import io

def evaldisp(GT, occl, disp):
    # mise à l'échelle pixellique normale des disparités et de la vérité
    GT = GT / 4.
    disp = disp / 4.
    # occultations passées en type binaire (1 = visible, 0 = occultation)
    occl = occl > 0
    #plt.figure()
    #plt.imshow(occl, cmap='gray')
    
    # disparités estimées invalides quand disp fixée à 0 par l'algo de correspondance (1 = valide, 0 = invalide)
    inval = disp > 0
    #plt.figure()
    #plt.imshow(inval, cmap='gray')
    
    # masque pour évaluer les erreurs en ignorant les occultations et estimations invalides
    Meval = occl & inval
    #plt.figure()
    #plt.imshow(Meval, cmap='gray')

    # mesures d'erreur des disparités estimées

    nbpts_Meval = np.count_nonzero(Meval)
    abserrmap = np.absolute(GT-disp) * Meval
    #plt.figure()
    #plt.imshow(abserrmap, cmap='jet')

    # mesure de l'erreur moyenne des disparités
    errmoy = np.sum(abserrmap) / nbpts_Meval

    # mesure du pourcentage de points de disparité > seuil ; les erreurs d'occultations sont comptées ici
    nbpts_total = GT.shape[0]*GT.shape[1]
    s1 = 1
    s2 = 2

    occevalmap = (occl != inval)
    
    print('erreur occulations: ', np.sum(occevalmap) * 100. / nbpts_total)
    
    #plt.figure()
    #plt.imshow(occevalmap, cmap='gray')
    s1map = (abserrmap > s1) | (occevalmap)
    #plt.figure()
    #plt.imshow(s1map, cmap='gray')
    s2map = (abserrmap > s2) | (occevalmap)
    #plt.figure()
    #plt.imshow(s2map, cmap='gray')

    ps1 = np.count_nonzero(s1map) / nbpts_total
    ps2 = np.count_nonzero(s2map) / nbpts_total
    
    return errmoy, ps1, ps2
    

def main(fgt, foccl, fdisp):
    GT = io.imread(fgt)
    occl = io.imread(foccl) # occultations point de vue image gauche
    disp = io.imread(fdisp)

    assert GT.shape == occl.shape == disp.shape, "Les images de cartes de disparités et d'occultations d'entrée n'ont pas les mêmes dimensions !"


    errmoy, ps1, ps2 = evaldisp(GT, occl, disp)
    
    print("Erreur moyenne des disparités estimées (hors estimations invalides et occultations) = {:.2f} px".format(errmoy))
    print("Pourcentage de disparités dont l'erreur est > à 1 px (y compris erreurs d'occultations) = {:.2f} %".format(ps1*100.))
    print("Pourcentage de disparités dont l'erreur est > à 2 px (y compris erreurs d'occultations) = {:.2f} %".format(ps2*100.))


if __name__ == '__main__':
    # arguments
    assert len(sys.argv) == 4, "Il faut 3 arguments : evaldisp.py disparites_verite.png occultations_verite.png disparites_estimees.png"
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
