import numpy as np
from skimage import io
import cv2 as cv
from skimage import util
from skimage import color
import sys
from tqdm import tqdm
import time
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import as_strided


class BlockMatching:
    """Block Matching module.

    Attributes
    ----------
    max_disp (int) : Valeur de disparité maximum.
    window (int) : Taille du fenêtre.
    filtre_taille (int) : Taille du filtre médian.
    left_I (np.array) : Image gauche.
    right_I (np.array) : Image right.
    height (int) : Longueur d'image.
    width (int) : Largeur d'image.
    occlusion_seuil (int) : seuilf de région occultée.
    disparities (np.array) : Carte de disparité finale.


    """

    def __init__(
        self,
        path_left: str,
        path_right: str,
        max_disp: int,
        window: int,
        filtre_taille: int,
        occlusion_seuil: int,
    ):
        self.max_disp = max_disp
        self.window = window
        self.filtre_taille = filtre_taille
        self.occlusion_seuil = occlusion_seuil
        self.left_I, self.right_I = self.image_loading(path_left, path_right)
        self.height, self.width = self.left_I.shape
        self.disparities = np.zeros(self.left_I.shape)

    def image_loading(self, path_left: str, path_right: str):
        """
        Charger les images gauche et droit.
        """
        imgL = util.img_as_ubyte(color.rgb2gray(io.imread(path_left))).astype("int")
        imgR = util.img_as_ubyte(color.rgb2gray(io.imread(path_right))).astype("int")
        return (imgL, imgR)

    def calcul_disp(self, y: int, x: int, max_disp: int) -> int:
        """
        Recherche la disparité avec le cost minimum pour le pixel(x, y).
        """
        half = int(self.window / 2)
        y_start, x_start = y - half, x - half
        # Patch pour lequel on cherche la disparité
        current_patch = self.left_I[
            y_start : y_start + self.window, x_start : x_start + self.window
        ]
        # Patch contenant toute la ligne en partant du patch actuel jusqu'à un décalage
        # maximum correspondant à la disparité maximale
        patch = self.right_I[
            y_start : y_start + self.window, x_start - max_disp : x_start + self.window
        ]
        shape = tuple(np.subtract(patch.shape, self.window - 1)) + (
            self.window,
            self.window,
        )
        # Array contenant tous les patchs à comparer au patch actuel, dans
        # l'ordre de leurs index de gauche à droite, donc de la disparité
        # maximale à minimale
        windowed_patch = as_strided(patch, shape=shape, strides=patch.strides * 2)
        # Calcul du coût pour tous les sous patchs avec la somme des carrés des différences
        cost = (
            (windowed_patch - current_patch) * (windowed_patch - current_patch)
        ).sum(axis=(-1, -2))
        # On choisit la différence la plus faible
        # print(np.argmin(cost, axis = 1))
        t, index = np.unravel_index(np.argmin(cost), cost.shape)
        # On retourne la disparité
        return max_disp - index

    def get_disparity_map(self):
        """
        Contruire la carte de disparité avec Block Matching.
        """
        half = int(self.window / 2)
        for y in tqdm(range(half, self.height - half)):
            for x in range(half, self.width - half):
                # Si le x est inférieur à la disparité maximale
                if x <= self.max_disp + half:
                    # On ne pourra pas aller 64 px à gauche donc on réduit la taille
                    current_max_d = x - half
                else:
                    # Sinon on va jusqu'à la disparité maximale
                    current_max_d = self.max_disp - 1
                # Calcul de la disparité avec la fonction précédente
                self.disparities[y, x] = self.calcul_disp(y, x, current_max_d)

    def process_stereo_matching(self):
        """
        Processus final de Block Matching avec les opérations d'optimisation.
        """
        # Calculer disparity map.
        self.get_disparity_map()

        # Normaliser disparity map.
        self.disparities = np.uint8(
            cv.normalize(self.disparities, 0, 255, norm_type=cv.NORM_MINMAX)
        )

        # Traiter les bords.
        half = int(self.window / 2)
        for i in range(half):
            self.disparities[i, :] = self.disparities[half, :]
            self.disparities[self.height - i - 1, :] = self.disparities[
                self.height - half - 1, :
            ]
            self.disparities[:, self.width - i - 1] = self.disparities[
                :, self.height - half - 1
            ]

        # Appliquer seuilage.
        self.disparities[self.disparities <= self.occlusion_seuil] = 0

        # Nettoyer output map avec médian filtre.
        self.disparities = median_filter(
            self.disparities, (self.filtre_taille, self.filtre_taille)
        )


if __name__ == "__main__":
    assert len(sys.argv) == 4, "invalids arguments"
    block_matching = BlockMatching(
        path_left=sys.argv[1],
        path_right=sys.argv[2],
        max_disp=64,
        window=7,
        filtre_taille=17,
        occlusion_seuil=59,
    )

    block_matching.process_stereo_matching()

    io.imsave(sys.argv[3], block_matching.disparities)
