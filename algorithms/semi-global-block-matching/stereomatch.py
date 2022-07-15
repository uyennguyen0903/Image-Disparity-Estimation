import sys
import time
import numpy as np
from skimage import io
from skimage import color
from skimage import util
import cv2
from tqdm import tqdm


class SemiGlobalMatching:
    """Semi-Global Block Matching module.

    Attributes
    ----------
    max_disp (int) : Valeur de disparité maximum.
    census_size (int) : Taille du fenêtre.
    filtre_taille (int) : Taille du filtre médian.
    left_I (np.array) : Image gauche.
    right_I (np.array) : Image right.
    height (int) : Longueur d'image.
    width (int) : Largeur d'image.
    occlusion_seuil (int) : seuilf de région occultée.
    penalty_1 (int) : penalité pour les disparité = 1 px.
    penalty_2 (int) : penalité pour les disparité > 1 px.
    left_cost_volume (np.array): np.array 3d (longueur x largeur x max_disparité) contenant les distances de Hamming pour tous les niveaux de disparité.
    aggregation_volum (np.array): np.array 3d (longueur x largeur x max_disparité) contenant les couts totales pour toutes les directions.
    disparities (np.array) : Carte de disparité finale.


    """

    def __init__(
        self,
        path_left: str,
        path_right: str,
        max_disp: int,
        census_size: int,
        filter_size: int,
        occlusion_seuil: int,
        penalty_1: int,
        penalty_2: int,
    ):
        self.max_disp = max_disp
        self.census_size = census_size
        self.filter_size = filter_size
        self.occlusion_seuil = occlusion_seuil
        self.penalty_1 = penalty_1
        self.penalty_2 = penalty_2
        self.left_I, self.right_I = self.image_loading(path_left, path_right)
        self.height, self.width = self.left_I.shape
        self.disparities = np.zeros(self.left_I.shape)
        self.left_cost_volume = np.zeros((self.height, self.width, self.max_disp))
        self.aggregation_volume = np.zeros((self.height, self.width, self.max_disp, 4))

    def image_loading(self, path_left: str, path_right: str):
        """
        Charger les images gauche et droit.
        """
        imgL = util.img_as_ubyte(color.rgb2gray(io.imread(path_left))).astype("int")
        imgR = util.img_as_ubyte(color.rgb2gray(io.imread(path_right))).astype("int")
        return (imgL, imgR)

    def compute_costs(self):
        """
        Compute les distances de Hamming pour tous les niveaux de disparité.
        """
        left_census = np.zeros(self.left_I.shape)
        right_census = np.zeros(self.left_I.shape)

        half = int(self.census_size / 2)
        center_index = half * self.census_size + half

        # Etape 1: Center-Symmetric Census Transform (CSCT).
        for y in tqdm(range(half, self.height - half)):
            for x in range(half, self.width - half):
                curr_block = self.left_I[
                    (y - half) : (y + half + 1), (x - half) : (x + half + 1)
                ]
                bits = np.full(
                    shape=(self.census_size, self.census_size),
                    fill_value="0",
                    dtype=str,
                )
                bits[curr_block < self.left_I[y, x]] = "1"
                bits = "".join(bits.flatten())
                bits = bits[0:center_index:] + bits[center_index + 1 : :]
                left_census[y, x] = int(bits, 2)

                curr_block = self.right_I[
                    (y - half) : (y + half + 1), (x - half) : (x + half + 1)
                ]
                bits = np.full(
                    shape=(self.census_size, self.census_size),
                    fill_value="0",
                    dtype=str,
                )
                bits[curr_block < self.right_I[y, x]] = "1"
                bits = "".join(bits.flatten())
                bits = bits[0:center_index:] + bits[center_index + 1 : :]
                right_census[y, x] = int(bits, 2)

        rcensus = np.zeros(shape=self.left_I.shape, dtype=np.int64)

        # Etape 2: Calculer les distances de Hamming.
        for d in tqdm(range(0, self.max_disp)):
            rcensus[:, (half + d) : (self.width - half)] = right_census[
                :, half : (self.width - d - half)
            ]
            left_xor = np.int64(np.bitwise_xor(np.int64(left_census), rcensus))
            left_distance = np.zeros(shape=self.left_I.shape, dtype=np.uint32)
            while not np.all(left_xor == 0):
                tmp = left_xor - 1
                mask = left_xor != 0
                left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
                left_distance[mask] += 1
            self.left_cost_volume[:, :, d] = left_distance

    def get_path_cost(self, block):
        """
        Compute les couts de mise en correspondance pour une direction spécifique.
        """
        block_dim = block.shape[0]

        disparities = [d for d in range(self.max_disp)] * self.max_disp
        disparities = np.array(disparities).reshape(self.max_disp, self.max_disp)

        penalties = np.zeros(shape=(self.max_disp, self.max_disp))
        penalties[np.abs(disparities - disparities.T) == 1] = self.penalty_1
        penalties[np.abs(disparities - disparities.T) > 1] = self.penalty_2

        minimum_cost_path = np.zeros(shape=(block_dim, self.max_disp))
        minimum_cost_path[0, :] = block[0, :]

        for i in range(1, block_dim):
            costs = np.repeat(
                minimum_cost_path[i - 1, :], repeats=self.max_disp, axis=0
            ).reshape(self.max_disp, self.max_disp)
            costs = np.amin(costs + penalties, axis=0)
            minimum_cost_path[i, :] = (
                block[i, :] + costs - np.amin(minimum_cost_path[i - 1, :])
            )

        return minimum_cost_path

    def aggregate_costs(self):
        # Direction verticales.
        for x in tqdm(range(0, self.width)):
            # North
            self.aggregation_volume[:, x, :, 0] = self.get_path_cost(
                self.left_cost_volume[:, x, :]
            )
            # South
            self.aggregation_volume[:, x, :, 1] = np.flip(
                np.flip(self.left_cost_volume[:, x, :], axis=0), axis=0
            )

        # Directions horizontales.
        for y in tqdm(range(0, self.height)):
            # West
            self.aggregation_volume[y, :, :, 2] = self.get_path_cost(
                self.left_cost_volume[y, :, :]
            )
            # East
            self.aggregation_volume[y, :, :, 3] = np.flip(
                np.flip(self.left_cost_volume[y, :, :], axis=0), axis=0
            )

    def SGM_process(self):
        """
        Processus final de SGBM avec les opérations d'optimisation.
        """
        # Compute cost.
        print("Compute cost ...")
        self.compute_costs()

        # Aggregate cost volume.
        print("Aggregate cost ...")
        self.aggregate_costs()

        # Select optimal result.
        self.aggregation_volume = np.sum(self.aggregation_volume, axis=3)
        self.aggregation_volume = np.argmin(self.aggregation_volume, axis=2)

        # Normalize result.
        self.disparities = np.uint8(255.0 * self.aggregation_volume / self.max_disp)

        # Occlusion thresholding.
        self.disparities[self.disparities <= self.occlusion_seuil] = 0

        self.disparities = cv2.medianBlur(self.disparities, self.filter_size)


if __name__ == "__main__":
    assert len(sys.argv) == 4, "invalids arguments"

    semi_global_matching = SemiGlobalMatching(
        path_left=sys.argv[1],
        path_right=sys.argv[2],
        max_disp=64,
        census_size=5,
        filter_size=5,
        occlusion_seuil=55,
        penalty_1=10,
        penalty_2=30,
    )

    semi_global_matching.SGM_process()

    io.imsave(sys.argv[3], semi_global_matching.disparities)
