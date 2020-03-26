"""

Example of how to figure out which DES tile given RA, dec coords are in.


"""

import DESTiler

# Set-up of this is slow, so do only once...
tiler=DESTiler.DESTiler("KnownLenses/DES_DR1_TILE_INFO.csv")

# How to get tile name
RADeg, decDeg=10, -40
tileName=tiler.getTileName(RADeg, decDeg)

# How to fetch all images for tile which contains given coords
tiler.fetchTileImages(RADeg, decDeg, 'DESTileImages')

