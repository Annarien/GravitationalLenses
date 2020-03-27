"""

Example of how to figure out which DES tile given RA, dec coords are in.


"""
# IMPORTS
import DESTiler
#_____________________________________________________________________________________________________________________________
# FUNCTIONS
#_____________________________________________________________________________________________________________________________
# MAIN

# Set-up of this is slow, so do only once...
tiler=DESTiler.DESTiler("KnownLenses/DES_DR1_TILE_INFO.csv")

# How to get tile name
RADeg, decDeg=10, -40
tileName=tiler.getTileName(RADeg, decDeg)

# How to fetch all images for tile which contains given coords
tiler.fetchTileImages(RADeg, decDeg, 'KnownLenses/DESTileImages/%s/' % tileName)

# # TESTING with arrays:
# RADeg = [-1.055084, -1.055798, -1.055363, -42.136973, -42.139283]
# decDeg = [1.055084, 1.055798, 1.055363, 42.136973, 42.139283]

# for num in range(len(RADeg)):
#     ra = RADeg[num]
#     dec = decDeg[num]
#     tileName = tiler.getTileName(ra, dec)
#     tiler.fetchTileImages(ra, dec, num, tileName) 




