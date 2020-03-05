import random 

# create a definition for parameters to be created below 
#class RandomParameters():
def random_ml(self, LensMag): # Mags for lens (dictionary of magnitudes by band)
    # random ml (in ranges) from cosmos and Des
    #for i in range(3):
    #    num = float(random.uniform(1,20)) # range of ml
    if LensMag == 20:
        ml = {'g_SDSS' : 20.0,
            'r_SDSS' : 13.0,
            'i_SDSS' : 14.0}
    return(ml)

def random_rl():    # Lens half-light radius in arcsec (weirdly, dictionary by band, all values the same, presumably arcsec)
    # random rl
    rl = random.random(0,3)
    rl={'g_SDSS': 3.0,      
        'r_SDSS': 3.0,
        'i_SDSS': 3.0}

def random_ql():    # Lens flattening (1 = circular, 0 = line)
    # random ql
    ql = random.random(0,1)
    ql = 0.5

def random_b():     # Einstein radius in arcsec
    # random b
    b = random.random(2,6) 
    b = 5.0
    
def random_ms():    
    # random ms
    ms = random.random(1,20)
    ms={'g_SDSS': 19.0,     # Mags for source (dictionary of magnitudes by band)
        'r_SDSS': 19.0,     
        'i_SDSS': 19.0} 

def random_xs():    # x-coord of source relative to lens centre in arcsec
    # random xs
    xs = random.random(0,5)
    xs = 1.0

def random_ys():       # y-coord of source relative to lens centre in arcsec
    # random ys
    ys = random.random(0,5)
    ys = 1.0

def random_qs():    # Source flattening (1 = circular, 0 = line)
    # random qs
    qs =random.random(0,5)
    qs = 1.0

def random_ps():    # Position angle of source (in degrees)
    # random ps
    ps = random.random(1,90)
    ps = 90.0

def random_rs():    # Source half-light radius in arcsec
    # random rs (higher than rl)   
    rs = random.random(0,5)
    rs = 2.0