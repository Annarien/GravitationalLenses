from __init__ import *
import cPickle
#import pyfits
import sys
import pylab as plt
import time
from tabulate import tabulate
import plotly.plotly as pyt  
import plotly.graph_objs as got
sigfloor=200

#Load lens population
lensSample = LensSample(reset=False,sigfloor=sigfloor,cosmo=[0.3,0.7,0.7]) 

#sys.argv are command line arguments passed in during execution. For example: python ModelAll.py <arg1> <arg2> ...
#setting up conditions for execution if no args were provided.
experiment = sys.argv[1] if len(sys.argv) > 1 else "Euclid"
fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
snThreshold = int(sys.argv[3]) if len(sys.argv) > 3 else 20
magnificationThreshold = int(sys.argv[4]) if len(sys.argv) > 4 else 3
c = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
d = int(sys.argv[6]) if len(sys.argv) > 6 else 1000

firstOD = 1
nSources = 1
surveys = []

#switch method in python, establishing which surveys to use
#if experiment == "Euclid":
#    surveys += ["Euclid"]
#if experiment == "CFHT":
#    surveys += ["CFHT"]    #full coadd (Gaussianised)
#if experiment == "CFHTa":
#    surveys += ["CFHTa"]   #dummy CFHT
if experiment == "DES":
    surveys += ["DESc"]    #Optimal stacking of data
    surveys += ["DESb"]    #Best Single epoch image
    surveys += ["DESa"]    #full coadd (Gaussianised)
#if experiment == "LSST":
#    surveys += ["LSSTc"]   #Optimal stacking of data
#    surveys += ["LSSTb"]   #Best Single epoch image
#    surveys += ["LSSTa"]   #full coadd (Gaussianised)

simulation = {}
n = {}
for survey in surveys:
    simulation[survey] = FastLensSim(survey,fractionofseeing=1)
    simulation[survey].bfac = float(2)
    simulation[survey].rfac = float(2)

#scientific notation for t0 = the first timestamp
t0 = time.clock()

dataFile = open("dataFile.txt","a")
dataFile.write("| Folder    | zs (source redshift) | zl (lens redshift)  | mag (magnification)      | b (einstein radius)     | ms(r band) | ml (r band)       | rfsn (Ring Finder SN) \n")
dataFile.close()
#for sourcepop in ["lsst","cosmos"]:
for sourcePopulation in ["lsst"]:
    chunk = 0
    Si = 0
    SSPL = {}
    foundCount = {}

    #for each survey, set the initial count of "foundCount" to 0.
    for survey in surveys:
        foundCount[survey] = 0

    #define variable nAll as the number of all elements in the source population.
    if sourcePopulation == "cosmos":
        nAll = 1100000
    elif sourcePopulation == "lsst":
        nAll=12530000

    #downscale the number of all elements by the fraction provided as an argument / default value :)
    nAll=int(nAll*fraction)

    for i in range(nAll):
        #for every 10,000th element, print & load the lens population
        if i % 10000 == 0:
            print ("about to load")
            lensSample.LoadLensPop(i,sourcePopulation)
            print ("i=%d, nAll=%d"%(i,nAll))

        if i != 0 and i % 10000 == 0 or i == 100 or i == 300 or i == 1000 or i == 3000:
                t1 = time.clock()             #scientific notation for t1 = the current timestamp.
                ti = (t1-t0)/float(i)         #time interval: get the difference (delta) between t1 and t0 and divide it by the current index in the loop, getting average time for this iteration.
                tl = (nAll-i)*ti              #time left in seconds.
                tl/=60                        #time left in minutes
                hl = numpy.floor(tl/(60))     #get the number of hours left.
                ml = tl-(hl*60)               #get the number of minutes left.
                print i,"%ih%im left"%(hl,ml) #time detemined left to run

        #discard this lense from the sample if the condition lens is false.
        lenspars = lensSample.lens[i]
        if lenspars["lens?"] == False:
            del lensSample.lens[i]
            continue #skip the rest of the execution for the current value of i and move on to i = i+1

        #set [rl][VIS] for this lens sample to the average colour of R, I & Z.
        lenspars["rl"]["VIS"] = (lenspars["rl"]["r_SDSS"]+lenspars["rl"]["i_SDSS"]+lenspars["rl"]["z_SDSS"])/3

        #set [ml][VIS] for this lens sample to the average colour of R, I & Z.
        lenspars["ml"]["VIS"] = (lenspars["ml"]["r_SDSS"]+lenspars["ml"]["i_SDSS"]+lenspars["ml"]["z_SDSS"])/3

        #set [ms][1][VIS] for this lens sample to the average colour of R, I & Z.
        lenspars["ms"][1]["VIS"] = (lenspars["ms"][1]["r_SDSS"]+lenspars["ms"][1]["i_SDSS"]+lenspars["ms"][1]["z_SDSS"])/3

        #if lenspars["zl"]>=0.3 or lenspars["b"]<=2.0:continue # this is a CFHT compare quick n dirty test
        #if lenspars["zl"]<0.3: continue 
        #if lenspars["b"]<=2.0 :continue   

        # strtofile = str(lenspars["ms"][1]["VIS"])+" "+str(lenspars["ms"][1]["VIS"])
        #some more setup for this lens pars
        lenspars["mag"]={}
        lenspars["msrc"]={}
        
        lenspars["msrc"]={}
        lenspars["SN"]={}
        lenspars["bestband"]={}
        lenspars["pf"]={}
        lenspars["resolved"]={}
        lenspars["poptag"]={}
        lenspars["seeing"]={}
        lenspars["rfpf"]={}
        lenspars["rfsn"]={}

        #iterate over the surveys, setting the
        lastSurvey="non"
        accepted = False    
        for survey in surveys:
            simulation[survey].setLensPars(lenspars["ml"], lenspars["rl"], lenspars["ql"], reset=True)
            for j in range(nSources):
                simulation[survey].setSourcePars(lenspars["b"][j+1],\
                                        lenspars["ms"][j+1],\
                                        lenspars["xs"][j+1],\
                                        lenspars["ys"][j+1],\
                                        lenspars["qs"][j+1],\
                                        lenspars["ps"][j+1],\
                                        lenspars["rs"][j+1],\
                                        sourcenumber=j+1)

            if survey[0:3] + str(i) != lastSurvey:
                model = simulation[survey].makeLens(stochasticmode="MP")
                SOdraw = numpy.array(simulation[survey].SOdraw)

                #if model is not a None type, set the lastSurvey to the first 3 chars of the survye name + i. e.g.: DES0.
                if type(model) != type(None):
                    lastSurvey = survey[0:3] + str(i)

                #if the survey's seeing test failed, set everything to default & false.
                if simulation[survey].seeingtest == "Fail":
                    lenspars["pf"][survey] = {}
                    lenspars["rfpf"][survey] = {}
                    for srcNumber in simulation[survey].sourcenumbers:
                        lenspars["pf"][survey][srcNumber] = False
                        lenspars["rfpf"][survey][srcNumber] = False
                    continue
            else: 
                simulation[survey].loadModel(model)
                simulation[survey].stochasticObserving(mode="MP",SOdraw=SOdraw)

                #if the survey's seeing test failed, set everything to default & false.
                if simulation[survey].seeingtest == "Fail":
                    lenspars["pf"][survey] = {}
                    for srcNumber in simulation[survey].sourcenumbers:
                        lenspars["pf"][survey][srcNumber] = False
                    continue

                simulation[survey].ObserveLens()

            #setting up some defaults :)
            lenspars["SN"][survey] = {}
            lenspars["bestband"][survey] = {}
            lenspars["pf"][survey] = {}
            lenspars["resolved"][survey] = {}
            lenspars["poptag"][survey] = i
            lenspars["seeing"][survey] = simulation[survey].seeing
            rfpf = {}
            rfsn = {}

            #extract the source meta data into separate variables
            mag, msrc, SN, bestband, pf = simulation[survey].SourceMetaData(SNcutA=snThreshold, magcut=magnificationThreshold, SNcutB=[c,d])
            for srcNumber in simulation[survey].sourcenumbers:
                rfpf[srcNumber] = False      
                rfsn[srcNumber] = [0]
                lenspars["mag"][srcNumber] = mag[srcNumber]
                lenspars["msrc"][srcNumber] = msrc[srcNumber]
                lenspars["SN"][survey][srcNumber] = SN[srcNumber]
                lenspars["bestband"][survey][srcNumber] = bestband[srcNumber]
                lenspars["pf"][survey][srcNumber] = pf[srcNumber]
                lenspars["resolved"][survey][srcNumber] = simulation[survey].resolved[srcNumber]
               
                # print (rfpf[srcNumber])
                # print (rfsn[srcNumber])
                # print (lenspars["mag"])
                # print(lenspars["SN"])
                # print(lenspars["bestband"])
                # print(lenspars["pf"])
                # print(lenspars["resolved"])

            #get the value for rfpf & rfsn and set it for the current survey.
            if survey != "Euclid" and simulation[survey].seeingtest != "Fail":
                if survey not in ["CFHT", "CFHTa"]:
                    simulation[survey].makeLens(noisy=True, stochasticmode="1P", SOdraw=SOdraw, MakeModel=False)
                    mode = "donotcrossconvolve"
                else:
                    mode = "crossconvolve"

                rfpf, rfsn = simulation[survey].RingFinderSN(SNcutA=snThreshold, magcut=magnificationThreshold, SNcutB=[c,d], mode=mode)
            
            lenspars["rfpf"][survey] = rfpf
            lenspars["rfsn"][survey] = rfsn
            
            ###
            #This is where you can add your own lens finder
            #e.g.
            
            #found=Myfinder(S[survey].image,S[survey].sigma,S[survey].psf,S[survey].psfFFT)
            #NB/ image,sigma, psf, psfFFT are dictionaries 
            #    The keywords are the filters, e.g. "g_SDSS", "VIS" etc

            #then save any outputs you'll need to the lenspars dictionary:
            #lenspars["my_finder_result"]=found

            ###

            #If you want to save the images (it may well be a lot of data!):
            import pyfits #(or the astropy equivalent)  #saving a pyfits 

            import os
            folder="fits_images_parameters"
            if not os.path.exists(folder):#if fits image directory doesnt exist, make it
                os.makedirs(folder)

            folder="%s/%i"%(folder,i)
            if not os.path.exists(folder):#if loop folder(i) doesnt exist, make it
                 os.makedirs(folder)

            # make text file here
            # open file and creating a data file

            #import json
            # dataFile = (dataFile.json,"a")
            dataFile = open("dataFile.txt","a")
            # need to tabulate zs, zl, mag, b, msrc, ml
            #dataFile.write("| Folder        | zs            | zl            | mag           | b             | msrc          | ml            |")
            dataFile.write("| "+str(i)+(" "*(10-len(str(i))))+
                           "| "+str(lenspars["zs"])+(" "*(21-len(str(lenspars["zs"]))))+
                           "| "+str(lenspars["zl"])+(" "*(20-len(str(lenspars["zl"]))))+ 
                           "| "+str(lenspars["mag"])+(" "*(25-len(str(lenspars["mag"]))))+
                           "| "+str(lenspars["b"])+(" "*(24-len(str(lenspars["b"]))))+
                           "|"+str(lenspars["ms"][1]["r_SDSS"])+(" "*(12-len(str(lenspars["ms"][1]["r_SDSS"]))))+
                           "|"+str(lenspars["ml"]["r_SDSS"])+(" "*(19-len(str(lenspars["ml"]["r_SDSS"]))))+
                           "|"+str(lenspars["rfsn"])+                           "\n")
                       
                        #        + str(lenspars["zl"])
                        #        + str(lenspars["mag"])
                        #        + str(lenspars["b"])
                        #        + str(lenspars["msrc"])
                        #        + str(lenspars["ml"]))
            dataFile.close()

            for band in simulation[survey].bands:
                
                img = simulation[survey].image[band]
                sig = simulation[survey].sigma[band]
                psf = simulation[survey].psf[band] 
                
                #The lens subtracted
                #resid contains the lensed source, with the lens subtracted
                #assuming the subtraction is poisson noise limited (i.e. ideal)
                resid = simulation[survey].fakeResidual[0][band]

                pyfits.PrimaryHDU(img).writeto("%s/image_%s.fits"%(folder,band), clobber=True)            
                pyfits.PrimaryHDU(sig).writeto("%s/sigma_%s.fits"%(folder,band), clobber=True)                   
                pyfits.PrimaryHDU(psf).writeto("%s/psf_%s.fits"%(folder,band), clobber=True)                     
                pyfits.PrimaryHDU(resid).writeto("%s/galsub_%s.fits"%(folder,band), clobber=True)

            #delete used data to save memory
            lensSample.lens[i]=None


            if lenspars["pf"][survey][1]:
                accepted = True

        if accepted:
            # simulation[survey].display(band="VIS", bands=["VIS", "VIS", "VIS"])
            # if Si > 100:
            #     exit()
                
            #for every 1000th element in lenspars, open the file and     
            Si+=1
            SSPL[Si] = lenspars.copy() #make a copy -> reference safety :)
            if (Si+1) % 1000 == 0:
                f = open("LensStats/%s_%s_Lens_stats_%i.pkl"%(experiment, sourcePopulation, chunk), "wb")
                cPickle.dump([fraction, SSPL], f, 2)
                f.close()
                SSPL = {} # reset SSPL or memory fills up
                chunk+=1

        del lensSample.lens[i] #delete this lense as we done with it now

    f = open("LensStats/%s_%s_Lens_stats_%i.pkl"%(experiment, sourcePopulation, chunk), "wb")
    cPickle.dump([fraction, SSPL], f, 2)
    f.close()
    print Si


bl = []
for j in SSPL.keys():
    try: 
        if SSPL[j]["rfpf"][survey][1]:
            bl.append(SSPL[j]["b"][1])
    except KeyError:
        pass


