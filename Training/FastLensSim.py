import indexTricks as iT
import numpy
from pylens import *
from imageSim import profiles,convolve,SBModels, models
import distances as D
import cPickle
import indexTricks as iT
import numpy,pylab
#from imageSim import profiles,convolve,models
import pylab as plt
from Surveys import Survey
from StochasticObserving import SO
from SignaltoNoise import S2N

class FastLensSim(SO, S2N):

    def __init__(self, surveyname, fractionofseeing = 1):
        #setting surveyname to be the surveyname and declaring the parameters to be the same
        ### Read in survey
        self.surveyName = surveyname
        survey=Survey(surveyname)              #This stores typical survey  in Surveys.Survey
        self.survey = Survey(surveyname)
        self.pixelsize = self.survey.pixelsize #let self.pixelsixe be equal to the pixelsize of survey
        self.side = self.survey.side
        self.readnoise = self.survey.readnoise
        self.nexposures = self.survey.nexposures
        self.f_sky = self.survey.f_sky
        self.bands = self.survey.bands
        self.strategy = self.survey.strategy
        self.strategyx = self.survey.strategyx
        self.exposuretimes = {}
        self.zeropoints = {}
        self.stochasticobservingdata = {}
        self.gains = {}
        self.seeing = {}
        self.psfscale = {}
        self.psf = {}
        self.psfFFT = {}
        self.ET = {}
        self.SB = {}

        #for i in range of the length of bands in survey, set the parameters to self.parameter
        for i in range(len(survey.bands)):
            self.exposuretimes[survey.bands[i]] = survey.exposuretimes[i]
            self.zeropoints[survey.bands[i]] = survey.zeropoints[i]
            self.gains[survey.bands[i]] = survey.gains[i]
            self.stochasticobservingdata[survey.bands[i]] = survey.stochasticobservingdata[i]
        self.zeroexposuretime = survey.zeroexposuretime
        
        ###do some setup    
        self.xl = (self.side - 1.)/2.
        self.yl = (self.side - 1.)/2.
        self.x, self.y = iT.coords((self.side, self.side))            #creating an array of the sides coordinates
        self.r2 = (self.x - self.xl) ** 2 + (self.y - self.yl) ** 2 # r squared (r2) = distance between (x,y) and (xl, yl)
        self.pixelunits = False
        
        self.Reset()    #resets all parameters as seen in definition below
#_____________________________________________________________________________________________________________________-

    def Reset(self):
        self.sourcenumbers = []
        #Some objects that need pre-defining as dictionaries
        self.magnification = {}
        self.image = {}
        self.sigma = {}
        self.residual = {}
        self.zeroMagCounts = {}
        self.xSum = {}
        self.ySum = {}
        self.ms = {}
        self.qs = {}
        self.ps = {}
        self.rs = {}
        self.ns = {}
        self.bl = {}
        self.src = {}
        self.galModel = {}
        self.sourceModel = {}
        self.model = {}
        self.totallensedsrcmag={}
        self.fakeLens={}
        self.fakeResidual={}
        self.fakeResidual[0]={}
        self.SN={}
        self.SNRF={}
        self.convolvedsrc={}
        self.convolvedGal={}

#______________________________________________________________________________________________________________

    def trytoconvert(self,par,p):        
        try:
            return par/p
        except NameError:
            print "Warning one of the parameters is not defined"

#_______________________________________________________________________________________________________________

    def setLensPars(self, m, r, q, n = 4, pixelunits = False, reset = True, xb = 0, xp = 0, jiggle = 0):

        if reset: self.Reset()              #if reset is true, reset all parameters
        self.rl={}                          #define rl variable as a dictionary

        if pixelunits == False:                                         #if there is no pixelunits
            
            # loop to fill rl array with bands and pixelsizes
            for band in r.keys():                                       #if band is within the dictionary r keys, then...
                self.rl[band] = self.trytoconvert(r[band], self.pixelsize) # the rl band array is set to be the rband captured by the trytoconvert method
        self.ml = m  # defining an array of band, pixelsize where ml is nodel?
        self.ql = q  
        
        self.deltaxl = (numpy.random.rand() - 0.5) * 2 * jiggle   #delta xl coordinate = 0, when jiggle = 0
        self.deltayl = (numpy.random.rand() - 0.5) * 2 * jiggle   #delta yl coordinate, when jiggle = 0
        if jiggle != 0:                                        
            self.deltap = 0.0 + (numpy.random.rand() - 0.5) * 180 # if jiggle is not 0, then delta p is numpy.random.rand()-0.5)*180
            n = (numpy.random.rand() + 1) * 4
        else:
            self.deltap = 0.0

        self.nl = n 
        
        # defining gal 
        self.gal = SBModels.Sersic('gal',{'x':self.xl + self.deltaxl,  
                                          'y':self.yl + self.deltayl,
                                          'q':self.ql,
                                          'pa':90+self.deltap,
                                          're':self.rl[band],
                                          'n':self.nl})
        
        #defining xb and xp as themselves in this class
        self.xb = xb
        self.xp = xp

#_________________________________________________________________________________________________________

    def setSourcePars( self, b, m, x, y, q, p, r, n = 1, pixelunits = False, sourcenumber = 1):
        if pixelunits == False:                                        # if pixelunits doesnt exist
            x = self.trytoconvert(x, self.pixelsize)                   # x is the result of trytoconvert function where par is x, and p is the pixelsize 
            y = self.trytoconvert(y, self.pixelsize)                   # y is the result of trytoconvert function where par is y, and p is the pixelsize
            r = self.trytoconvert(r, self.pixelsize)                   # r is the result of trytoconvert function where par is r, and p is the pixelsize
            b = self.trytoconvert(b, self.pixelsize)                   # b is the result of trytoconvert function where par is b, and p is the pixelsize
        self.xSum[sourcenumber] = x + self.xl + self.deltaxl + 0.000001   # xsum with a sourcenumber of 1 is the sum of all x 
        self.ySum[sourcenumber] = y + self.yl + self.deltayl + 0.000001           # ysum with a sourcenumber of 1 is the sum of all y 
        self.ms[sourcenumber] = m
        self.qs[sourcenumber] = q
        self.ps[sourcenumber] = p
        self.rs[sourcenumber] = r
        self.ns[sourcenumber] = n
        self.bl[sourcenumber] = b
        
        # The identified sourcenumber in the source dictionary is stated. 
        # where x = xSum, y = ySum, q = qs, pa = ps, re =rs, n=ns where sourcenumber is the key
        self.src[sourcenumber] = SBModels.Sersic('src%i'%sourcenumber, {'x':self.xSum[sourcenumber],
                                                                      'y':self.ySum[sourcenumber],
                                                                      'q':self.qs[sourcenumber],
                                                                      'pa':self.ps[sourcenumber],
                                                                      're':self.rs[sourcenumber],
                                                                      'n':self.ns[sourcenumber]})

        # sourcenumbers array increases with source number
        self.sourcenumbers.append(sourcenumber)

        # sourceModel, totallensedssrcmag, fakeResidual, SN, SNRF, convolvedsrc using the 
        # input of sourcenumber is now a dictonary
        self.sourceModel[sourcenumber] = {}
        self.totallensedsrcmag[sourcenumber] = {}
        self.fakeResidual[sourcenumber] = {}
        self.SN[sourcenumber] = {}
        self.SNRF[sourcenumber] = {}
        self.convolvedsrc[sourcenumber] = {}

#______________________________________________________________________________________________________________________

    def lensASource(self,sourcenumber,bands):
        
        # src is created as a sourcenumber
        src = self.src[sourcenumber]

        # lens is a powerlaw instance of massmodel which includes x, y, q, pa, b, eta
        lens = massmodel.PowerLaw('lens',{},{'x':self.xl+self.deltaxl,
                                            'y':self.yl+self.deltayl,
                                            'q':self.ql,
                                            'pa':90+self.deltap,
                                            'b':self.bl[sourcenumber],
                                            'eta':1})

        # es is a ExtShear instance of massmodel. Not sure what ExtShear is
        es = massmodel.ExtShear('lens',{},{'x':self.xl+self.deltaxl,
                                           'y':self.yl+self.deltayl,
                                           'pa':self.xp,
                                           'b':self.xb})

        # list lenses created including lens and es 
        lenses = [lens, es]

        a = 51

        # makes array of ox, and oy which shows the shape of the array as a float
        ox , oy = iT.coords((a,a))
        
        # ps is a float which is calculated by taking the rs value and multiplying it by (10.0/a)
        ps = (self.rs[sourcenumber] * (10.0 / a))

        ox = (ox - (a - 1) / 2.0) * ps + (self.xSum[sourcenumber]) # ox are arrays in which values are calculates as ox-(a-1)/2 + the xSum
        oy = (oy - (a - 1) / 2.0) * ps + (self.ySum[sourcenumber]) # oy are arrays in which values are calculates as oy-(a-1)/2 + the ySum

        unlensedSourceModel = (src.pixeval( ox, oy, csub = 5) * ( ps ** 2)).sum() #sum of pixelvalues of souces 
        sourceNorm = unlensedSourceModel.sum() # sum of unlensedSourceModel
        unlensedSourceModel /= sourceNorm      # unlensedSourceModel = unlensedSourceModel / sourceNorm

        #creating a model for the lenses
        sourceModel = pylens.lens_images(lenses, src, [self.x,self.y], getPix = True, csub = 5)[0]
        sourceModel[sourceModel < 0] = 0  # if the sourceModel is < 0,set it to be 0
        sourceModel /= sourceNorm #sourceModel = sourceModel / sourceNorm

        #creating a magnification for each source
        self.magnification[sourcenumber] = (numpy.sum(numpy.ravel(sourceModel)) / numpy.sum(numpy.ravel(unlensedSourceModel)))
        sm = {}

        #for each band that is seen in the bands list:
        # unlensedTotalSourceFlux is calculated as flux of the a band with regards to the sourcenumber,
        # and that same band at the zeropoint 
        # important to note:  flux = 10 ** (0.4 *( m2 - m1)), where m2 is the zeropoints
        #                  or fluw = 10 ** (-(m1 - m2) / 2.5)

        for band in bands:
            unlensedtotalsrcflux = 10 **(-(self.ms[sourcenumber][band] - self.zeropoints[band]) / 2.5)
            sm[band] = sourceModel * unlensedtotalsrcflux 
        
            if sm[band].max() > 0:
                
                # if the maximum value in sm[band] > 0, then the magnitude calculated
                # magnitude = m1/m2 ((m1/m2) from calculations of flux in comments above) 

                self.totallensedsrcmag[sourcenumber][band] = -2.5 * numpy.log10(sm[band].sum()) + self.zeropoints[band]
            else:
                # else magnitude = 99
                self.totallensedsrcmag[sourcenumber][band] = 99
        return sm

#_____________________________________________________________________________________________________________________
    # defining a function to Evaluate a Galaxy
    def EvaluateGalaxy(self, light, mag, bands):
        model = {}                  # model is defined 
        lightMag = light.pixeval(self.x, self.y, csub = 5) # the magnitude of light in the pixels are calculated in lightMag array
        lightMag [lightMag < 0] = 0 # set the lightMag to if it is less than 0
        lightMag /= lightMag.sum()  # lightMag = lightMag / sum of the lightMag
        
        # for each band that is seen in the bands list:
        # calculat the flux in each band regarding the magnitude
        # a model array calculated
        for band in bands:
            flux = 10 ** (-(mag[band] - self.zeropoints[band]) /2.5)
            model[band] = lightMag * flux

        return model

#______________________________________________________________________________________________________________________ 
      
    def MakeModel(self, bands):
        #did you know that self.gal is actually fixed for all bands currently?
        self.galModel = self.EvaluateGalaxy(self.gal,self.ml,bands)

        for sourcenumber in self.sourcenumbers:
            self.sourceModel[sourcenumber] = self.lensASource(sourcenumber,bands)
            # sourceModel is set to be the lensASource function including sourcenumber and bands

        for band in bands:
            self.model[band] = self.galModel[band] * 1
            for sourcenumber in self.sourcenumbers:
                self.model[band] += self.sourceModel[sourcenumber][band]

#_________________________________________________________________________________________________________________________

    def ObserveLens(self, noisy = True, bands = []):
        if bands == []:
            bands = self.bands # if bands exists, then bands is set to be the bands
      
        for band in bands:               
            if self.seeing[band] == 0:   
                # convolvedGalaxy and the psfForwardFourierTransform is the convolution of the galaxyModel and the psf from convolve.py and edgeCheck is True
                convolvedGalaxy, self.psfFFT[band] = convolve.convolve(self.galModel[band], self.psf[band], True)
              
                convolvedGalaxy[convolvedGalaxy <0] = 0
                self.convolvedGal[band] = convolvedGalaxy


                convolvedmodel = convolvedGalaxy * 1
                convolvedsrc = {}
                
                for sourcenumber in self.sourcenumbers:
                    convolvedsrc[sourcenumber] = convolve.convolve(self.sourceModel[sourcenumber][band], self.psfFFT[band], False)[0]
                    convolvedsrc[sourcenumber][convolvedsrc[sourcenumber]<0]=0
                    self.convolvedsrc[sourcenumber][band]=convolvedsrc[sourcenumber]
                    convolvedmodel+=convolvedsrc[sourcenumber]

                self.zeroMagCounts[band]=(10**(-(0-self.zeropoints[band])/2.5))

                exposurecorrection=((self.ET[band]*1./self.zeroexposuretime))*self.gains[band]
                convolvedmodel*=exposurecorrection

                #skybackground per second per square arcsecond
                background=(10**(-(self.SB[band]-self.zeropoints[band])/2.5))*(self.pixelsize**2)
                tot_bg=background*exposurecorrection
                
                sigma=((convolvedmodel+tot_bg)+self.nexposures*(self.readnoise**0.5)**2)**.5

                fakeLens=convolvedmodel*1.
                if noisy:fakeLens+=(numpy.random.randn(self.side,self.side)*(sigma))

                #convert back to ADU/second:
                fakeLens/=exposurecorrection
                sigma/=exposurecorrection

                self.image[band]=fakeLens*1 
                self.fakeLens[band]=fakeLens*1
                self.sigma[band]=sigma*1
                self.fakeResidual[0][band]=fakeLens-convolvedGalaxy
                for sourcenumber in self.sourcenumbers:
                    self.SN[sourcenumber][band]=self.SNfunc(\
                        convolvedsrc[sourcenumber],sigma)
                    self.fakeResidual[sourcenumber][band]=\
                        fakeLens-convolvedmodel+convolvedsrc[sourcenumber]
            
#===========================================================================
    def loadModel(self,ideallens):
        if ideallens is not None:
            self.galModel,self.sourceModel,self.model,self.magnification,self.totallensedsrcmag=ideallens
            self.image=self.model

#===========================================================================

    def loadConvolvedModel(self,ideallens):
        self.galModel,self.sourceModel,self.model,self.magnification,self.totallensedsrcmag=ideallens
        self.image=self.model


#===========================================================================

    def makeLens(self, stochastic=True, save=False, noisy=True, stochasticmode="MP", SOdraw=[], bands=[], musthaveallbands=False, MakeModel=True):
        if stochastic==True:
            self.stochasticObserving(mode=stochasticmode, SOdraw=SOdraw, musthaveallbands=musthaveallbands)
        
        if self.seeingtest=="Fail":
            return None

        if bands==[]:
            bands=self.bands
        
        if MakeModel:
            self.MakeModel(bands)

        if self.strategy=="resolve" and stochastic==True:
            self.stochasticObserving(mode=stochasticmode,SOdraw=SOdraw) #have to rerun stochastic observing now we know the magnification

        self.ObserveLens(noisy=noisy)
        return [self.galModel, self.sourceModel, self.model, self.magnification, self.totallensedsrcmag]
        
        
#===========================================================================


    def makeColorLens(self,bands=["g_SDSS","r_SDSS","i_SDSS"],recolourize=True):
        if self.surveyName=="Euclid" and bands==["g_SDSS","r_SDSS","i_SDSS"]:
            bands=["VIS","VIS","VIS"]
        import colorImage
        goodbands=[]
        for band in bands:
            try:
                self.image[band]
                goodbands.append(band)
            except KeyError:
                pass
        bands=goodbands
        if len(bands)==1:
            bands=[bands[0],bands[0],bands[0]]
        if len(bands)==2:
            bands=[bands[0],"dummy",bands[1]]
            self.ml["dummy"]=(self.ml[bands[0]]+self.ml[bands[2]])/2
            self.image["dummy"]=(self.image[bands[0]]+self.image[bands[2]])/2
        if recolourize:
            self.color = colorImage.ColorImage()
            self.color.bMinusr=(self.ml[bands[0]]-self.ml[bands[2]])/4.
            self.color.bMinusg=(self.ml[bands[0]]-self.ml[bands[1]])/4.
            self.color.nonlin=4.
            self.colorimage = self.color.createModel(\
                 self.image[bands[0]],self.image[bands[1]],self.image[bands[2]])
        else:
            self.colorimage = self.color.colorize(\
                 self.image[bands[0]],self.image[bands[1]],self.image[bands[2]])

        return self.colorimage


#===========================================================================

    def display(self,band="g_SDSS",bands=["g_SDSS","r_SDSS","i_SDSS"]):
        if self.surveyName=="Euclid":bands=["VIS","VIS","VIS"]
        import pylab as plt
        plt.ion()
        plt.figure(1)
        plt.imshow(self.makeColorLens(bands=bands),interpolation="none")
        plt.figure(2)
        import colorImage
        self.color = colorImage.ColorImage()#sigma-clipped single band residual
        plt.imshow(self.color.createModel(self.fakeResidual[0][band],self.fakeResidual[0][band],self.fakeResidual[0][band])[:,:,0],interpolation="none")
        plt.figure(3)
        plt.imshow(self.fakeResidual[0][band],interpolation="none")
        try:
            self.fakeResidual[1]["RF"]
            plt.figure(4)
            plt.imshow(self.fakeResidual[1]["RF"],interpolation="none")
        except KeyError: pass

        plt.draw()
        raw_input()
        plt.ioff()

#===========================================================================

    def Rank(self,mode,band="g_SDSS",bands=["g_SDSS","r_SDSS","i_SDSS"]):
        import pylab as plt
        plt.ion()
        rank="d"
        while rank not in ["0","1","2","3","4","-1","-2","-3"]:
            if mode=="colour":
                plt.imshow(self.makeColorLens(bands=bands),interpolation="none")
                plt.draw()
            if mode=="rf":
                plt.imshow(self.fakeResidual[0]["RF"],interpolation="none")
                plt.draw()
            if mode=="best":
                plt.imshow(self.fakeResidual[0][band],interpolation="none")
                plt.draw()
            rank=raw_input()
            if rank=="":rank="0"
        plt.ioff()
        return rank
#===========================================================================
