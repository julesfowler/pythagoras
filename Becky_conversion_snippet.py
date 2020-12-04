#1) Calculating modal coefficients from the DM commands (volts) or WFS residuals (volts):


def pupil(N):
    p = np.zeros([N,N])
    radius = N/2.
    [X,Y] = np.meshgrid(np.linspace(-(N-1)/2.,(N-1)/2.,N),np.linspace(-(N-1)/2.,(N-1)/2.,N))
    R = np.sqrt(pow(X,2)+pow(Y,2))
    p[R<=radius] = 1
    return p

zern = fits.open('./zbasis_cmds.fits')[0].data[:,pup] 
zern_pinv = np.linalg.pinv(zern[0:250, :]) # 0:mode_truncation

DM_commands = np.load('./dm.npy')[:,pup]
DM_commands_modal = np.dot(DM_commands, zern_pinv)

#2) Extracting the stellar photometry from the QACITS PSF:

#In the fits header of psf_noscale.fits, there's an entry called STARPHOT and an entry called SCALEFAC. STARPHOT is just the counts inside a 1FWHM aperture centered on the PSF, where the FWHM was found by fitting a 2D gaussian to the PSF using VIP. SCALEFAC is the following:

#SCALEFAC = (science exposure integration time * science exposure coadds) / (PSF exposure integration time * PSF exposure coadds)

#So, for the PSF to be representative of the star for the same integration time and number of coadds as the science exposure, you'll want to multiply psf_noscale.fits by STARPHOT * SCALEFAC.
