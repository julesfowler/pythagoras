'''
Created Nov. 3, 2021 
Author: Ben Calvin

Meant to be used as a foundation for working with HCI data
'''

import numpy as np
import matplotlib.pyplot as plt
import hcipy as hci

import time as time_pkg#just nice to have

################################################################
### SETUP
################################################################
wavelength_wfs = 842.0E-9 #meters
telescope_diameter = 10.0 #meters

#make pupil slightly larger than the telescope diameter to control edge effects
pupil_grid_diameter = 60/56 * telescope_diameter 
num_pupil_pixels = 100 #100 pupil pixels across the pupil diameter

#pupil_grid is the fundamental "what apertures get evaluated on"
pupil_grid = hci.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

Hz = 1e3

################################################################
### FUNCTIONS
################################################################

def make_keck_ap(grid):
    '''
    Iteratively step through the rings of keck to generate the primary mirror aperture.
    This currently ignores any spiders or gaps between mirrors.
    '''
    #Find the positions where there are mirrors
    centroid_pos = [[0,0]]
    radials = [1,2,3]
    in_seg_len = 10/7 #meters from one flat edge of a mirror segment to the other
    ang_types = np.arange(30,360, 60) + 120
    for rad in radials:
        cur_x = rad*in_seg_len*np.cos(30*np.pi/180)
        cur_y = rad*in_seg_len*np.sin(30*np.pi/180)
        for ang in ang_types:
            for iii in range(rad):
                cur_x += in_seg_len*np.cos(ang*np.pi/180)
                cur_y += in_seg_len*np.sin(ang*np.pi/180)
                
                centroid_pos.append([cur_x, cur_y])
                
    #Now put mirrors on each of those points
    out_seg_len = 20*np.sqrt(3)/21 #size of the circle enclosing the hexagons
    oversamp = 10
    
    keck_aperture = -5*hci.evaluate_supersampled(hci.circular_aperture(2.4), 
                                                 grid, oversamp)
    for cent in centroid_pos:
        aper = hci.hexagonal_aperture(out_seg_len, angle=np.pi/6, center=cent)
        heck_aperture += hci.evaluate_supersampled(aper, grid, oversamp)
    keck_aperture[keck_aperture<0] = 0
    keck_aperture[keck_aperture>1] = 1
    
    #hci.imshow_field(keck_aperture)
    #plt.xlabel('x position(m)')
    #plt.ylabel('y position(m)')
    #plt.colorbar()
    #plt.show()
    return keck_aperture

def build_DM(grid, diam = telescope_diameter):
    '''
    Generate a DM like how I've done
    '''
    num_acts = 21 #21 x 21 DM (a lot of actuators are outside the aperture)
    act_spacing = diam / num_acts
    influence_functions = hci.make_gaussian_influence_functions(grid, num_acts, act_spacing)
    DM = hci.DeformableMirror(influence_functions)
    n_modes = DM.num_actuators
    DM.flatten()
    
    grid_size = np.mean(grid.delta * grid.dims)
    dm_grid = hci.make_pupil_grid(num_acts, grid_size)
    fft_grid = hci.make_fft_grid(dm_grid)
    Fouriers = hci.make_fourier_basis(dm_grid, fft_grid)
    Fouriers = hci.ModeBasis([mode for mode in Fouriers.orthogonalized], dm_grid)
    
    DM.actuators = Fouriers[0]
    return DM, dm_grid, Fouriers

def build_IM(pywfs, camera, img_ref, wf, DM, basis, wavelength=wavelength_wfs):
    probe_amp = 0.01 * wavelength
    slopes = []
    nmodes = DM.num_actuators
    for ind in range(nmodes):
        if (ind+1)%25 == 0:
            print('Measuring response to mode {:d} / {:d}'.format(ind+1, nmodes))
        slope = 0
        
        #Probe the phase response
        for s in [1, -1]:
            amp = basis[ind] * s * probe_amp #phase response to each basis mode
            DM.actuators = amp
            
            dm_wf = DM.forward(wf)
            wfs_wf= pywfs.forward(dm_wf)
            camera.integrate(wfs_wf, 1)
            image = camera.read_out()
            image /= np.sum(image)
            
            slope += s * (image-img_ref)/(2*probe_amp)
        slopes.append(slope)
    slopes = hci.ModeBasis(slopes)
    
    rcond = 1e-3
    IM_prime = hci.inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)
    return IM_prime

def build_PYWFS(grid, aperture, DM = None, basis=None, wavelength = wavelength_wfs):
    '''
    Builds the Pyramid, the camera reading the pyramid, and 
    optionally the control matrix for a DM
    
    OUTPUTS:
    pywfs: the pyramid that you pass a wavefront to to generate 4 pupils
    camera_py: the camera that reads out the pywfs output
    img_ref: the reference image of a flat wavefront
    IM_prime: the matrix that converts pywfs measurements to DM actuator space
    '''
    pywfs = hci.PyramidWavefrontSensorOptics(grid, wavelength_0=wavelength)
    camera_py = hci.NoiselessDetector(pywfs.output_grid)
    
    wf = hci.Wavefront(aperture, wavelength)
    wf.total_power = 1
    camera_py.integrate(pywfs.forward(wf), 1)
    img_ref = camera_py.read_out()
    img_ref /= img_ref.sum()
    
    if DM is None:
        return pywfs, camera_py, img_ref, None
    else:
        IM_prime = build_IM(pywfs, camera_py, img_ref, wf, DM, basis)
        return pywfs, camera_py, img_ref, IM_prime


#Fname means one wind layer at 9m/s, random seed 123, 1kHz sensing, 100 frames for 
# AO burn-in, and 4000 frames for reading/control. Units of radians phase.
fname = 'onelayer_9ms123_1kHz_100_4000.npy' #This one has an r0 of ~2meters

def make_wf(phase, aper, wavelength=wavelength_wfs):
    wf = hci.Wavefront(np.exp(1j*phase)*aper, wavelength)
    return wf

def read_atm(fname, grid, aper, Hz = Hz):
    atm_frames = np.load(fname)
    #The resultant numpy array is 2D where axis=0 is the time step and axis=1 is the
    # spatial info of the phase that got flattened by HCIPy. If you want to avoid using
    # the hci.Field and pupil stuff, you get the correct shaping by calling
    # np.reshape(atm_frames[tstep], (100,100)) #(num_pupil_pixels,num_pup_pix)
    
    phase_frames = []
    #wf_frames = []
    times = np.arange(1, atm_frames.shape[0]+1)/Hz
    for ind in range(atm_frames.shape[0]):
        phase = hci.Field(atm_frames[ind], grid)
        phase_frames.append(phase)
        #wf_frames.append(make_wf(phase, aper))
        
    return times, phase_frames#,wf_frames

