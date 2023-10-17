#! /usr/bin/env python


# LOADING FUNCTIONS
from lics_unwrap import *
import dask.array as da
#import dask_ndfilters as ndfilters    # use this if the below doesn't work!
from dask_image import ndfilters
import sys
import os
import subprocess
from scipy.ndimage import generic_filter
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter
import shutil
'''
This script is adapted from https://github.com/comet-licsar/licsar_proc/blob/main/python/tureq_proc.py to process the rubber-sheeting automatically in licsar system.

Nergizci, Lazecky 10/10/2023. 
'''


# LOADING FUNCTIONS

def get_disk_ones(block):
    mask=np.zeros(block.shape) #should be square
    #nyquistlen=int(mask.shape[0]/2+0.5) + 1 #+ extrapx
    circle=unit_circle(int(mask.shape[0]/2+0.5)-1) #will contain +1 px for zero
    i=int((mask.shape[0]-circle.shape[0])/2+0.5)
    j=int((mask.shape[1]-circle.shape[1])/2+0.5)
    mask[i:i+circle.shape[0],j:j+circle.shape[1]]=circle
    mask[mask==0]=np.nan
    return mask


def filterhistmed(block, amin, amax, bins=20, medbin=True):
    """Support function to be used with generic_filter (where only 1D array is passed, expecting one output->using median here only
    """
    if np.isnan(block).all():
        return np.nan
    histc, histe = np.histogram(block,range=(amin,amax), bins=bins)
    histmax=np.argmax(histc)
    #minval=histe[histmax]
    #maxval=histe[histmax+1]
    # tiny update
    if histmax == 0:
        histmax=1
    elif histmax == bins-1:
        histmax = bins-2
    minval=histe[histmax-1]
    maxval=histe[histmax+2]
    # add median or interpolate:
    if medbin:
        bb=block[block>minval]
        bb=bb[bb<maxval]
        outval = np.nanmedian(bb)
    else:
        try:
            blockxr=xr.DataArray(block)
            blockxr.values=interpolate_nans(blockxr.where(blockxr>minval).where(blockxr<maxval).values, method='linear')
            blockxr.values=interpolate_nans(blockxr.values, method='nearest') # just to be sure..
            outval = float(blockxr.values[int(block.shape[0]/2),int(block.shape[1]/2)])
        except:
            outval = np.nan
    return outval


def filter_histmed_ndarray(ndarr, winsize=32, bins=20, medbin=True):
    """Main filtering function, works with both numpy.ndarray and xr.DataArray
    Args:
        medbin (boolean): if False, it will interpolate (fit) the central value from the bin subset. otherwise returns its median
    """
    #footprint=disk(winsize)
    amin=np.nanmin(ndarr)
    amax=np.nanmax(ndarr)
    footprint=unit_circle(int(winsize/2))
    return generic_filter(ndarr, filterhistmed, footprint=footprint, mode='constant', cval=np.nan,
                      extra_keywords= {'amin': amin, 'amax':amax, 'bins':bins, 'medbin':medbin})

'''
from skimage.restoration import inpaint
def inpaintxr(xra):
    xrb=xra.copy()
    mask = np.isnan(xra.values)*1
    xrb.values = inpaint.inpaint_biharmonic(xra.values, mask)
    return xrb
'''

origfigsize=plt.rcParams['figure.figsize']
def plot2(A,B = None):
    if type(B) == type(None):
        numpl = 2
    else:
        numpl = 3
    plt.rcParams["figure.figsize"] = [int(6*numpl),4]
    plt.subplot(1,numpl,1)
    A.rename('px').plot()
    plt.subplot(1,numpl,2)
    A.plot.hist(bins=20)
    plt.axvline(A.median().values, color='black')
    #B.rename('rad').plot()
    if numpl > 2:
        plt.subplot(1,numpl,3)
        #AA.toremove
        B.plot()
    plt.show()
    plt.rcParams['figure.figsize']=origfigsize
    return


# older filter, working but.. median..
def medianfilter_array(arr, ws = 32):
    """use dask median filter on array
    works with both xarray and numpy array
    """
    chunksize = (ws*8, ws*8)
    if type(arr)==type(xr.DataArray()):
        inn = arr.values
    else:
        inn = arr
    arrb = da.from_array(inn, chunks=chunksize)
    arrfilt=ndfilters.median_filter(arrb, size=(ws,ws), mode='reflect').compute()
    if type(arr)==type(xr.DataArray()):
        out = arr.copy()
        out.values = arrfilt
    else:
        out = arrfilt
    return out

def apply_median_filter(data_array, filter_window_size):
    """
    Apply median filtering to an xarray.DataArray.

    Parameters:
        data_array (xarray.DataArray): The input data array.
        filter_window_size (int): The size of the median filter window.

    Returns:
        xarray.DataArray: The filtered data array.
    """
    # Convert the DataArray to a NumPy array
    data_array_np = data_array.values

    # Apply median filtering using scipy's median_filter function
    filtered_data_np = median_filter(data_array_np, size=filter_window_size)

    # Create a new DataArray with the filtered data
    filtered_data_xr = xr.DataArray(filtered_data_np, coords=data_array.coords, dims=data_array.dims)

    return filtered_data_xr


# very gross initial filter
def init_filter(prevest, ml=20):
    tak=prevest.copy()
    prevest = prevest.coarsen({prevest.dims[0]:ml,prevest.dims[1]:ml}, boundary='trim').median()
    prevest = medianfilter_array(prevest, ws = 8)
    #prevest.values = interpolate_nans(prevest,'linear')
    prevest.values = interpolate_nans(prevest,'nearest') # for pixels outside of interpolation
    prevest = medianfilter_array(prevest, ws = 8)
    #xr.DataArray(azinn).plot()
    #prevest = prevest.interp_like(tak, method='linear')
    prevest = prevest.interp_like(tak, method='nearest')
    prevest.values = interpolate_nans(prevest,'nearest')
    prevest = medianfilter_array(prevest, ws = 16)
    return prevest


def filterhist_value(block, amin, amax, bins=20, medbin=True):
    #histc, histe = np.histogram(block,range=(amin,amax), bins=bins)
    #histmax=np.argmax(histc)
    #minval=histe[histmax]
    #maxval=histe[histmax+1]
    #
    #if medbin:
    #    outval = float(blockxr.where(blockxr>minval).where(blockxr<maxval).median())
    #else:
    #    blockxr.values=interpolate_nans(blockxr.where(blockxr>minval).where(blockxr<maxval).values, method='nearest')
    #    outval = float(blockxr.sel(a=blockxr.shape[0]/2,r=blockxr.shape[1]/2,method='nearest'))
    #
    if np.isnan(block).all():
        return np.nan
    try:
        # dask may change window size to very small ones. for this, we will skip the disk
        block=get_disk_ones(block)*block
    except:
        pass
    histc, histe = np.histogram(block,range=(amin,amax), bins=bins)
    histmax=np.argmax(histc)
    minval=histe[histmax]
    maxval=histe[histmax+1]
    #
    if medbin:
        bb=block[block>minval]
        bb=bb[bb<maxval]
        outval = np.nanmedian(bb)
    else:
        blockxr=xr.DataArray(block)
        try:
            blockxr.values=interpolate_nans(blockxr.where(blockxr>minval).where(blockxr<maxval).values, method='linear')
            blockxr.values=interpolate_nans(blockxr.values, method='nearest') # just to be sure..
            outval = float(blockxr.values[int(block.shape[0]/2),int(block.shape[1]/2)])
        except:
            outval = np.nan
    return outval

import time

_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Elapsed time: {}h:{}m:{}s'.format(t_hour, t_min, t_sec))


def calculate_gradient(xar, deramp=False):
    """Calculates gradient of continuous data (not tested for phase)

    Args:
        xar (xr.DataArray): e.g. ifg['unw']
        deramp (bool): if True, it will remove overall ramp

    Returns:
        xr.DataArray
    """
    gradis = xar.copy()
    vgrad = np.gradient(gradis.values)
    gradis.values = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2)
    if deramp:
        gradis = deramp_unw(gradis)
    return gradis






'''
The functions are defined above. The process starts from here.
'''
## Define the topdir which you run the code, it shoul be frame folder.
subdir = os.getcwd()

# Create the directory using subprocess which the off result from licsar_offset_tracking.py. You need to run the licsar_offset_tracking.py to produce the offset result.

try:
    subprocess.run(['mkdir', '-p', os.path.join(subdir, 'OFF')])
except subprocess.CalledProcessError as e:
    print("Error creating directory:", e)

# Check for command-line argument
if len(sys.argv) < 2:
    print('USAGE: provide pair, and keep being in the frame folder')
    sys.exit()

date_pair = sys.argv[1]  # Get the command-line argument (date pair)

# Check if the argument is a valid date pair
if len(date_pair) != 17 or date_pair[8] != '_':
    print('Invalid date pair format')
    sys.exit()

# Create subfolder with the date pair name inside OFF directory
pair_path = os.path.join(subdir, 'OFF', date_pair)
try:
    os.makedirs(pair_path, exist_ok=True)
except OSError as e:
    print("Error creating subfolder:", e)

# Copy files from source to destination
IFG_path = os.path.join(subdir, 'IFG', date_pair)
try:
    for filename in os.listdir(IFG_path):
        if filename.startswith('tracking'):
            shutil.copy(os.path.join(IFG_path, filename), os.path.join(pair_path, filename))
	    #print("The file is copied successfully", filename)
except OSError as e:
    print("Error copying files:", e)

print("The OFF folder established and offset result is copied succesfully from IFG folder.")


## inputs:
offset=os.path.join(pair_path,'tracking.offsets') # or just remove the '128x128'
off_par=os.path.join(pair_path,'tracking.off') 
# outputs:
outcpxfile=os.path.join(pair_path,'tracking.offsets.filtered')
outlutfile=os.path.join(pair_path,'offsets.filtered.lut')
outlutfilefull=os.path.join(pair_path,'offsets.filtered.lut.full')

#print(offs)
#print(outcpxfile)
#print(outlutfile)


print('The offset parameters has been extracting')

with open(off_par, "r") as off:
    for line in off:
        # Strip leading and trailing whitespace from the line
        stripped_line = line.strip()
        # Split the stripped line into a list of words
        line_list = stripped_line.split()

        # Check for specific phrases and extract relevant values
        if 'offset_estimation_range_samples:' in line_list:
            lenr = np.int64(line_list[1])
        elif 'offset_estimation_azimuth_samples:' in line_list:
            lena = np.int64(line_list[1])
        #elif 'interferogram_range_pixel_spacing:' in line_list:
        #    rngres = np.float64(line_list[1])
        #elif 'interferogram_azimuth_pixel_spacing:' in line_list:
        #    azires = np.float64(line_list[1])
        elif 'offset_estimation_range_spacing:' in line_list:
            mlrng = np.float64(line_list[1])
        elif 'offset_estimation_azimuth_spacing:' in line_list:
            mlazi = np.float64(line_list[1])
        
            ###for full_resampling
        elif 'interferogram_azimuth_lines:' in line_list:
            outlenazi = np.int64(line_list[1])
        elif 'interferogram_width:' in line_list:
            outlenrng = np.int64(line_list[1])

thresm=5.5
rngres=2.329562
azires=14.00


### Reading the offset which is the complexdata
offs = np.fromfile(offset, dtype=np.complex64).byteswap().reshape((lena,lenr))
print('tracking.offset has been readed. The thresholding is started before filtering...')

rng=xr.DataArray(np.real(offs))
azi=xr.DataArray(np.imag(offs))

rng = xr.DataArray(
    data=rng.values,
    dims=["a", "r"],
    coords={"a": rng.dim_0.values, "r": rng.dim_1.values},
)
azi = xr.DataArray(
    data=azi.values,
    dims=["a", "r"],
    coords={"a": azi.dim_0.values, "r": azi.dim_1.values},
)

print('1, Zero Filtering ')
rng=rng.where(rng!=0)
azi=azi.where(azi!=0)

print('2,thresholding of azimuth and range to remove outlier. ')
rng=rng.where(np.abs(rng)<thresm/rngres)
azi=azi.where(np.abs(azi)<thresm/azires)
# but also those cross-errors:
rng=rng.where(np.abs(azi)<thresm/azires)
azi=azi.where(np.abs(rng)<thresm/rngres)

print("3, remove_islands function doesn't work in terminal ask Milan!")
#rng.values=remove_islands(rng.values, pixelsno = 25)
#azi.values=remove_islands(azi.values, pixelsno = 25)

print('Thresholding has just finished. The filtering step is going to start.')

'''
print('filtering azi')
tic()
azifilt = filter_histmed_ndarray(azi, winsize=128, bins=10)
medres = (azi-azifilt).copy()
medres=medres.fillna(0)
medres=medianfilter_array(medres, ws=64)
outazi=azifilt+medres
tac()

print('filtering rng')
tic()
rngfilt = filter_histmed_ndarray(rng, winsize=64, bins=10)
medres = (rng-rngfilt).copy()
medres=medres.fillna(0)
medres=medianfilter_array(medres, ws=32)
outrng=rngfilt+medres
tac()
'''
print('Histogram filtering takes time;therefore. simple median filtering is running as a test. IF the code doesnt give error please apply histogram filtering (but it takes time like 12h).')

azi_filt_64=apply_median_filter(azi, 64)
rng_filt_32 = apply_median_filter(rng, 32)
outazi=azi_filt_64
outrng=rng_filt_32

# STORING DATA
outcpx = outrng.fillna(0).values + 1j* outazi.fillna(0).values
outcpx.astype('complex64').tofile(outcpxfile)

print('Generating LUT...')

# first, export the LUT as is (multilooked)
if np.max(np.isnan(outrng)):
    method = 'nearest'
    outrng.values = interpolate_nans(outrng.values, method=method)


if np.max(np.isnan(outazi)):
    method = 'nearest'
    outazi.values = interpolate_nans(outazi.values, method=method)


##adding the pixel numbers themselves:
rnglut = outrng/mlrng + np.tile(outrng.r.values, (len(outrng.a.values),1))
azilut = outazi/mlazi + np.tile(outazi.a.values, (len(outazi.r.values),1)).T

print('Storing LUT data')
outlut = rnglut.values + 1j* azilut.values
#outlutfile='outlutfile'
outlut.astype('complex64').byteswap().tofile(outlutfile)


print('LUT is stored... The rubber_sheeting ready to start')



# store also the range offsets to be used as prevest
#rnginmm = fullrng.values*rngres*1000
#mm2rad_s1(rnginmm).astype('float32').tofile('rngoffsets_prevest_LE')





