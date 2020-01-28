"""
12 January 2020
SM Harwood

Chaos Game inspired by music
"v3": triggering from sound
"""
import sys
from datetime import datetime
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sound_analyzer as SA
try:
    import fractal_loop as floop
    USE_FORTRAN = True
except ImportError:
    # OK, try building it
    import subprocess
    stat = subprocess.call("python -m numpy.f2py -c FractalLoop.f90 -m fractal_loop".split()) 
    if stat:
        USE_FORTRAN = False
    else:
        import fractal_loop as floop
        USE_FORTRAN = True
#import matplotlib
#matplotlib.use('agg')


def main(args):
    """
    Get base parameters of the image generation,
    and optionally listen to sound to seed some other parameters
    """
    if args:
        # Do NOT analyze sound - e.g. for testing
        ideal = True
    else:
        ideal = False

    # Get parameters for image
    n_basisPoints = 5
    verbose = True
    name,basisPoints,rawWeight,moveFrac = GetParameters(n_basisPoints, verbose)
    
    if ideal:
        randoms = None
    else:
        # record sound and find peak frequency near some target
        target = 440
        window = 10
        peak, data = SA.get_peak(target,window, duration_seconds=5, rate=6000)
        # Transform with Normal Cumulative Distribution Function
        # Data sort of looks normal; 
        # the distribution of CDF values should look uniform
        randoms = stats.norm.cdf(data, scale=np.std(data))
        if verbose:
            print("Randoms: Min={}, Max={}".format(min(randoms), max(randoms)))
            fig, ax = plt.subplots(1,2)
            ax[0].plot(randoms)
            ax[0].set_xlabel('Samples')
            ax[1].hist(randoms,bins=25)
            ax[1].set_xlabel('Random sample histogram')
            plt.show()
        """
        # Positive or negative fraction measuring how far off "ideal" we are;
        # perturb moveFrac based on sound analysis
        # Idea is to change it more for certain basis points
        # TODO: generalize??
        delta_fraction = (peak-target)/(2*window)
        moveFrac[-1] += delta_fraction
        name += '-{:5.3f}'.format(delta_fraction)
        name += '-RAND'
        if verbose:
            print("Peak frequency: {}".format(peak))
            print("Perturbation: {}".format(delta_fraction))
        """

    # Generate image file
    GenerateImage(name, basisPoints, rawWeight, moveFrac, randoms=randoms, 
            use_fortran=USE_FORTRAN, verbose=verbose)
    return


def GetParameters(n_basisPoints=5, verbose=True):
    """ 
    Get parameters for chaos game
    """
    assert n_basisPoints <= 5 and n_basisPoints > 0, "Number of basis points needs to be in (0,5]"

    timestring = '{}'.format(datetime.now()).split('.')[0].replace(' ','_')
    name ='fractal_{}'.format(timestring)

    # Set weighting for each basis point in chaos game
    # TODO:
    # These relative weights/frequencies could reflect the relative weight of some
    # component of the music or sound that is analyzed
    beats = [2,3,5,7,11]
    rawWeight = np.array([1.0/b for b in beats[:n_basisPoints]])

    # Set basis points
    basisPoints = []
    for k in range(n_basisPoints):
        # go around the circle of radius 0.5 in polar coordinates
        angle = k*2*np.pi/n_basisPoints
        basisPoints.append([0.5*np.cos(angle), 0.5*np.sin(angle)])
    # shift the center to (0.5, 0.5)
    basisPoints = np.array([0.5, 0.5]) + basisPoints

    # Move fractions
    # TODO: What could move fraction be?
    # Maybe this provides some dynamics-
    # all the beats speed up in proportion somehow
    # kind of like how a higher move fraction bunches points up near a certain basis point
    moveFrac = 0.5*np.ones(n_basisPoints)

    if verbose:
        print("Basis points:\n{}".format(basisPoints))
        print("(raw) weights: {}".format(rawWeight))
        print("(ideal) Move fractions: {}".format(moveFrac))

    return  name, basisPoints, rawWeight, moveFrac 


def GenerateImage(name, basisPoints, rawWeight, moveFrac, 
        n_Iterations=5e5, n_grid=1000, randoms=None, use_fortran=True, verbose=True):
    """
    Play the "Chaos Game" to generate an image (png) file

    Args:
    name : (string) Base name of image file to be produced
    basisPoints : (array, shape (n,2)) The "basis points" of the game
    rawWeight : (array, shape (n,)) The raw weights for choosing the basis points
    moveFrac : (array, shape (n,)) The fraction to move to each basis point
    n_Iterations : (int) How many points are generated (ignored of randoms is not None)
    n_grid : (int) How many grid point to plot (roughly, the resolution)
    randoms : (array) Optional pre-computed random numbers to use for chaos game iteration
    use_fortran : (boolean) Use fractal_loop module with fast compiled code
    verbose : (boolean) Display what's going on
    """
    # should probably check that basisPoints are in unit square

    # Calculate cumulative probability distribution based on raw weights
    prob = rawWeight/np.sum(rawWeight)
    cumulProb = np.array([ np.sum(prob[0:i+1]) for i in range(len(prob)) ])

    if verbose:
        print("Probabilities (relative frequencies): {}".format(prob))
        print("Starting chaos game iteration")

    # matrix that will hold density/produce image
    density = np.zeros((n_grid,n_grid))

    # random numbers:
    # use uniform random or perhaps something else
    if randoms is None:
        randoms = np.random.random(int(n_Iterations))

    if use_fortran:
        basis_indices_map = map(lambda p: np.argmax(p < cumulProb), randoms)
        basis_indices = np.fromiter(basis_indices_map, dtype=np.int)
        # Call compiled Fortran to do loop real fast
        density = floop.get_density(n_grid, basis_indices, moveFrac, basisPoints)
    else:
        # Initial point: can be any point in unit hypercube,
        # but first basis point works fine
        point = basisPoints[0]
        # Generate more points in loop
        for k in range(len(randoms)):
            # Pick a basis point according to the weighting:
            #   generate random number uniformly in [0,1)
            #   find first index/basis point with cumulative probability
            #   greater than random uniform
            p = randoms[k]
            i = np.argmax(p < cumulProb)

            # Calculate point:
            #   some fraction of distance between previous point and basis point i
            point = (1-moveFrac[i])*point + moveFrac[i]*basisPoints[i]
        
            # Copy to density matrix;
            # increment number of points occurring in the grid point of interest
            # Since the components of the points are all in the range [0,1],
            # floor the product of the point coordinate with the number of grid points
            # (and make sure that it's in the index range)
            x_coor = int(min(n_grid-1, np.floor(n_grid*point[0])))
            y_coor = int(min(n_grid-1, np.floor(n_grid*point[1])))
            density[y_coor][x_coor] += 1
        # end k loop

    if verbose:
        print("Done iterating")

    # Plot with imshow (simple and consistent)
    DPI = 300
    fig_dim = (1000.0)/DPI
    log_density = np.log(density + 1e-6)
    plt.figure(figsize=(fig_dim,fig_dim),dpi=DPI)
    plt.imshow(log_density, cmap='Greys',origin='lower',interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(name+'.png', bbox_inches='tight', pad_inches=fig_dim/6)
    plt.close()
    return


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

