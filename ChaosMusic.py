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

    # These are frequencies of the Bb overtone series from Bb1
    targets = [58.270, 87.307, 116.54, 146.83, 174.61]
    n_basisPoints = len(targets)
    verbose = True
    # Get base parameters for image;
    # These might be overwritten
    name,basisPoints,rawWeight,moveFrac = GetParameters(n_basisPoints, verbose)
    
    if ideal:
        randoms = None
        filter_seq = False
    else:
        # Record sound and analyze somehow
        # Look at peak:
        #peak, data = SA.get_peaks(targets[0], window=10, duration_seconds=12, rate=44100)
    
        # Look at strength/power of certain frequencies;
        # Make these the raw weights of the basis points in the chaos game
        # The notes appearing more strongly will be chosen more often in the iteration
        strengths, data = SA.get_relative_strengths(targets, duration_seconds=12, rate=44100)
        rawWeight = strengths
        filter_seq = False   

        # Transform raw audio data with Normal Cumulative Distribution Function
        # Data sort of looks normal; 
        # the distribution of CDF values should look uniform
        trans_data = stats.norm.cdf(data, scale=np.std(data))
        # subsample to reduce covariance, but results in too few points
#        randoms = trans_data[::10]
        randoms = None
        if verbose and randoms is not None:
            print("Randoms: Min={}, Max={}".format(min(randoms), max(randoms)))
            fig, ax = plt.subplots(1,2)
            ax[0].plot(randoms)
            ax[0].set_xlabel('Samples')
            ax[1].hist(randoms,bins=25)
            ax[1].set_xlabel('Random sample histogram')
            plt.show()

    # Generate image file
    GenerateImage(name, basisPoints, rawWeight, moveFrac, randoms=randoms, 
            filter_seq=filter_seq, use_fortran=USE_FORTRAN, verbose=verbose)
    return


def GetParameters(n_basisPoints=5, verbose=True):
    """ 
    Get parameters for chaos game
    """
    assert n_basisPoints <= 5 and n_basisPoints > 0, "Number of basis points needs to be in (0,5]"

    timestring = '{}'.format(datetime.now()).split('.')[0].replace(' ','_')
    name ='fractal_{}'.format(timestring)

    # Set basis points
    basisPoints = []
    for k in range(n_basisPoints):
        # go around the circle of radius 0.5 in polar coordinates
        angle = k*2*np.pi/n_basisPoints
        basisPoints.append([0.5*np.cos(angle), 0.5*np.sin(angle)])
    # shift the center to (0.5, 0.5)
    basisPoints = np.array([0.5, 0.5]) + basisPoints

    # Set weighting for each basis point in chaos game
    # TODO:
    # These relative weights/frequencies could reflect the relative weight of some
    # component of the music or sound that is analyzed
    beats = [2,3,5,7,11]
    rawWeight = np.array([1.0/b for b in beats[:n_basisPoints]])

    # Move fractions
    # TODO: What could move fraction be?
    # Maybe this provides some dynamics-
    # all the beats speed up in proportion somehow
    # kind of like how a higher move fraction bunches points up near a certain basis point
    moveFrac = 0.5*np.ones(n_basisPoints)

    if verbose:
        print("Basis points:\n{}".format(basisPoints))
        print("(ideal) raw weights: {}".format(rawWeight))
        print("(ideal) move fractions: {}".format(moveFrac))

    return  name, basisPoints, rawWeight, moveFrac 


def GenerateImage(name, basisPoints, rawWeight, moveFrac, 
        n_Iterations=5e5, n_grid=1000, randoms=None, filter_seq=True, 
        use_fortran=True, verbose=True):
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
    filter_seq : (boolean) Filter sequence of basis points ... somehow 
    use_fortran : (boolean) Use fractal_loop module with fast compiled code
    verbose : (boolean) Display what's going on
    """
    # should probably check that basisPoints are in unit square
    
    n_Iterations = int(n_Iterations)
    # Calculate cumulative probability distribution based on raw weights
    prob = rawWeight/np.sum(rawWeight)
    cumulProb = np.array([ np.sum(prob[0:i+1]) for i in range(len(prob)) ])

    if verbose:
        print("\nProbabilities (relative frequencies): {}".format(prob))
        print("Using ideal randoms: {}".format((randoms is None)))
        print("Filtering sequence of points: {}".format(filter_seq))
        print("Number iterations: {}".format(n_Iterations if randoms is None else len(randoms)))
        print("Starting chaos game iteration")

    # matrix that will hold density/produce image
    density = np.zeros((n_grid,n_grid))

    # Sequence of basis points to visit in iteration;
    # Ideally, points are chosen according to their probabilities in prob
    # (by generating a uniform random and using inverse CDF transform)
    # However, if randoms is not None, they might not be independent and uniform
    # and this may impose different results
    if randoms is None:
        basis_sequence_full = np.random.choice(np.arange(len(prob)), p=prob, size=n_Iterations)
    else:
        basis_sequence_map = map(lambda p: np.argmax(p < cumulProb), randoms)
        basis_sequence_full = np.fromiter(basis_sequence_map, dtype=np.int)
    # "Filter" the sequence:
    # this can create some cool structure
    if filter_seq:
        basis_sequence = [basis_sequence_full[i] for i in range(1,len(basis_sequence_full)) 
                        if basis_sequence_full[i] != basis_sequence_full[i-1] ]
    else:
        basis_sequence = basis_sequence_full

    if use_fortran:
        # Call compiled Fortran to do loop real fast
        density = floop.get_density(n_grid, basis_sequence, moveFrac, basisPoints)
    else:
        # Initial point: can be any point in unit hypercube,
        # but first basis point works fine
        point = basisPoints[0]
        # Generate more points in loop
        for i in basis_sequence:
            # Index i corresponds to a basis point chosen "at random"
            # (although that depends on the statistics of the sequence <randoms>)

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

