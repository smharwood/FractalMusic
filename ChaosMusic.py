"""
12 January 2020
SM Harwood

Chaos Game inspired by music
"v3": triggering from sound
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import sound_analyzer as SA
import fractalloop as floop
#import matplotlib
#matplotlib.use('agg')


def main(verbose=True):

    n_basisPoints = 5
    fundamental = 55
    harmonics = 'const'
    ideal = True

    # Get parameters for music and image
    name,basisNotes,basisBeatInterval,basisPoints,rawWeight,moveFrac = \
        GetParameters(n_basisPoints, fundamental, harmonics, verbose)
    
    if ideal:
        randoms = None
    else:
        # record sound and find peak frequency near some target
        target = 440
        window = 10
        peak, data = SA.get_peak(near=target, within=window, duration_seconds=10)
        min_dat = min(data)
        max_dat = max(data)
        randoms = (data - min_dat)/(max_dat - min_dat)
        if verbose:
            print("Randoms: Min={}, Max={}".format(min(randoms), max(randoms)))
            plt.plot(data)
            plt.show()
            plt.hist(data,bins=25)
            plt.show()
            #TODO: inverse CDF of normal transform

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

    # Generate image file
    GenerateImage(name, basisPoints, rawWeight, moveFrac, randoms=randoms, verbose=verbose)
    return


def GetParameters(n_basisPoints=5, fundamental=55, harmonics='const', verbose=True):
    """ 
    get parameters for chaos game
    """
    # Options
    # how many basis points
    # fundamental (e.g. 55=G3, 60=C4)
    # harmonics (const, Maj7, Dom7, mMaj7, Min7)
    assert n_basisPoints <= 5 and n_basisPoints > 0, "Number of basis points needs to be in (0,5]"

    name ='fractal_{}-{}-{}'.format(n_basisPoints,fundamental,harmonics)

    # Harmonics between the basis notes
    constant   = np.array([0,0,0,0,0])
    majSeventh = np.array([0,4,7,11,0])
    domSeventh = np.array([0,4,7,10,0])
    mMajSeventh= np.array([0,3,7,11,0])
    minSeventh = np.array([0,3,7,10,0])
    if harmonics == 'const':
        diffNotes = fundamental + constant
    elif harmonics == 'Maj7':
        diffNotes = fundamental + majSeventh
    elif harmonics == 'Dom7':
        diffNotes = fundamental + domSeventh
    elif harmonics == 'mMaj7':
        diffNotes = fundamental + mMajSeventh
    elif harmonics == 'Min7':
        diffNotes = fundamental + minSeventh
    # Set basis notes
    basisNotes = diffNotes[0:n_basisPoints]

    # TODO: modify
    # Beats between notes/events:
    # PRIME multiples of a sixteenth note
    # (beat = quarter,
    #  sixteenth = quarter of a beat)
    beats = [2,3,5,7,11]
    sixteenth = 0.25
    basisBeatInterval = sixteenth*np.array(beats[0:n_basisPoints])

    # Set weighting for each point in chaos game
    # These weights/relative frequencies should reflect how often they occur in the music
    rawWeight = np.array([1.0/b for b in basisBeatInterval])

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

    return  name, basisNotes, basisBeatInterval, basisPoints, rawWeight, moveFrac 


def GenerateImage(name, basisPoints, rawWeight, moveFrac, 
        n_Iterations=5e5, n_grid=1000, randoms=None, verbose=True):
    """
    Play the "Chaos Game" to generate an image (png) file
    """
    # basisPoints: n x 2 array, the "basis points" of the game
    # rawWeight: length n array, the raw weights for choosing the basis points
    # moveFrac: length n array, the fraction to move to each basis point
    # n_Iterations controls how many points are generated
    # n_grid controls the resolution of the plot
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

    if True:
        basis_indices = map(lambda p: np.argmax(p < cumulProb), randoms)
        b_indices = np.array(list(basis_indices))
        # Call fortran to do loop real fast
        density = floop.get_density(n_grid, b_indices, moveFrac, basisPoints)
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
    main()

