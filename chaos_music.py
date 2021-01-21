"""
12 January 2020
SM Harwood

Chaos Game inspired by music
"v3": triggering from sound
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    import fractal_loop as floop
    _USABLE_FORTRAN = True
except ImportError:
    _USABLE_FORTRAN = False
#    # OK, try building it
#    import subprocess
#    stat = subprocess.call("python -m numpy.f2py -c fractal_loop.f90 -m fractal_loop".split()) 
#    if stat:
#        _USABLE_FORTRAN = False
#    else:
#        import fractal_loop as floop
#        _USABLE_FORTRAN = True
print("Usable Fortran: {}".format(_USABLE_FORTRAN))


def main(args):
    """
    Get base parameters of the image generation,
    and optionally listen to sound to seed some other parameters
    """
    import scipy.stats as stats
    import sound_analyzer as SA
    from datetime import datetime

    # Bb2 F3 Bb3 F4 Bb4
    targets = [116.54, 174.61, 233.08, 349.23, 466.16]
    # These continue the overtone series:
    #     ~116.54*N:      10      11      12       20      30
    redundancies = [1174.659, 1318.5, 1396.9, 2349.318, 3520.0]
    # Bb2 Bb3 Bb4 F4 A4
    #targets = [116.5409, 233.0819, 349.2282, 466.1638, 440.0]
    # Frequencies of the Bb overtone series from Bb1:
    #targets = [58.270, 87.307, 116.54, 146.83, 174.61]
    n_basis_pts = len(targets)
    verbose = True
    n_slices=3
    # For testing, do NOT analyze sound (more reproducible)
    if args: ideal = True
    else:    ideal = False

    # Get base parameters for image;
    # These might be overwritten
    timestring = datetime.now().isoformat('_','seconds')
    name ='fractal_{}'.format(timestring)
    basis_pts, raw_wts, mfs = GetParameters(n_basis_pts, verbose=verbose)
    
    if ideal:
        randoms = None
        filter_seq = False
    else:
        # Record sound and analyze somehow

        # Look at strength/power of certain frequencies;
        # Make these the raw weights of the basis points in the chaos game
        # The notes appearing more strongly will be chosen more often in the iteration
        strengths, data = SA.get_relative_strengths(targets, redundancies,
            duration_seconds=0.1, rate=44100, name=name)
#            duration_seconds=4, rate=12000, name=name)
        raw_wts = strengths
        randoms = None
        filter_seq = False   
        """
        # Look at peak:
        peak, data = SA.get_peaks(targets[0], window=10, duration_seconds=12, rate=44100)
        # Transform raw audio data with Normal Cumulative Distribution Function
        # Data sort of looks normal; 
        # the distribution of CDF values should look uniform
        # Subsample to reduce covariance? but results in too few points
        trans_data = stats.norm.cdf(data, scale=np.std(data))
        randoms = trans_data[::10]
        if verbose and randoms is not None:
            print("Randoms: Min={}, Max={}".format(min(randoms), max(randoms)))
            fig, ax = plt.subplots(1,2)
            ax[0].plot(randoms)
            ax[0].set_xlabel('Samples')
            ax[1].hist(randoms,bins=25)
            ax[1].set_xlabel('Random sample histogram')
            plt.show()
        """

    # Generate fractal data
    density = GenerateImage(basis_pts, raw_wts, mfs, randoms=randoms, 
            filter_seq=filter_seq, verbose=verbose)

    # Plot
    if verbose: print("Creating fractal image")
    plotter(density, name, invert=True)
    #plotter_simple_invert(density, name+'-invert', n_slices=n_slices)
    return


def GetParameters(
        n_basis_pts=5,
        wts=None,
        mfs=None,
        twist=None,
        seed=None,
        verbose=True):
    """ 
    Get parameters for chaos game
    """
    # Set random seed if desired
    if seed is not None:
        np.random.seed(seed)

    # Random twist (up to 60 degrees)
    if twist is None:
        twist = np.pi*np.random.rand()/3

    # Take number of basis points from weights or move fractions
    if wts is not None and mfs is not None:
        n_basis_pts = min(len(wts), len(mfs))
    elif wts is not None:
        n_basis_pts = len(wts)
    elif mfs is not None:
        n_basis_pts = len(mfs)

    # Set basis points
    basis_pts = []
    for k in range(n_basis_pts):
        angle = twist + k*2*np.pi/n_basis_pts
        r = 0.25 + 0.25*np.random.random()
        basis_pts.append([r*np.cos(angle), r*np.sin(angle)])
    # shift the center to (0.5, 0.5)
    basis_pts = np.array([0.5, 0.5]) + basis_pts

    # Ideal (?) interpretations:
    # Move fractions:
    #   These parameters make me think of the instantaneous,
    #   or maybe overall/integrated, prevalence of a point.
    #   This could correspond to the time-independent
    #   (that is, frequency-domain) prevalence or strength of a musical element
    # Basis point weights:
    #   Thinking about the iteration that happens in the chaos game, the
    #   weights seem to reflect a dynamic or temporal prevalence of the point.
    #
    # Practical note:
    # Move fractions need to be somewhat comparable in size
    # Weights can differ more

    # move fractions
    # use random if None, otherwise scale values to [0,1] (or so)
    if mfs is None:
        #mfs = 0.6*np.ones(n_basis_pts)
        mfs = 0.25 + 0.5*np.random.rand(n_basis_pts)
    else:
        mfs = np.asarray(mfs[:n_basis_pts])
        mfs = 0.5 + 0.5*(mfs - np.mean(mfs))/(np.max(mfs) - np.min(mfs) + 1e-6)
    
    # weights
    # use random if None, otherwise just cast as numpy array
    if wts is None:
        #wts = np.ones(n_basis_pts)
        wts = np.random.rand(n_basis_pts)
    else:
        wts = np.asarray(wts[:n_basis_pts])

    if verbose:
        print("Basis points:\n{}".format(basis_pts))
        print("(default) move fractions: {}".format(mfs))
        print("(default) raw weights: {}".format(wts))

    return basis_pts, wts, mfs


def GenerateImage(basis_pts, raw_wts, move_fracs, 
        n_Iterations=5e5, n_grid=1000, randoms=None, filter_seq=False, 
        use_fortran=True, verbose=True):
    """
    Play the "Chaos Game" to generate an image (png) file

    Args:
    basis_pts : (array, shape (n,2)) The "basis points" of the game
    raw_wts : (array, shape (n,)) The raw weights for choosing the basis points
    move_fracs : (array, shape (n,)) The fraction to move to each basis point
    n_Iterations : (int) How many points are generated (ignored of randoms is not None)
    n_grid : (int) How many grid point to plot (roughly, the resolution)
    randoms : (array) Optional pre-computed random numbers to use for chaos game iteration
    filter_seq : (boolean) Filter sequence of basis points ... somehow 
    use_fortran : (boolean) Use fractal_loop module with fast compiled code (if possible)
    verbose : (boolean) Display what's going on
    """
    # should probably check that basis_pts are in unit square
    
    n_Iterations = int(n_Iterations)
    # Calculate cumulative probability distribution based on raw weights
    prob = np.asarray(raw_wts)/np.sum(raw_wts)
    cumulProb = np.array([ np.sum(prob[0:i+1]) for i in range(len(prob)) ])

    if verbose:
        print("\nProbabilities (relative frequencies): {}".format(prob))
        print("Move fractions (relative strengths): {}".format(move_fracs))
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

    if use_fortran and _USABLE_FORTRAN:
        # Call compiled Fortran to do loop real fast
        density = floop.get_density(n_grid, basis_sequence, move_fracs, basis_pts)
    else:
        # Initial point: can be any point in unit hypercube,
        # but first basis point works fine
        point = basis_pts[0]
        # Generate more points in loop
        for count, i in enumerate(basis_sequence):
            # Index i corresponds to a basis point chosen "at random"
            # (although that depends on the statistics of the sequence <randoms>)

            # Calculate point:
            #   some fraction of distance between previous point and basis point i
            point = (1-move_fracs[i])*point + move_fracs[i]*basis_pts[i]
        
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

    return density


def plotter_simple_invert(density, name, n_slices=1):
    """ 
    Plot with imshow (simple and consistent) 
    Also allow "slicing" image into layers 

    See also plotter for some of the motivation
    """
    # map thru log
    log_density = np.log(density + 1e-6)

    DPI = 100
    fig_dim = (2000.0)/DPI
    vmax = np.max(log_density)
    vmin = -0.5*vmax
    background = vmin*np.ones(density.shape)

    # Sort nonzeros, and figure out (n_slices)-iles of values
    logvals = np.log(density[np.nonzero(density)].flatten()+1e-6)
    logvals.sort()
    slice_vals = [logvals[(i+1)*len(logvals)//n_slices-1] for i in range(n_slices)]
    slice_vals.insert(0,0)

    for i in range(n_slices):
        condition = np.logical_and(log_density > slice_vals[i], 
                                   log_density <=slice_vals[i+1])
        slice_i = np.where(condition, log_density, background)
        plt.figure(figsize=(fig_dim,fig_dim),dpi=DPI)
        plt.imshow(slice_i, 
            vmin=vmin,
            vmax=vmax,
            cmap='Greys_r',
            origin='lower',
            interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(name+'-{}'.format(i), facecolor='black', 
            bbox_inches='tight', pad_inches=fig_dim/6)
        plt.close()
    return


def plotter(density, name, invert=False):
    """ Plot in a more custom way """

    # assume square
    n_grid = density.shape[0]

    # sparse version of density matrix is useful
    (rows,cols) = np.nonzero(density)
    vals = density[(rows,cols)]

    # Plotting parameters
    # All this is to get a particular look with a scatterplot
    # DPI and fig_dim don't matter much;
    #   they are defined so we always get a 2000x2000 pixel image
    # But there is an interaction between DPI and marker size
    #   and the right marker size is required to avoid aliasing
    # DPI : dots per inch
    # fig_dim : figure dimension in inches
    # markershape : Shape of marker in scatterplot. 's' = square
    # markersize : Marker size in square points
    # alphaval : Alpha channel value for markers
    DPI = 100
    fig_dim = 2000.0/DPI
    markershape = 's'
    markersize = (3.1*72.0/DPI)**2 # 3.1 pixels wide?
    alphaval = 1.0
    if invert:
        transparent = True
        colormap = 'Greys_r'
    else:
        transparent = False
        #facecolor = 'white'
        colormap = 'Greys'

    # Map density values thru log
    # (log of density seems to produce more interesting image)
    # order the points so that higher values are plotted last and on top
    logvals = np.log(vals)
    ordering = np.argsort(logvals)
    rows = rows[ordering]
    cols = cols[ordering]
    logvals = logvals[ordering]

    # Use only about 2/3 of the colormap's range
    # Take advantage of fact that minimum value of logvals is ~zero
    # (log(min nonzero) = log(1) = 0)
#    image_frac = len(vals)/float(n_grid**2)
#    inverse_scaling = max(0, 0.70 - image_frac)
    vmax = logvals[-1]
    vmin = -0.5*vmax

    fig = plt.figure(figsize=(fig_dim,fig_dim), dpi=DPI)
    plt.scatter(cols, rows, c=logvals, 
        s=markersize, 
        marker=markershape, 
        linewidths=0, 
        cmap=colormap, 
        vmin=vmin,
        vmax=vmax, 
        alpha=alphaval)
    plt.axis([0,n_grid,0,n_grid])
    plt.axis('equal')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name, transparent=transparent,
        bbox_inches='tight', pad_inches=fig_dim/6)
    plt.close()
    return


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

