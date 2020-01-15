! Compile with
! python -m numpy.f2py -c FractalLoop.f95 -m fractalloop

subroutine get_density(n_iter, n_basis, n_grid, randomIndices, moveFrac, basisPoints, density)
! =====================================================
! Implement the core iteration of the Chaos Game
! =====================================================
    integer, intent(in)   :: n_iter
    integer, intent(in)   :: n_basis
    integer, intent(in)   :: n_grid
    real, intent(in) :: randomIndices(n_iter)
    real, intent(in) :: moveFrac(n_basis)
    real, intent(in) :: basisPoints(n_basis,2)
    integer, intent(out)  :: density(n_grid,n_grid)

    real :: point(2)
    integer :: i,k,x,y

    ! Initialize first point of iteration
    point(1) = basisPoints(1,1)
    point(2) = basisPoints(1,2)
    ! Iterate
    do k = 1, n_iter
        ! get index of basis point, add one because these are Python-indexed
        i = randomIndices(k)+1
        ! update coordinates of current point
        point(1) = (1-moveFrac(i))*point(1) + moveFrac(i)*basisPoints(i,1)
        point(2) = (1-moveFrac(i))*point(2) + moveFrac(i)*basisPoints(i,2)
        ! what are the cooresponding (x,y) coordinates/indices in density array
        ! components of point should be in [0,1]
        ! so just make sure value is integer in [1,n_grid]
        x = max(1,int(point(1)*n_grid))
        y = max(1,int(point(2)*n_grid))
        ! increment in density
        ! this will be imaged like imshow, so y is row index, x is col index
        density(y,x) = density(y,x) + 1
    end do
    return
end subroutine
