module radiation_mod

use constants_mod, only : grav, stefan, pi

implicit none
private

public :: radiation_down, radiation_up, radiation_init, lw_tau

! real(8) :: linear_tau      = 0.6199
real(8) :: wv_tau          = 0. !1.
real(8) :: atm_abs         = 0. !#0.22
! real(8) :: sw_atten        = 0.0
real(8) :: wv_exponent     = 4.
real(8) :: lw_exponent     = 2.
real(8) :: solar_exponent  = 2.
real(8) :: reference_slp   = 1.e5

real(8) :: solar, lw_tau_0, b_surf
real(8) :: olr, albedo, column_water_vapor
real(8), allocatable, dimension(:) :: b!, tdt_rad, tdt_sw
real(8), allocatable, dimension(:) :: up, down, net, solar_down!, flux_rad, flux_sw
real(8), allocatable, dimension(:) :: dtrans, lw_tau, solar_tau
real(8), allocatable, dimension(:) :: dp_half

contains

subroutine radiation_init(num_levels)

integer, intent(in) :: num_levels

allocate(b(num_levels))
!allocate(tdt_rad(num_levels))
!allocate(tdt_sw(num_levels))

allocate(up(num_levels+1))
allocate(down(num_levels+1))
!allocate(net(num_levels+1))
allocate(solar_down(num_levels+1))
!allocate(flux_rad(num_levels+1))
!allocate(flux_sw(num_levels+1))

allocate(dtrans(num_levels))
allocate(lw_tau(num_levels+1))
allocate(solar_tau(num_levels+1))
allocate(dp_half(num_levels))

end subroutine radiation_init


subroutine radiation_down(p_half, q, t, solar_constant, sw_atten, window, linear_tau, &
                 down)

real(8), intent(in) :: solar_constant, sw_atten, window, linear_tau !lat, lon not used
real(8), intent(in), dimension(:) :: p_half, q, t
real(8), intent(out),dimension(:) :: down
! real(8), dimension(:) :: solar_down
integer :: n, i

     
    n = size(t,1)

    solar = solar_constant
    dp_half = p_half(2:n+1) - p_half(1:n)

    lw_tau(1)   = 0.

    do i = 1,n 
       lw_tau(i+1) = wv_tau /grav/10. * sum(q(1:i) * dp_half(1:i))
    end do
    lw_tau = lw_tau + linear_tau * (p_half /p_half(n+1))**lw_exponent ![-1] #reference_slp
    ! lw_tau = lw_tau * linear_tau / lw_tau(n+1)
    ! solar_tau   = solar_tau_0 * (p_half / reference_slp)**solar_exponent
    solar_tau = lw_tau*sw_atten

    b = (1.-window) * stefan * t*t*t*t

    dtrans = exp(-(lw_tau(2:n+1) - lw_tau(1:n)))
    
    !down = p_half *1.
    down(1) = 0.
    do i = 1, n
    !for i in range(len(down)-1):
       down(i+1) = down(i) * dtrans(i) + b(i) * (1 - dtrans(i))
    end do
    ! solar_down = solar * exp(-solar_tau)
    down = down + solar * exp(-solar_tau)


end subroutine radiation_down
!    return surf_lw_down, net_surf_sw_down, down, solar_down


subroutine radiation_up(p_half, t_surf, window, t, q, up)

real(8), intent(in), dimension(:) :: p_half
real(8), intent(in) :: t_surf, window
real(8), intent(in), dimension(:) :: t, q
! real(8), intent(out) :: olr
real(8), intent(out),dimension(:) :: up

integer :: n, i

n = size(t)

    !dp_half = p_half(2:n+1) - p_half(1:n) ![:-1]
    !column_water_vapor = sum(q * dp_half) / grav
    !lw_tau_0    = wv_tau * column_water_vapor /10.
    !lw_tau      = p_half*1.
    !lw_tau[0]   = 0.
    !for i in range(len(dp_half)):
    !    lw_tau[i+1] = wv_tau /cons.grav/10. * sum(q[:i+1] * dp_half[:i+1])
    !lw_tau = lw_tau + linear_tau * p_half /p_half[-1] #reference_slp
    !b = (1.-window) * cons.stefan * t*t*t*t
    !dtrans = np.exp(-(lw_tau[1:] - lw_tau[:-1]))

!#============================================================
    b_surf = stefan *t_surf*t_surf *t_surf *t_surf

   ! up = p_half*1.
    up(n+1) = b_surf * (1.-window)

    !for i in var1:
    do i = n, 1, -1
       up(i) = up(i+1) * dtrans(i) + b(i)*(1-dtrans(i))
    end do
    up = up + b_surf * window

    olr = up(1) ![0]

end subroutine radiation_up
    !return olr, up
    

end module radiation_mod
