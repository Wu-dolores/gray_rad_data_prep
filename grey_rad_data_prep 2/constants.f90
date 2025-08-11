module constants_mod

implicit none
private

!=================physcial constants
real(8), public, parameter :: RADIUS = 6371.e3
real(8), public, parameter :: OMEGA  = 7.292e-5
real(8), public, parameter :: GRAV   = 9.80
!=================dry air constants
real(8), public, parameter :: RDGAS  = 287.00421071138067 !phys.air.R
real(8), public, parameter :: KAPPA  = 0.2858607676408174 !phys.air.Rcp
real(8), public, parameter :: CP_AIR = 1004.0 !phys.air.cp
!=================ocean constants
real(8), public, parameter :: CP_OCEAN = 3989.24495292815
real(8), public, parameter :: RHO0 = 1.035e3
real(8), public, parameter :: RHO0R  = 1./RHO0
real(8), public, parameter :: RHO_CP = RHO0 * CP_OCEAN    
!=================water vapor constants
real(8), public, parameter :: RVGAS     = 461.9173324615943 !phys.water.R
real(8), public, parameter :: CP_VAPOR  = 1847.0 !phys.water.cp
real(8), public, parameter :: DENS_H2O  = 1000.
real(8), public, parameter :: HLV       = 2493000.0 !phys.water.L_vaporization_TriplePoint
real(8), public, parameter :: HLF       = 3.34e5
real(8), public, parameter :: HLS       = hlv + hlf
real(8), public, parameter :: TFREEZE   = 273.16

real(8), public, parameter :: cv_vapor = cp_vapor - rvgas
real(8), public, parameter :: cv_air = cp_air - rdgas

!##=================radiation constants
real(8), public, parameter :: wtmair  = 2.896440e1
real(8), public, parameter :: wtmh2o  = wtmair * (rdgas / rvgas)
!##wtmo3   = 47.99820e1 #wrong
real(8), public, parameter :: diffac  = 1.660000
real(8), public, parameter :: seconds_per_day = 86400.
real(8), public, parameter :: avogno  = 6.023000e23
!##pstd    = 1.013250e6 #wrong
real(8), public, parameter :: pstd_mks= 101325.

!##radcon  = ((1.e2*grav)/(1.e4*cp_air))*seconds_per_day
real(8), public, parameter :: radcon_mks  = grav / cp_air * seconds_per_day
real(8), public, parameter :: o2mixrat    = 2.0953e-1
real(8), public, parameter :: rhoair      = 1.292269
real(8), public, parameter :: alogmin     = -50.

!##=================miscellaneous constants
real(8), public, parameter :: stefan  = 5.6734e-8
real(8), public, parameter :: vonkarm = 0.4
real(8), public, parameter :: pi      = 3.14159265358979323846 !math.pi
real(8), public, parameter :: rad_to_deg = 180./pi
real(8), public, parameter :: deg_to_rad = pi/180.
real(8), public, parameter :: radian  = rad_to_deg
real(8), public, parameter :: c2dbars = 1.e-4
real(8), public, parameter :: kelvin  = 273.15
real(8), public, parameter :: epsln   = 1.e-40


!##================================latent heat

real(8), public, parameter :: cp_water = 4187. !#phys.water.cp #4187.

public :: latent_heat

contains

subroutine latent_heat(T, L)

real(8), intent(in) :: T
real(8), intent(out) :: L

L = hlv + (cp_vapor - cp_water)*(T - 273.15) ! phys.water.TriplePointT)

end subroutine latent_heat

!def latent_heat(T):
!    return hlv + (cp_vapor - cp_water)*(T - phys.water.TriplePointT)

end module constants_mod
