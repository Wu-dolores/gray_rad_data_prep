program prep_data

use radiation_mod, only : radiation_init, radiation_down, radiation_up

implicit none

real(8) :: r, solar, sw_atten, window, tau0, t0, t1
real(8), allocatable :: p_half(:), q(:), t(:)
real(8), allocatable :: down(:), up(:)

integer :: num_levels, ncases, i, j
! Initialize the number of levels and cases
print *, "Enter the number of vertical levels:"
read *, num_levels
print *, "Enter the number of cases:"
read *, ncases

! Initialize radiation module
call radiation_init(num_levels)
! generate p_half, q, t for testing
allocate(p_half(num_levels + 1))
allocate(q(num_levels+1))
allocate(t(num_levels+1))
allocate(down(num_levels + 1))
allocate(up(num_levels + 1))
print *, "Enter the surface pressure (Pa):"
read *, p_half(num_levels + 1)

call cpu_time(t0)

open(1, file='data.txt', status='replace', action='write')
do j = 1, ncases
    write(1, *) "Case number:", j, 'Surface pressure (Pa):', p_half(num_levels + 1)
    write(1, '(4(A8))') 'solar_constant', 'sw_atten', 'window', 'tau0'

    ! Generate random values for q, t
    do i = 1, num_levels+1
        p_half(i) = p_half(num_levels + 1) * i / (num_levels+1)
        call random_number(r)
        q(i) = r * 0.01  ! Random water vapor mixing ratio
        call random_number(r)
        t(i) = 150 + r * 200.0  ! Random temperature between 150K and 350K
    end do
    
    ! Set constants for radiation calculations
    call random_number(r)
    solar = (r*0.5 + 0.5) * 1361.0  ! Random solar constant in W/m^2
    call random_number(r)
    sw_atten = r * 0.1  ! Random shortwave attenuation factor
    call random_number(r)
    window = r * 0.4  ! Random window factor
    call random_number(r)
    tau0 = 0.5 +  r * 5  ! Random linear optical depth
    write(1, *) solar, sw_atten, window, tau0
    write(1, '(5(A8))') 'p_half', 'q', 't', 'up', 'down'
    ! Call radiation_down subroutine
    call radiation_down(p_half, q(1:num_levels), t(1:num_levels), solar, sw_atten, window, tau0, down)
    call radiation_up(p_half, t(num_levels+1), window, t(1:num_levels), q(1:num_levels), up)
    ! Write results to file
    do i = 1, num_levels + 1
        write(1, *) p_half(i), q(i), t(i), up(i), down(i)
    end do
end do

call cpu_time(t1)
print *, "Time taken for data preparation: ", t1 - t0, "seconds"
print *, "Data preparation complete. Results written to data.txt"

end program prep_data