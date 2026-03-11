module lorenz04

implicit none
private

!> required routines with code in this module
public :: static_init_model, &
          adv_1step, &
          comp_dt


integer    :: model_size        = 960
real(8)    :: forcing           = 15.00
real(8)    :: delta_t           = 0.001
real(8)    :: space_time_scale  = 10.00
real(8)    :: coupling          = 3.008
integer     :: K                 = 32
integer     :: smooth_steps      = 12
integer     :: time_step_days    = 0
integer     :: time_step_seconds = 3600

!---------------------------------------------------------------- 
! Define some parameters for computational efficiency
integer  :: H
integer  :: K2
integer  :: K4
integer  :: ss2
real(8) :: sts2

!---------------------------------------------------------------- 
! Define the averaging function for the production of x from z (L2k4)
real(8), allocatable :: a(:)

contains

subroutine static_init_model()

real(8) :: x_loc
real(8) :: ri
real(8) :: alpha, beta
integer  :: i, j

! if K is even, H = K/2 and the first/last summation 
! terms are divided by 2.  if K is odd, H = (K-1)/2 and 
! the first and last terms are taken as-is.  this code
! only implements the algorithm for even K so test for it.
if (int((K+1)/2) /= int(K/2)) then
   print *, 'Model only handles even values of K'
   stop
endif

! Generate the alpha and beta parameters for the calculation of "a"
alpha = (3.0*(smooth_steps**2) + 3.0) &
      / (2.0*(smooth_steps**3) + 4.0*smooth_steps)
beta  = (2.0*(smooth_steps**2) + 1.0) &
      / (1.0*(smooth_steps**4) + 2.0*(smooth_steps**2))

! The "a" vector is a smoothing filter for the production of x and y from z
! in L2k4. Apologies for the "ri" and "j" construct
allocate(a(2*smooth_steps + 1))
ri = - smooth_steps - 1.0
j = 0
do i = - smooth_steps, smooth_steps
   j = j + 1
   ri = ri + 1.0
   a(j) = alpha - beta*abs(ri)
end do

! defining parameters to help reduce the number of operations in the calculation
! of dz/dt
H    = K/2
K2   = 2*K
K4   = 4*K
ss2  = 2*smooth_steps
sts2 = space_time_scale**2   

end subroutine static_init_model

!------------------------------------------------------------------
!> Computes the time tendency of the lorenz 2004 model given current state.
!>
!> The model equations are given by
!> 
!> Model 2 (II)
!>      dX_i
!>      ---- = [X,X]_{K,i} -  X_i + F 
!>       dt                
!>                         
!>
!> Model 3 (III)
!>      dZ_i
!>      ---- = [X,X]_{K,i} + b^2 (-Y_{i-2}Y_{i-1} + Y_{i-1}Y_{i+1})
!>       dt                +  c  (-Y_{i-2}X_{i-1} + Y_{i-1}X_{i+1})
!>                         -  X_i - b Y_i + F,
!>
!> where
!>
!>     [X,X]_{K,i} = -W_{i-2K}W_{i-K} 
!>                 +  sumprime_{j=-(K/2)}^{K/2} W_{i-K+j}X_{i+K+j}/K,
!>
!>      W_i =  sumprime_{j=-(K/2)}^{K/2} X_{i-j}/K,
!>
!> and sumprime denotes a special kind of summation where the first
!> and last terms are divided by 2.
!>
!> NOTE: The equations above are only valid for K even.  If K is odd,
!> then sumprime is replaced by the traditional sum, and the K/2 limits
!> of summation are replaced by (K-1)/2. THIS CODE ONLY IMPLEMENTS THE
!> K EVEN SOLUTION!!!
!>
!> The variable that is integrated is X (model II) or Z (model III), 
!> but the integration of Z requires
!> the variables X and Y.  For model III they are obtained by
!>
!>      X_i = sumprime_{j= -J}^{J} a_j Z_{i+j}
!>      Y_i = Z_i - X_i.
!>
!> The "a" coefficients are given by
!>
!>      a_j = alpha - beta |j|,
!> 
!> where
!>
!>      alpha = (3J^2 + 3)/(2J^3 + 4J)
!>      beta  = (2J^2 + 1)/(1J^4 + 2J^2).
!>
!> This choice of alpha and beta ensures that X_i will equal Z_i
!> when Z_i varies quadratically over the interval 2J.   This choice
!> of alpha and beta means that sumprime a_j = 1 and 
!> sumprime (j^2) a_j = 0.
!>
!> Note that the impact of this filtering is to put large-scale
!> variations into the X variable, and small-scale variations into
!> the Y variable.
!> 
!> The parameter names above are based on those that appear in
!> Lorenz 04.  To map to the code below, set:
!>
!>       F = forcing 
!>       b = space_time_scale
!>       c = coupling
!>       K = K
!>       J = smooth_steps
subroutine comp_dt(z, dt) 

real(8), intent( in)        ::  z(:)
real(8), intent(out)        :: dt(:)
real(8), dimension(size(z)) :: x, y
real(8)                     :: xwrap(- K4:model_size + K4)
real(8)                     :: ywrap(- K4:model_size + K4)
real(8)                     ::    wx(- K4:model_size + K4)
real(8)                     :: xx
integer                      :: i, j


call z2xy(z,x,y)

! Deal with cyclic boundary conditions using buffers
do i = 1, model_size
   xwrap(i) = x(i)
   ywrap(i) = y(i)
end do

! Fill the xwrap and ywrap buffers
do i = 1, K4
   xwrap(- K4 + i)       = xwrap(model_size - K4 +i)
   xwrap(model_size + i) = xwrap(i)
   ywrap(- K4 + i)       = ywrap(model_size - K4 +i)
   ywrap(model_size + i) = ywrap(i)
end do

! Calculate the W's
do i = 1, model_size
   wx(i) = xwrap(i - (-H))/2.00
   do j = - H + 1, H - 1
      wx(i) = wx(i) + xwrap(i - j)
   end do
   wx(i) = wx(i) + xwrap(i - H)/2.00
   wx(i) = wx(i)/K
end do

! Fill the W buffers
do i = 1, K4
   wx(- K4 + i)       = wx(model_size - K4 + i)
   wx(model_size + i) = wx(i)
end do

! Generate dz/dt
do i = 1, model_size
   xx = wx(i - K + (-H))*xwrap(i + K + (-H))/2.00
   do j = - H + 1, H - 1
      xx = xx + wx(i - K + j)*xwrap(i + K + j)
   end do
   xx = xx + wx(i - K + H)*xwrap(i + K + H)/2.00
   xx = - wx(i - K2)*wx(i - K) + xx/K
      
   dt(i) = xx + (sts2)*( - ywrap(i - 2)*ywrap(i - 1) &
       + ywrap(i - 1)*ywrap(i + 1)) + coupling*( - ywrap(i - 2)*xwrap(i - 1) &
       + ywrap(i - 1)*xwrap(i + 1)) - xwrap(i) - space_time_scale*ywrap(i) &
       + forcing

end do

end subroutine comp_dt

!------------------------------------------------------------------
!> Decomposes z into x and y for L2k4

subroutine z2xy(z,x,y)

integer :: i, j, ia
real(8), intent( in) :: z(:)
real(8), intent(out) :: x(:)
real(8), intent(out) :: y(:)
real(8)              :: zwrap(- ss2:model_size + ss2)

! Fill zwrap
do i = 1, model_size
   zwrap(i) = z(i)
end do
zwrap( - ss2) = zwrap(model_size - ss2)
do i = 1, ss2
   zwrap( - ss2 + i) = zwrap(model_size - ss2 + i)
   zwrap(model_size + i) = zwrap(i)
end do

! Generate the x variables
do i = 1, model_size
   ia = 1
   x(i) = a(ia)*zwrap(i - ( - smooth_steps))/2.00
   do j = - smooth_steps + 1, smooth_steps - 1
      ia = ia + 1
      x(i) = x(i) + a(ia)*zwrap(i - j)
   end do
   ia = ia + 1
   x(i) = x(i) + a(ia)*zwrap(i - smooth_steps)/2.00
end do

! Generate the y variables
do i = 1, model_size
   y(i) = z(i) - x(i)
end do

end subroutine z2xy

!> Does single time step advance for lorenz 04 model
!> using four-step rk time step

subroutine adv_1step(x)

real(8), intent(inout) :: x(:)

real(8), dimension(size(x)) :: x1, x2, x3, x4, dx, inter
real(8), dimension(size(x)) :: dxt

call comp_dt(x, dx)    !  Compute the first intermediate step
x1    = delta_t * dx
inter = x + x1 / 2.0

call comp_dt(inter, dx)!  Compute the second intermediate step
x2    = delta_t * dx
inter = x + x2 / 2.0

call comp_dt(inter, dx)!  Compute the third intermediate step
x3    = delta_t * dx
inter = x + x3

call comp_dt(inter, dx)!  Compute fourth intermediate step
x4 = delta_t * dx

!  Compute new value for x

dxt = x1/6.0 + x2/3.0 + x3/3.0 + x4/6.0
x = x + dxt

end subroutine adv_1step

end module lorenz04
