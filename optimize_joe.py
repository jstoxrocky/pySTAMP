from __future__ import division, print_function, absolute_import
import warnings
import numpy

from scipy.lib.six import callable
from numpy import atleast_1d, eye, mgrid, argmin, zeros, shape, \
     squeeze, vectorize, asarray, sqrt, Inf, asfarray, isinf
from linesearch_joe import \
     line_search_wolfe1, line_search_wolfe2, \
     line_search_wolfe2 as line_search

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev' : 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}

class Result(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess : ndarray
        Values of objective function, Jacobian and Hessian (if available).
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

class OptimizeWarning(UserWarning):
    pass

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

_epsilon = sqrt(numpy.finfo(float).eps)

#this gets the norm of the gradient vector
def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(numpy.abs(x))
    elif ord == -Inf:
        return numpy.amin(numpy.abs(x))
    else:
        return numpy.sum(numpy.abs(x)**ord, axis=0)**(1.0 / ord)




#so wrap function simply creates and returns 2 objects
# ncalls: a list with 1 element
# and a function: function wrapper
def wrap_function(function, args):
    
    ncalls = [0]

    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
        
    return ncalls, function_wrapper





def approx_fprime(xk, f, epsilon, *args):
    """Finite-difference approximation of the gradient of a scalar function.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        The function of which to determine the gradient (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a scalar, the value of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives.  If an array, should contain one value per element of
        `xk`.
    \*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    grad : ndarray
        The partial derivatives of `f` to `xk`.

    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    The main use of `approx_fprime` is in scalar function optimizers like
    `fmin_bfgs`, to determine numerically the Jacobian of a function.

    Examples
    --------
    >>> from scipy import optimize
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(np.float).eps)
    >>> optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004198])

    """
    
    f0 = f(*((xk,) + args))
    grad = numpy.zeros((len(xk),), float)
    ei = numpy.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k] = (f(*((xk+d,)+args)) - f0) / d[k]
        ei[k] = 0.0

    return grad


#this is the first thing that gets called
def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False,
                   **unknown_options):
                   
    """
    Options for the BFGS algorithm are:
        disp : bool
            Set to True to print convergence messages.
        maxiter : int
            Maximum number of iterations to perform.
        gtol : float
            Gradient norm must be less than `gtol` before successful
            termination.
        norm : float
            Order of norm (Inf is max, -Inf is min).
        eps : float or ndarray
            If `jac` is approximated, use this value for the step size.

    This function is called by the `minimize` function with `method=BFGS`.
    It is not supposed to be called directly.
    """
    
    #if there are any arguments sent to the minimize function
    #that don't make any sense, the _check_unknown_options functions
    #finds them and prints a warning
    _check_unknown_options(unknown_options)
    
    #f is the function (kalman)
    f = fun
    
    #fprime is the jacobian (default None)
    fprime = jac
    
    #by default eps is set to eps where it is the smallest possible number such that
    #1.0 + eps != 1.0 
    epsilon = eps
    
    #********NOT SURE WHAT THIS IS YET*************
    #set to false by default
    retall = return_all

    #change x0 from list to array if not already a numpy array
    x0 = asarray(x0).flatten()
    
    
    #if number of dimensions is zero (?) shape so that its 1.
    #don't quite get but leave anyways
    if x0.ndim == 0:
        x0.shape = (1,)
    
    #set max interation to 200 times number of parameters if not set by user
    if maxiter is None:
        maxiter = len(x0)*200
        
    #send the function (kalman) and any args (flag) to wrap function
    #so now f is actually function_wrapper
    #func calls is the count that goes up eacht time function_wrapper is called
    func_calls, f = wrap_function(f, args)
    
    #fprime is jac and we set it to its default None
    #approx_fprime is is the gradient, appoximated because we don't send in any 
    #jacobian
    #again...wrapping...
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)  
    #myfprime is now the approx_fprime function wrapped
    
    #gfk is the approximated gradient
    gfk = myfprime(x0)
    
    k = 0
    N = len(x0)
    
    #create idenitity matrix with demensions equal to number of parameters
    I = numpy.eye(N, dtype=int)
    Hk = I
    
    #get old LL
    old_fval = f(x0)
    #create an older one + 5000 (since minimizing)
    old_old_fval = old_fval + 5000
    
    #rename x0 to xk
    xk = x0
    
    #retall is false by default
    if retall:
        allvecs = [x0]
    
    #gtol is our criteria. The norm (abs) must be less than this for termination
    #sk is a  list containing 2 times the gtol value
    sk = [2*gtol]
    warnflag = 0
    
    #gnorm is the norm (absolute value) of the gradient vector
    #must be a scalar
    gnorm = vecnorm(gfk, ord=norm)
    

    #this is the actual optimization loop
    #while the norm of the gradient is less than our criteria (gtol)
    #or before we hit maxiter
    #keep looping

    while (gnorm > gtol) and (k < maxiter):


        #Hk is an identity matrix at first
        #gfk is the un-normed gradient
        
        #what is pk (Hk*gfk)
        #thisis the DIRECTION!
        pk = -numpy.dot(Hk, gfk)
        
        #do a line search LINE SEARCH WOLFE1 first
        #***************
        #inputs:
        #f: wrapped objective function
        #myfprime: wrapped fapprox_gradient function
        #xk: parameters
        #pk: direction: dot product of Hk and gfk (Hk is initially an identity matrix) 
        #gfk: approximated gradient
        #***************
        #ouputs:
        alpha_k, fc, gc, old_fval2, old_old_fval2, gfkp1 = \
           line_search_wolfe1(f, myfprime, xk, pk, gfk,
                              old_fval, old_old_fval)       

        if alpha_k is not None:
            old_fval = old_fval2
            old_old_fval = old_old_fval2
        else:
            #do a line search LINE SEARCH WOLFE2 second
            # line search failed: try different one.
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search_wolfe2(f, myfprime, xk, pk, gfk,
                                        old_fval, old_old_fval)
            #joe
            break
            
            if alpha_k is None:
                # This line search also failed to find a better solution.
                warnflag = 2
                break
        
        ###Joey Edit###
        xkp1 = xk + alpha_k * pk
        #xkp1 = xk + float(alpha_k) * pk

        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)
            

        yk = gfkp1 - gfk
        gfk = gfkp1

        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  #this was handled in numeric, let it remaines for more safety
            ###Joey Edit###
            #sk = numpy.squeeze(numpy.asarray(sk))

            rhok = 1.0 / (numpy.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  #this is patch for numpy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rhok
        A2 = I - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + rhok * sk[:, numpy.newaxis] \
                * sk[numpy.newaxis, :]

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    result = Result(fun=fval, jac=gfk, hess=Hk, nfev=func_calls[0],
                    njev=grad_calls[0], status=warnflag,
                    success=(warnflag == 0), message=msg, x=xk)
    if retall:
        result['allvecs'] = allvecs
    return result
