# -*- coding: utf-8 -*-

from scipy.optimize import newton, brentq
import numpy as np

import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('RootFinder')

import plotly.offline as py  # for debugging
import plotly.graph_objs as go


class RootFinder:
    """
    Look for a zero of a function, using either the Newton-Raphson
    method or Brent's method. The required arguments to the constructor
    are:
        f           a function taking a single independent variable
                    and a list or tuple of additional arguments/parameters
        args        a list or tuple of additional arguments
        description a string describing the root to find (for debugging)
        xseeds      an optional list of values of the independent variable
        yseeds      an optional list of corresponding values of the
                    function evaluated at the values in xseeds. If not
                    provided, the values will be calculated by calls to
                    f.
        kwargs      optional keyword arguments:
            method      either 'newton' or 'brent'
            maxiter     (defaults to 50)
            lowerbound  optional limit on the range used to look for a root
            upperbound  optional limit on the range used to look for a root


    After configuring a RootFinder, use the __call__() method to
    search for the root. The numerical result is returned, but additional
    information is stored in the .result field of the RootFinder object,
    including the number of calls to evaluate the function, whether a root
    was found, etc., as described in the documentation of
    scipy.optimize

    """

    def __init__(self, f, args, description, xseeds=[], yseeds=[], **kwargs):
        self.function = f
        self.args = args
        self.description = description
        self.x = []     # list of x values for which we have corresponding ys
        self.y = []     # said corresponding y values
        self.atol = 1e-8  # absolute tolerance
        self.rtol = 1e-5  # relative tolerance
        self.a = None   # smaller of 2 bracketing x values
        self.b = None   # larger of 2 bracketing x values
        self.method = 'brent'
        self.maxiter = 50
        self.lowerbound = None
        self.upperbound = None
        self.set_x(xseeds, yseeds)
        self.handle_kwargs(**kwargs)

    def __str__(self):
        self.show()
        return self.description

    def handle_kwargs(self, **kwargs):
        for key, val in kwargs.items():
            if key in ('method', 'maxiter', 'lowerbound', 'upperbound',
                       'tol', ):
                setattr(self, key, val)

    @staticmethod
    def tolist(x):
        if isinstance(x, (float, int)):
            return [x]
        if isinstance(x, (list, tuple)):
            return list(x)
        try:
            y = x[0]
            return x
        except:
            raise ValueError(f'What type is {x}? {type(x)}')

    @staticmethod
    def straddle(a, b):
        return (a < 0 and b > 0) or (a > 0 and b < 0)

    def set_x(self, xseeds, yseeds):
        if xseeds is not None:
            self.x = self.tolist(xseeds)
            if yseeds is not None:
                self.y = self.tolist(yseeds)
            if len(self.x) != len(self.y):
                self.y = []
                # Compute only as many y values as needed to straddle
                for x in self.x:
                    self.y.append(self.function(x, *self.args))
                    if len(self.y) > 1 and self.straddle(self.y[-1], self.y[-2]):
                        self.x = self.x[:len(self.y)]
                        break

    def append(self, x):
        y = self.function(x, *self.args)
        self.x.append(x)
        self.y.append(y)
        logger.debug(f'appended ({self.x}) and ({self.y}')

    def bound(self):
        pairs = sorted(zip(self.y, self.x))
        # reset x and y to sorted order
        self.y = [p[0] for p in pairs]
        self.x = [p[1] for p in pairs]
        if pairs[0][0] > 0:
            raise ValueError(f'Minimum f = {pairs[0][0]} is positive')
        if pairs[-1][0] < 0:
            raise ValueError(f'Maximum f = {pairs[-1][0]} is negative')
        for n in range(1, len(pairs)):
            if pairs[n][0] > 0:
                self.a = min(pairs[n][1], pairs[n - 1][1])
                self.b = max(pairs[n][1], pairs[n - 1][1])
                return
        raise RuntimeError('could not bound??')

    def find_bounds(self):
        """
        Look for a pair of x values that straddle a root, expanding
        the range of values until a bracket is found.
        """
        if len(self.x) == 1:
            x0 = self.x[0]
            self.append(0.999 * x0)
            self.append(1.001 * x0)
        n = 0
        dx = None
        while n < self.maxiter:
            n += 1
            try:
                self.bound()
                return
            except ValueError:
                # Either all the y values are negative or positive
                if self.y[0] < 0:
                    dx = self.x[-1] - self.x[-2]
                    dy = self.y[-1] - self.y[-2]  # > 0
                    x0, stepsToZero = self.x[-1], -self.y[-1] / dy
                else:
                    dx = self.x[0] - self.x[1]
                    dy = self.y[0] - self.y[1]
                    x0, stepsToZero = self.x[0], -self.y[0] / dy
                if abs(dx) < 1e-4:
                    dx = 1e-4 * (1 if dx > 0 else -1)
                if stepsToZero > 8:
                    if n > 5:
                        dx *= stepsToZero
                    else:
                        dx *= 2
                x = x0 + dx
                if self.upperbound and dx > 0 and x > self.upperbound:
                    logger.debug(f'{x:.3f} exceeds upperbound of {self.upperbound:.3f}')
                    x = self.upperbound
                elif self.lowerbound and dx < 0 and x < self.lowerbound:
                    logger.debug(f'{x:.3f} exceeds lowerbound of {self.lowerbound:.3f}')
                    x = self.lowerbound
                self.append(x)
        logger.debug(f'Catastrophic failure in find_bounds for {self}')
        logger.debug('\n'.join([f'{x:.8f}\t{y:.8f}' for (x, y) in zip(self.x, self.y)]))

    def show(self):
        if len(self.x) > 0:
            py.plot({'data': [go.Scatter(x=self.x, y=self.y, mode='markers')],
                     'layout': {'width': 800, 'height': 600, }})

    def __call__(self):
        n, result = 3, None
        while n > 0 and result == None:
            n -= 1
            try:
                if self.method == 'newton':
                    result = self.newton()
                elif self.method == 'brent':
                    result = self.brent()
            except (RuntimeError, TypeError) as e:
                self.method = 'brent' if self.method == 'newton' else 'newton'
        # logger.debug(f'Found root {result:.5g} for {self.description}')
        # logger.debug('+' * 40)
        return result

    def newton(self):
        if len(self.x) == 1:
            x0 = self.x[0]
        else:
            n = np.argmin(np.abs(self.y))
            x0 = self.x[n]
        # logger.debug('-' * 40)
        # logger.debug(f'Newton on {self.description} starting at {x0:.3f}')
        root, self.results = newton(
            self.function,
            x0,
            args=self.args,
            tol=self.atol,
            maxiter=self.maxiter,
            full_output=True
        )
        return root

    def brent(self):
        self.find_bounds()
        #logger.debug('*' * 80)
        # logger.debug(f'Brent on {self.description} with')
        # logger.debug(f'({self.a:.3f}, {self.b:.3f})')
        root, self.results = brentq(
            self.function,
            self.a, self.b,
            args=self.args,
            xtol=self.atol,
            rtol=self.rtol,
            maxiter=self.maxiter,
            full_output=True)
        return root
