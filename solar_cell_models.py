# -*- coding: utf-8 -*-

"""
  Author:  Peter N. Saeta --<saeta@hmc.edu>
  Purpose: Modeling solar installations, often with odd geometries
  Created: 05/29/18

  Model a single solar cell.
  
  This code defines a number of classes (SolarCell, CellString,
  SolarPanel, PanelString) to model the behavior of a set of solar panels
  connected in series and tied to a device that attempts to operate at the
  maximum power point.

  The SolarCell class keeps track of placement information (size, position,
  area), cell temperature and voltage, but the actual behavior of the cell
  is governed by a model of the class SolarCellModel, which does the actual
  computation of I-V curves for given conditions of illumination and ambient
  temperature.

  Because there are typically many cells with similar behavior and conditions,
  dynamic programming is employed to minimize the recomputation of cell response.
  This is implemented by a VoltageCache, which must be passed into the
  constructor of a SolarCellModel. The VoltageCache maintains computed I-V curves
  using a common vector of current values, which is set via the set_currents
  method.

  A SolarCellModel
"""

import abc
import os
import numpy as np
from math import exp

import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt

from solar_config import SolarConfig
# import configparser  # to specify properties of cells


import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('SolarCellModels')


DEG = chr(0xb0)  # to avoid headaches with pylint
BOLTZMANN_CONSTANT = 8.6173303e-5  # eV/K
ZERO = 273.15  # °C

solar_config = SolarConfig(__file__.replace('.py', '.ini'))


def process_kwargs(obj, keys, **kwargs):
    "Update object with any values in kwargs matching a list of keys."
    for key in keys:
        if key in kwargs:
            setattr(obj, key, kwargs[key])


class SolarCellModel(metaclass=abc.ABCMeta):
    """
    SolarCellModel is an abstract base class that promises a function
    that will take a tuple (or list of tuples) of the form

    (current, insolation)

    and compute the voltage across the cell. All cell models must
    define the following fields:

        name:       a string giving the name of the model
        t_nominal:  the nominal temperature in °C, typically 25°C
        g_nominal:  standard insolation condition, typically 1000 W/m^2
        i_sc:       the short-circuit current of the cell under nominal conditions (A)
        v_oc:       the open-circuit voltage of the cell under nominal conditions (V)
        width:      the cell width (m)
        height:     the cell height (m)

    for a single solar
    cell
    to compute cell voltage for a shared
    array of current values and a given pair of (insolation, temperature).
    To implement dynamic programming, the model is not generally called
    by a client, but instead called by the VoltageCache, which returns a
    reference to voltage vector, fetching from cache if available, and
    computing it by calling SolarCellModel.voltage() if it doesn't. Subclasses
    must override this method.
    """

    _required_fields = (
        'name',       # name of the model
        't_nominal',  # temperature at which v_oc and i_sc are specified
        'g_nominal',  # standard illumination, typically 1 kW/m^2
        'i_sc',       # short-circuit current, under nominal illumination
        'v_oc',       # open-circuit voltage
        'width',      # width of cell, in meters
        'height',     # height of cell, in meters
        'temperature_coefficient',  # heating of cell, K/W
    )

    def __init__(self, key="DEFAULT", **kwargs):
        self.name = ''
        self.t_nominal = None
        self.g_nominal = None
        self.i_sc = None
        self.v_oc = None
        self.height = None
        self.width = None
        self.temperature_coefficient = None
        self._t_celsius = None
        # now process the keyword arguments,
        params = {**solar_config.section(key), **kwargs}
        process_kwargs(self, self._required_fields, **params)
        try:
            self.area = self.height * self.width
        except:
            raise ValueError('You must define a cell height and width')

    @abc.abstractmethod
    def voltage(self, ig_tuples):
        "Must be overridden by the subclass"
        pass

    @property
    def t_celsius(self):
        return _t_celsius

    @t_celsius.setter
    def t_celsius(self, TC):
        "Typically overridden"
        self._t_celsius = t_celsius

    def __call__(self, ig_tuple):
        return self.voltage((gt_tuple[0], gt_tuple[1]))

    def mpp(self, g_value, t_value):
        """
        Find the maximum power point for this cell at the given
        level of insolation (W/m^2) and temperature (°C).
        """
        self.t_celsius = t_value
        i_max = g_value / self.g_nominal * self.i_sc
        i_min = i_max / 2
        pts = 12
        # This is silly: we should have a better termination condition
        for n in range(10):
            ivals = np.linspace(i_min, i_max, pts)
            vvals = self.voltage([(i, g_value) for i in ivals])
            power = ivals * vvals
            nmax = np.argmax(power)
            i_min = ivals[nmax - 1 if nmax > 0 else 0]
            i_max = ivals[nmax + 1 if nmax < pts - 1 else pts - 1]
        return {
            'pmp': power[nmax],
            'imp': ivals[nmax],
            'vmp': vvals[nmax],
        }

    def iv_curve(self, g_value, t_value, v_min, nvals=1000):
        """
        Generate a single i-v curve; return dictionary with fields
        insolation, temperature, i, v, power, pmp, imp, vmp, efficiency
        """
        self.t_celsius = t_value
        ivals = np.linspace(0.0, 1.25 * self.i_sc * g_value / 1000, nvals)
        vvals = self.voltage([(i, g_value) for i in ivals])
        power = ivals * vvals
        nmax = np.argmax(power)
        pmp, imp, vmp = power[nmax], ivals[nmax], vvals[nmax]
        efficiency = pmp / (g_value * self.area)
        return {
            'insolation': g_value,
            'temperature': t_value,
            'i': ivals,
            'v': vvals,
            'power': power,
            'pmp': pmp,
            'imp': imp,
            'vmp': vmp,
            'efficiency': efficiency,
        }

    def iv_curves(self, Gvals, Tvals, nvals=1000, **kwargs):
        """Generate a plot of I-V curves for various inputs.
        possible keyword arguments:

            i_min
            i_max
            split  default is True; whether to use a different scale for reverse bias
            v_min
            v_max
            mpp    default is True, to display a dot at the MPP of the traces
        """
        from plotly import tools
        import pandas as pd
        try:
            gvals = [e for e in Gvals]
        except:
            gvals = [Gvals]
        try:
            tvals = [e for e in Tvals]
        except:
            tvals = [Tvals]

        minv, maxv = 10, -100

        # process keyword arguments
        i_min = kwargs.get('i_min', 0)
        i_max = kwargs.get('i_max', None)
        split = kwargs.get('split', True)
        v_min = kwargs.get('v_min', -25)
        v_max = kwargs.get('v_max', None)
        mpp = kwargs.get('mpp', True)

        if split:
            fig = tools.make_subplots(cols=2, shared_yaxes=True)
        else:
            data = []
        colors = [f'hsl({h},70%,50%)'
                  for h in np.linspace(0, 360, 1 + len(gvals) * len(tvals))]
        i = 0
        results = {
            'i': [],
            'v': [],
            'label': [],
        }
        for insolation in gvals:
            power_in = insolation * self.area
            for temp in tvals:
                self.t_celsius = temp
                iMax = i_max if i_max else 0.001 * insolation * self.i_sc * 1.3
                if iMax < 1:
                    iMax += (1 - v_min / self.r_shunt)
                ivals = np.linspace(i_min, iMax, nvals)
                vvals = self.voltage([(current, insolation)
                                      for current in ivals])
                power = ivals * vvals
                nmax = np.argmax(power)
                pmax, imax, vmax = power[nmax], ivals[nmax], vvals[nmax]
                efficiency = pmax / power_in

                lab = f'{insolation/1000:.2g} sun{"" if insolation == 1000 else "s"}, {temp:.0f}{DEG}C, {efficiency*100:.1f}%, ({vmax:.3f} V, {imax:.2f} A)'
                trace = go.Scatter(
                    x=vvals,
                    y=ivals,
                    marker={'color': colors[i], },
                    # showlegend = not split
                    name=lab
                )

                results['i'].append(ivals)
                results['v'].append(vvals)
                results['label'].append(lab)

                if mpp:
                    # add a MPP point
                    dot = go.Scatter(
                        x=[vmax],
                        y=[imax],
                        marker={'color': colors[i], 'size': 12},
                        showlegend=False,
                        mode='markers'
                    )

                if split:
                    # add the trace for forward bias
                    fig.append_trace(trace, 1, 2)
                    # add the trace for reverse bias
                    fig.append_trace(
                        go.Scatter(x=vvals, y=ivals,
                                   marker={'color': colors[i], },
                                   showlegend=False),
                        1, 1)
                    if mpp:
                        fig.append_trace(dot, 1, 2)
                else:
                    data.append(trace)
                    if mpp:
                        data.append(mpp)

                minv = min(minv, min(vvals))
                maxv = max(maxv, max(vvals))
                i += 1
        layout = {
            'title': f'I-V Curves for {self.name}',
            'yaxis': {'title': 'Current (A)', },
            'font': {'family': 'Georgia', 'size': 18, },
        }
        if split:
            layout['xaxis1'] = {
                'range': (minv, 0),
                'title': 'Voltage (reverse bias)',
                'domain': [0, 0.5],
            }
            layout['xaxis2'] = {
                'range': (0, 1.1 * maxv),
                'title': 'V',
                'domain': [0.5, 1],
            }
        else:
            layout['xaxis'] = {
                'title': 'Voltage (V)',
                'range': (minv, 1.1 * maxv),
            }
        if split:
            fig['layout'].update(**layout)
        else:
            fig = go.Figure(data=data, layout=layout)
        return fig, results  # py.plot(fig)


class StaticKimModel(SolarCellModel):
    """
    This class implements a static version of the model described by Katherine Kim
    et al. in IEEE J. Photovolt. 3 (2013) 1334. That is, capacitative effects are
    ignored. We implement Eqs. (1)--(13), excluding (6)--(10) which describe the
    capacitative effects. As shown in Fig. 6 of that paper, these effects should
    be negligible for frequencies below 1 kHz.

    See __init__() below for a list of the required parameters. These may be
    supplied using keyword arguments, but generally you should create a section
    in solar_cell_models.ini corresponding to the type of cell. An example:

    [KimMC]
    # the monocrystalline cell
    r_shunt = 91.8
    r_series = 0.046
    v_temperature_coefficient = -0.00224   # V/°C
    i_temperature_coefficient = 0.00495    # A/°C
    ideality = 1.31
    i_sc = 7.61 # A
    i_sr = 0.0997 # A
    v_oc = 0.624 # V
    t_nominal = 25  # °C
    k_reverse = 0.0114
    v_breakdown = -23.5 # V
    width = 0.156   # m
    height = 0.156  # m
    name = KimMC

    You would then create the model with

    my_model = StaticKimModel('KimMC')
    """

    def __init__(self, inikey='DEFAULT', **kwargs):
        # required parameters, to be taken from the .ini file
        self.r_shunt = None
        self.r_series = None
        self.v_temperature_coefficient = None  # V/°C
        self.i_temperature_coefficient = None  # A/°C
        self.i_sr = None  # A, reverse saturation current
        self.v_breakdown = None  # V
        self.k_reverse = None
        self.ideality = None  # forward diode ideality factor

        # local variables that are set by calling t_celsius()
        self.t_kelvin = ZERO
        self.kT = BOLTZMANN_CONSTANT * ZERO  # thermal energy, eV
        self.dI = 0  # temperature-induced shift in i_sc
        self.dV = 0  # temperature-induced shift in v_oc
        self.v_eff_thermal = 0  # effective thermal energy, including diode ideality
        self.i_sat = 0
        self.i_photo = 0
        self.i_pv = 0
        self.v_diode = 0

        # If a string has been passed for initialization, read values from
        # the solar_cell_models.ini configuration file; otherwise, look in
        # kwargs.
        super().__init__(inikey, **kwargs)
        params = {** solar_config.section(inikey), ** kwargs}

        # Prepare a list of fields to be used to report the status of a model
        self.fields = (
            ("g_nominal", "W/m^2"),
            ("t_nominal", "°C"),
            ("height", "m"),
            ("width", "m"),
            ("r_shunt", "Ω"),
            ("r_series", "Ω"),
            ("ideality", ""),
            ("i_sc", "A"),
            ("v_oc", "V"),
            ("v_breakdown", "V"),
            ("k_reverse", ""),
            ("i_temperature_coefficient", "A/K"),
            ("v_temperature_coefficient", "V/K"),
            ("i_sr", "A")
        )
        # Now look for overrides to the values we have used to initialize fields
        # that are supplied either in the config file or kwargs.
        process_kwargs(
            self,
            [x[0] for x in self.fields],
            **params)

        # i_reverse is the extra current we can get in reverse bias between short
        # circuit and breakdown
        self.i_reverse = -self.v_breakdown / self.r_shunt

        # Store a modest list of typical voltage values to help guide the root finding
        self.coarseV = list(self.v_oc * np.asarray([-20, -10, 0.5, 0.95, 1]))

        # Call set_temperature to initialize several fields
        self.t_celsius = self.t_nominal

    def __str__(self):
        width = 2 + len(max(self.fields, key=lambda x: len(x[0]))[0])
        fmt = f'{{0:{width}}}{{1:.4g}} {{2}}'
        fields = "\n".join([fmt.format(key[0], getattr(self, key[0]), key[1])
                            for key in self.fields])
        return f'{self.name}\n{fields}'

    @property
    def t_celsius(self):
        return self._t_celsius

    @t_celsius.setter
    def t_celsius(self, TC):
        r"""
        \begin{align}
          V_{\rm eff\ therm} &= a k_{\rm B} T \\
        \end{align}
        """
        if TC > 200:
            logger.info(f'Attempted to set model temperature to {TC}')
            TC = 200
        self._t_celsius = TC
        self.t_kelvin = TC + ZERO
        self.kT = self.t_kelvin * BOLTZMANN_CONSTANT  # eV
        self.v_eff_thermal = self.kT * self.ideality

        # dI is the temperature-induced shift in photocurrent
        # dV is the temperature-induced shift in the nominal open-circuit voltage
        dT = self.t_celsius - self.t_nominal
        self.dI = self.i_temperature_coefficient * dT
        self.dV = self.v_temperature_coefficient * dT

        # i_sat is the saturation current for the forward diode
        self.i_sat = (self.i_sc - self.dI) / (
            exp((self.v_oc + self.dV) / self.v_eff_thermal) - 1
        )
        if self.i_sat < 0:
            raise Exception(f'negative saturation current for {self}')
        # The following is the prefactor for I_dr
        self.I_reverse_sat = self.i_sr * exp(
            self.k_reverse * self.v_breakdown / self.v_eff_thermal)
        # Make sure that we have a satisfactory vector of coarse voltage values
        # that are used to get close to the right answer before turning over to
        # Newton-Raphson

    @staticmethod
    def _comp_residual(vd_, *args):
        """This staticmethod is used in root finding. It represents
        satisfying Kirchhoff's current law. Because the root-finding
        routines expect a function, not another sort of object,
        this routine is a static method and the zeroth element of args
        is the actual (subclassed) StaticKimModel object.
        """
        # the zeroth argument is the StaticKimModel
        self = args[0]
        # set the diode voltage according to the passed parameter
        self.v_diode = vd_
        exponent = vd_ / self.v_eff_thermal
        try:
            i_forward_diode = self.i_sat * (exp(exponent) - 1)
        except:
            # the exponent is too large
            return -1e4
        try:
            i_reverse_diode = self.I_reverse_sat * \
                (exp(-self.k_reverse * exponent) - 1)
        except:
            return 1e4

        i_shunt = vd_ / self.r_shunt
        delta_i = self.i_photo + i_reverse_diode - \
            (i_forward_diode + i_shunt + self.i_pv)
        # logger.debug(f'{vd_:0.4f} V and {1000*delta_i:0.2f} mA')
        return delta_i

    def v_pv(self, insolation, current, v0=None):
        """
        For given insolation (W/m^2) and current (A), find the voltage across
        this cell. Actually, the insolation is compared to self.g_nominal to
        determine the photocurrent. Remember that you must first set the
        cell temperature with self.t_celsius.
        """
        from rootfinder import RootFinder

        self.i_photo = (self.i_sc * (1 + self.r_series / self.r_shunt) +
                        self.dI) * insolation / self.g_nominal
        self.i_pv = current

        # Can we estimate the right voltage??
        # If current < i_photo, then we should be in forward bias.
        # If current > i_photo, we could look for the current at which
        #   breakdown begins. If we are below that, then linearly
        #   interpolate to get a starting estimate and use newton.
        # If current

        if current > self.i_photo:
            # reverse bias
            v = max((self.i_photo - current) *
                    self.r_shunt, self.v_breakdown)
            rf = RootFinder(
                self._comp_residual,
                (self, ),
                f'v_pv calling _comp_residual for G = {insolation}, I = {current:.3f}',
                xseeds=[v],
                method='newton'
            )
        else:
            # Forward bias
            # We could use straightline approximations sloped by rshunt from i_photo and
            # r_series from v_oc, although I don't think r_series is the right quantity.
            seeds = np.array([0, 0.7, 0.9, 0.95, 1]) * self.v_oc
            rf = RootFinder(
                self._comp_residual,
                (self, ),
                f'v_pv calling _comp_residual for G = {insolation}, I = {current:.3f}',
                xseeds=seeds,
                method='brent'
            )
        # now find the root
        self.v_diode = rf()
        if self.v_diode == None:
            self.v_diode = 0
            # raise Exception(f'Got None from RootFinder for {rf}')
        return self.v_diode - self.r_series * current

    def voltage(self, IGtuples):
        voltages = np.zeros([len(IGtuples)])
        v = self.v_oc * 0.8
        for n, ig in enumerate(IGtuples):
            i, g = ig
            v = self.v_pv(g, i, v)
            voltages[n] = v
        return voltages

    def generate_interpolant(self, filename, **kwargs):
        insolations = kwargs.get('insolations', np.arange(0, 1200, 20))
        temperatures = kwargs.get('temperatures', np.arange(0, 200, 5))
        currents = kwargs.get('currents', np.arange(0, 8.5, 0.01))

        # create the three-dimensional "cube" of voltage values
        voltages = np.zeros(
            (len(currents), len(insolations), len(temperatures))
        )
        for k in range(len(temperatures)):
            temp = temperatures[k]
            self.t_celsius = temp
            for j in range(len(insolations)):
                insol = insolations[j]
                v = self.v_oc
                logger.info(f'({insol:.1f} W/m^2, {temp}{DEG}C)')
                for i in range(len(currents)):
                    v = self.v_pv(insol, currents[i], v)
                    voltages[i, j, k] = v
        from scipy.io import savemat
        if not filename.endswith('.mat'):
            filename += '.mat'
        filename = os.path.join(os.path.split(__file__)[0], filename)
        savemat(filename, {
            'currents': currents,
            'insolations': insolations,
            'temperatures': temperatures,
            'v': voltages,
        })


class KimParameterAdjustor(object):
    """
    The goal of this class is to explore ranges of various parameters
    used in the Kim model to achieve a match with the parameters listed
    for the panel. These values are the power, current, and voltage at
    the maximum power point at standard conditions (1 kW/m^2, 25°C) and
    at reduced input (800 W/m^2, temperature_800).
    """

    def __init__(self,
                 model: StaticKimModel,
                 n_cells: int,
                 power_stc, v_stc, i_stc,
                 power_800, v_800, i_800, temperature_800):
        self.model = model
        self.n_cells = n_cells
        self.power_stc = power_stc
        self.v_stc = v_stc
        self.i_stc = i_stc
        self.t_800 = temperature_800
        self.power_800 = power_800
        self.v_800 = v_800
        self.i_800 = i_800

    def evaluate(self):
        fullsun = self.model.mpp(1000, 25)
        partial = self.model.mpp(800, self.t_800)
        n = self.n_cells
        values = {
            'pstc': n * fullsun['pmp'],
            'istc': fullsun['imp'],
            'vstc': n * fullsun['vmp'],
            'p800': n * partial['pmp'],
            'i800': partial['imp'],
            'v800': n * partial['vmp'],
        }
        errors = {
            'pstc': values['pstc'] / self.power_stc - 1,
            'istc': values['istc'] / self.i_stc - 1,
            'vstc': values['vstc'] / self.v_stc - 1,
            'p800': values['p800'] / self.power_800 - 1,
            'i800': values['i800'] / self.i_800 - 1,
            'v800': values['v800'] / self.v_800 - 1,
        }
        # Now, we want to evaluate how close we are. Maybe the square root
        # of the sum of squares. Also the worst agreement.
        keys = sorted(errors.keys())
        for key in keys:
            errors[key] *= 100  # convert to percentages
        errs = np.asarray([errors[k] for k in keys])  # into percentage
        total = np.dot(errs, errs)  # sum of squares
        abserrs = np.abs(errs)
        nworst = np.argmax(abserrs)
        worst = errs[nworst]
        return {
            'values': values,
            'errors': errors,
            'max_error': worst,
            'offender': keys[nworst],
            'total': total,
        }

    def scan(self, param, pmin, pmax, update=True, steps=15):
        """
        Scan a range of one parameter and look for the best value, updating
        the model if update is True.
        """
        candidates = np.linspace(pmin, pmax, steps)
        totals = []
        original = getattr(self.model, param)
        for c in candidates:
            setattr(self.model, param, c)  # set the value
            totals.append(self.evaluate()['total'])
        ibest = np.argmin(totals)
        worst = np.argmax(totals)
        if update:
            setattr(self.model, param, candidates[ibest])
        else:
            setattr(self.model, param, original)
        results = pd.DataFrame()
        results[param] = totals
        results.index = candidates
        return results

    def tweak(self, param):
        for x in (0.5, 0.2, 0.1, 0.05, 0.01):
            val_i = getattr(self.model, param)
            self.scan(param, val_i * (1 - x), val_i * (1 + x), True)
        return getattr(self.model, param)

    def __str__(self):
        ev = self.evaluate()
        v, e = ev['values'], ev['errors']
        keys = sorted(e.keys())
        lines = [
            f'Total     = {ev["total"]:.2f}%',
            f'Max Error = {ev["max_error"]:.2f}% ({ev["offender"]})',
        ]
        for key in keys:
            lines.append(f'{key:8}{v[key]:8.2f}{e[key]:8.2f}%')
        lines.append('\n')
        lines.append(f'r_shunt     = {self.model.r_shunt:.3f} Ω')
        lines.append(f'r_series    = {self.model.r_series * 1000:.3f} mΩ')
        lines.append(f'ideality    = {self.model.ideality:.3f}')
        lines.append(f'k_reverse   = {self.model.k_reverse:.3e}')
        lines.append(f'v_breakdown = {self.model.v_breakdown:.2g} V')
        return "\n".join(lines)


class IPVModel(StaticKimModel):
    """
    Attempt to model the idealPV cells
    """

    def __init__(self, efficiency=0.173, **kwargs):
        "We scale i_sc by the rated efficiency"
        super().__init__('IdealPV')
        self.set_efficiency(efficiency)

    def set_efficiency(self, eff):
        self.name = f'idealPV {100*eff:.1f}%'
        self.i_sc *= eff / 0.181


if __name__ == '__main__':
    if False:
        ku = IPVModel(vc, 0.173)
        print(ku)
    if True:
        model = StaticKimModel('KimMC')
        print(model)
    #  model = ku265 = KU265Model(vc)
    model.t_celsius = 25
    volts = model.voltage([(2, 1000), (0.0, 1000)])
    py.plot(model.iv_curves(
        [0, 60], [25, 35, 45], i_min=-2, split=False, mpp=False)[0])
    # fig = model.iv_curves([200 * x for x in range(6)], [25, 45], i_min=-2)
    # py.plot(fig[0])
