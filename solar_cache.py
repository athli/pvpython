# -*- coding: utf-8 -*-

"""
  Author:   Peter N. Saeta --<saeta@hmc.edu>
  Purpose:  implement dynamic programming for solar_cell_models
  Created:  14 June 2019
  
  We may want to revive these routines; at the moment, they are
  just sitting here and need to be revised to work.
"""

import os

import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('SolarCellModels')

from solar_config import SolarConfig
solar_config = SolarConfig(
    os.path.join(os.path.split(__file__)[0], 'solar_cell_models.ini')
)

from solar_cell_models import SolarCellModel


class SolarCellDPModel(object):
    """
    The job of a SolarCellDPModel is to use dynamic programming to
    reduce the amount of root-finding required by caching results
    at a predefined comb of current values for (insolation, temperature) pairs.
    To implement dynamic programming, the model is not generally called
    by a client, but instead called by the VoltageCache, which returns a
    reference to voltage vector, fetching from cache if available, and
    computing it by calling SolarCellModel.voltage() if it doesn't. Subclasses
    must override this method.
    """

    def __init__(self, voltage_cache, model):
        self._cache = voltage_cache
        self._model = model

    @property
    def cache(self): return self._cache

    @property
    def model(self):
        return self._model


class Interpolant(object):
    """
    Load a three-dimensional data cube we can use for interpolation from
    a MatLab .mat file that contains a data "cube" of currents, insolations,
    temperatures, and potentials. Create a linear RegularGridInterpolant
    to be used to return voltage values.

    The filename passed to the creator should reference a matlab .mat
    file that contains three one-dimensional vectors,
        'currents' (in A), 'insolations' (in W/m^2), and 'temperatures' (C),
    and a three-dimensional vector of corresponding voltages.
    """

    def __init__(self, filename):
        from scipy.io import loadmat
        from scipy.interpolate import RegularGridInterpolator

        self.filename = filename
        matdata = loadmat(filename)
        self.currents = matdata['currents'][0]
        self.insolations = matdata['insolations'][0]
        self.temperatures = matdata['temperatures'][0]
        potentials = matdata['v']

        #

        # Now develop the interpolant
        self.interpolant = RegularGridInterpolator(
            (self.currents, self.insolations, self.temperatures),
            potentials, method="linear", bounds_error=True, fill_value=np.nan)

    def __str__(self):
        return f'Interpolant({os.path.split(self.filename)[1]})'

    def __call__(self, vigt):
        """
        Given a vector of (current, insolation, and TC) values, return the vector of
        voltage values.
        """
        return self.interpolant(vigt)


class VoltageCache(object):
    """
    The purpose of the cache is to minimize the number of calls to compute
    the I-V curve for a particular (G,T) condition on a particular model.
    Because a simulation may involve more than one model, we need to use
    a coordinated approach with a shared cache.

    I-V curves are calculated for a fixed set of current values, which
    are set by set_currents(array). If a new array of currents is passed,
    different from the currently cached one, the database of stored I-V
    curves resets.

    Caches store I-V curves for individual cells in self.cell_voltages
    and for strings of cells in self.string_voltages, which are lists.
    Each entry in the list is an np.array of voltages corresponding to
    the currents in self.currents. For each triplet of (model, G, T),
    where G is insolation in W/m^2 and T is cell temperature in °C,
    an entry is made in the self.cell_keys dictionary that maps the
    key (model, G, T) to the integer index in the self.cell_voltages
    list. That is, the key is a tuple (to be immutable, as is required)
    for dictionary keys.

    At the moment, computation of keys is the responsibility of the
    caller, which is SolarCellModel.gt_voltages(gt_tuple). Sorta dumb!
    By the way, gt_voltages does not return voltages. Rather, it returns
    the index into the cell_voltages list.

    For cell strings, the key is more complicated and consists of

    key = tuple([self.model] + list(result.items()))

    where the results dictionary maps indices in the self.cell_voltages
    list to the number of cells matching that condition:

    {index: num_cells}

    However, what's stored in the self.string_voltages list is more
    complicated. Each entry is a dictionary of

    {'voltages': array, 'bypass': array_of_boolean or None}

    If this string has a bypass diode, the boolean array reports
    whether this string is bypassed or not at each current.

    Write about merge_strings, which is called in
    PanelString.set_voltages().

    """

    def __init__(self):
        self.currents = np.array([])
        self.cell_keys = dict()  # mapping of (model, G, T) tuples to index
        self.cell_voltages = []
        self.string_keys = dict()
        self.string_voltages = []

    def set_currents(self, new_currents):
        if np.array_equal(new_currents, self.currents):
            return
        # reset the database
        self.cell_keys = dict()
        self.cell_voltages = []
        self.string_keys = dict()
        self.string_voltages = []
        # and set the currents
        self.currents = np.array(new_currents)
        logger.debug(f'Set a {len(new_currents)}-element vector of currents.')

    def gt_voltages(self, key):
        """Return the integer index into self.voltages corresponding to the
        given key, computing the values if necessary"""
        # If we already have the (model, G, T) key, we're done
        if key in self.cell_keys:
            return self.cell_keys[key]
        model, insolation, temperature = key  # unpack the key
        index = len(self.cell_voltages)
        self.cell_keys[key] = index
        self.cell_voltages.append(
            model.voltage([(i, insolation, temperature) for i in self.currents]))
        logger.debug(f'[{index:04d}] CV {insolation:.1f}   {temperature:.1f}{DEG}C')
        return index

    def str_voltages(self, key, bypass_voltage):
        """Return the integer index into self.voltages corresponding to the
        given key, computing the values if necessary"""
        # If we already have the key, we're done
        if key in self.string_keys:
            return self.string_keys[key]
        model = key[0]  # unpack the key
        pattern = key[1:]
        index = len(self.string_voltages)
        self.string_keys[key] = index
        # compute the combined voltage array
        try:
            cindex, multiple = pattern[0]
            svoltages = self.cell_voltages[cindex] * multiple
            for cindex, multiple in pattern[1:]:
                svoltages += self.cell_voltages[cindex] * multiple
        except:
            svoltages = self.cell_voltages[pattern[0]] * pattern[1]

        if bypass_voltage > 0:
            bypassed = svoltages < -bypass_voltage
            svoltages[bypassed] = -bypass_voltage
            self.string_voltages.append({
                'voltages': svoltages,
                'bypass': bypassed,
            })
        else:
            self.string_voltages.append({
                'voltages': svoltages,
                'bypass': None,
            })
        logger.debug(f'[{index:04d}] SV {pattern}')
        return index

    def merge_strings(self, pattern):
        voltages = np.zeros(len(self.currents))
        for key, val in pattern.items():
            voltages += val * self.string_voltages[key]['voltages']
        return voltages

    def plot_cell_iv(self, n, show=True):
        """
        Make a plot of the I-V curve for a stored curve n.
        Raises an exception for invalid index.
        """
        from solar_plots import ivplot

        # Figure out the key
        for k, v in self.cell_keys.items():
            if v == n:
                (model, g, t) = k
                title = f'{model.name} {g:.1f} W/m$^2$  {t:.1f}$^{{\circ}}$C'
                title = title.replace('%', r'\%')
        f, nax, pax = ivplot(
            self.currents, [self.cell_voltages[n]], title=title)
        if show:
            plt.show()

    def plot_string_iv(self, n, show=True):
        from solar_plots import ivplot
        ivplot(self.currents, [self.string_voltages[n]['voltages']])
        if show:
            plt.show()


class InterpolantModel(SolarCellDPModel):
    def __init__(self, filename, voltage_cache, **kwargs):
        super().__init__(voltage_cache, **kwargs)
        self.interpolant = Interpolant(filename)

    def __call__(self, vIGT):
        """
        Calculate using the interpolant voltage values corresponding to
        the insolation insolation (W/m^2), temperature temperature (C), and current(s) I (A).
        """
        return self.interpolant(vIGT)

    def voltage(self, IGTtuples):
        return self.interpolant(IGTtuples)


class KyoceraModel(InterpolantModel):
    def __init__(self, voltage_cache, **kwargs):
        filename = os.path.join(os.path.split(__file__)[0], 'Kyocera.mat')
        super().__init__(filename, voltage_cache, name='Kyocera', **kwargs)


class idealPVModel(InterpolantModel):
    """
    This is all sort of wrong, because the different panels have two different
    efficiencies. It needs to be fixed.
    """

    def __init__(self, voltage_cache):
        filename = os.path.join(os.path.split(__file__)[0], 'idealPV.mat')
        super().__init__(filename, voltage_cache, name='idealPV')


class SaetaModel(InterpolantModel):
    def __init__(self, voltage_cache):
        filename = os.path.join(os.path.split(__file__)[0], 'Saeta.mat')
        super().__init__(filename, voltage_cache, name='Saeta')


if __name__ == '__main__':
    print("Boo!")
