# -*- coding: utf-8 -*-

"""
  Author:  Peter N. Saeta --<saeta@hmc.edu>
  Purpose: Modeling solar installations, often with odd geometries
  Created: 05/29/18

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

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rootfinder import RootFinder
from solar_cell_models import SolarCellModel

import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('SolarPanels')


DEG = chr(0xb0)  # to avoid headaches with pylint
ZERO_TEMPERATURE = 273.15  # K


class SolarCell(object):
    """

    """

    def __init__(self, model, **kwargs):
        assert isinstance(
            model, SolarCellModel), "You must pass in a valid SolarCellModel"
        self.model = model

        self.x = np.nan                 # left edge, m
        self.y = np.nan                 # bottom edge, m
        self._temperature = 25          # celsius
        self._insolation = 0            # W/m^2
        self._voltage = np.nan          # set by ?
        self._current = np.nan
        self._power = np.nan
        self._units = {
            'R_series': 'Ω',
            'R_shunt': 'Ω',
            'width': 'm',
            'height': 'm',
            'x': 'm',
            'y': 'm',
            'V_oc': 'V',
            'I_sc': 'A',
            '_temperature': f'{DEG}C',
            '_insolation': 'W/m^2',
            'OCV_temp_coefficient': f'V/{DEG}C',
            'radiation_coefficient': f'{DEG}C/W',
        }
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _set_current(self, ambient_temperature, avg=False):
        """
        We assume that the current has actually been set by self.current.
        Given the ambient temperature and the cell's currently assumed temperature,
        compute the cell's voltage and therefore the power generated in the cell.
        Take this power away from the incident power and apply the temperature
        coefficient to determine the temperature rise of the cell. The return value
        is the difference between the calculated rise and the current temperature
        of the cell, which is updated before returning.
        """
        self._voltage = self.model.v_pv(self._insolation, self._current)
        self._power = self._voltage * self._current
        warming = (self._insolation * self.model.area - self._power) * \
            self.model.temperature_coefficient
        nt = warming + ambient_temperature
        new_temperature = 0.5 * (nt + self.temperature) if avg else nt
        delta_temperature = new_temperature - self.temperature
        self.temperature = new_temperature
        return delta_temperature

    def voltage(self, current, ambient_temperature, tolerance=0.01):
        """Iterate calls to _set_current until the temperature change
        is smaller than tolerance. This sets the value of _voltage, which
        we return here. If recomputation is not required, the value is
        preserved as self._voltage
        """
        self._current = current
        n, dT = 0, 10 * tolerance  # initialize dT with a value exceeding tolerance
        temperatures, dTs = [], []
        average = False
        while n < 8 and abs(dT) > tolerance:
            dT = self._set_current(ambient_temperature, average)
            n += 1
            dTs.append(dT)
            temperatures.append(self.temperature)
            if n > 2 and not average:
                average = abs(dTs[-1]) > abs(dTs[-2])
        if abs(dT) > tolerance:
            logger.debug(f'Failed to converge to temperature for {current} A.')
            # Let's see if we bracket and can try a bisection approach
            pairs = sorted(zip(temperatures, dTs), key=lambda x: x[1])
            if pairs[0][1] < 0 and pairs[-1][1] > 0:
                for j in range(len(pairs)):
                    if pairs[j][1] > 0:
                        Tlo, Thi = pairs[j - 1][0], pairs[j][0]
                        break

            while n < 20 and abs(dT) > tolerance:
                t = 0.5 * (Tlo + Thi)
                self.temperature = t
                dT = self._set_current(ambient_temperature)
                n += 1
                dTs.append(dT)
                temperatures.append(self.temperature)
                if dT < 0:
                    Thi = t
                else:
                    Tlo = t
        return self._voltage

    @property
    def temperature(self): return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -ZERO_TEMPERATURE:
            raise ValueError(f'Temperatures below {-ZERO_TEMPERATURE}°C are not possible!')
        self._temperature = value
        self.model.t_celsius = value

    @property
    def insolation(self): return self._insolation

    @insolation.setter
    def insolation(self, value):
        if value < 0:
            raise ValueError('Insolation cannot be negative')
        self._insolation = value

    def __str__(self):
        strs = []
        for key in self.__dict__.keys():
            if key.startswith('_'):
                strs.append(f'{key[1:]} = {self.valunit(key)}')
        return '\t' + "\n\t".join(sorted(strs))

    def valunit(self, field, prec=3):
        """Return a string representing the value in the field, formatted
        to the requisite precision and with appropriate units.
        """
        try:
            v = getattr(self, field)
            form = '%.' + '%d' % prec + 'f'
            value = form % v
        except:
            return str(v)
        try:
            unit = self.units[field]
            return value + ' ' + unit
        except:
            pass
        return value


class NCellString(object):
    """
    An N Cell String is a set of cells wired in series, with an optional
    bypass diode. The cells are assumed to derive from the same model, but
    the amount of sunlight hitting various cells may differ from one another.

    """

    def __init__(self, model, num_cells,
                 bypass=True, diode_voltage=0.6, tolerance=1e-3):
        self.model = model
        self.cell = SolarCell(model)  # make a cell

        self.num_cells = num_cells
        self.has_bypass_diode = bypass
        self.diode_voltage = diode_voltage

        self._insolation = []
        self._bypassed = False
        self._current = 0
        self._voltage = 0
        self._voltages = []
        self._tolerance = tolerance

        # Set the level at which the bypass diode will kick in
        self._bypass_level = -diode_voltage

    @property
    def insolation(self):
        return self._insolation

    @insolation.setter
    def insolation(self, values):
        """
        Pass in a list of the form [(num1, val1), (num2, val2), ...], or
        a single value to set all cells to that insolation.
        """
        if isinstance(values, (int, float)):
            self._insolation = [(self.num_cells, values)]
        elif isinstance(values, (tuple, list)):
            n = self.num_cells
            self._insolation = []
            for setting in values:
                assert len(
                    setting) == 2, "Pass in [(num1, val1), (num2, val2)...]"
                self._insolation.append(tuple(setting))
                n -= setting[0]  # subtract from the total
            assert n == 0, "You need to specify insolation conditions for all the cells"

    def voltage(self, i_val, ambient_temperature):
        self._voltage = 0.0
        self._current = i_val
        self._voltages = []
        self._temperatures = []
        for n, g in self._insolation:
            self.cell.insolation = g
            vcell = self.cell.voltage(i_val, ambient_temperature)
            self._voltage += n * vcell
            self._voltages.append(vcell)
            self._temperatures.append(self.cell.temperature)
        # Should we bypass?
        self._bypassed = self.has_bypass_diode and self._voltage < self._bypass_level
        return -self.diode_voltage if self._bypassed else self._voltage

    def bypass(self, i_val, ambient_temperature):
        """
        If we are in a condition to be bypassed, the current i_val is greater
        than the string will support without opening the bypass diode. This
        function will look for a value of current smaller than i_val that
        satisfies Kirchhoff's loop law. We can probably do a bisection search
        to home in on the limiting case where the 
        """
        di, i_lo, i_hi = i_val * 0.1, i_val, i_val
        self.voltage(i_val, ambient_temperature)
        voltages = [self._voltage - self._bypass_level, ]
        currents = [i_val, ]
        while i_lo > 0 and voltages[-1] < self._bypass_level:
            i_lo -= di
            self.voltage(i_lo, ambient_temperature)
            currents.append(i_lo)
            voltages.append(self._voltage - self._bypass_level)
        n, i_hi = 0, i_lo + di
        rf = RootFinder(self.Kirchhoff_loop,
                        (self, ambient_temperature),
                        f'none',
                        xseeds=currents,
                        yseeds=voltages,
                        method='brent')
        i_at_bypass = rf()
        return i_at_bypass

    @staticmethod
    def Kirchhoff_loop(*args):
        i = args[0]
        self = args[1]
        self.voltage(i, args[2])
        return self._voltage - self._bypass_level


class SolarPanel(object):
    """
    A panel comprises one or more NCellStrings. I expect the interesting question
    to be what happens as we vary the number of strings that experience one
    sort of illumination when we shade a cell in one string.
    """

    def __init__(self, model, num_strings, cells_per_string, **kwargs):
        self.model = model
        self.num_strings = num_strings
        self.strings = []
        self._string_params = {
            'model': model,
            'num_cells': cells_per_string,
            'bypass': kwargs.get('bypass', True),
            'diode_voltage': kwargs.get('diode_voltage', 0.6),
            'tolerance': kwargs.get('tolerance', 1e-3),
        }
        self._insolation = []

    @property
    def insolation(self):
        return self._insolation

    @insolation.setter
    def insolation(self, values):
        print(values)


if __name__ == "__main__":
    from solar_cell_models import StaticKimModel

    model = StaticKimModel(inikey='KimMC')
    nstring = NCellString(model, 20)
    nstring.insolation = ((19, 1000.0), (1, 500))
    # nstring.insolation = 800
    currents = np.linspace(3, 4.25, 50)
    voltages = np.zeros(50)
    temperatures = []
    ambient = 35.0
    for n, i in enumerate(currents):
        v = nstring.voltage(i, ambient)
        if nstring._bypassed:
            nstring.bypass(i, ambient)
        voltages[n] = nstring._voltage
        temperatures.append(nstring._temperatures)
    df = pd.DataFrame(voltages, index=currents, columns=('V', ))
    for n in range(len(nstring.insolation)):
        df[f'G{nstring.insolation[n][1]:.0f}'] = [t[n] for t in temperatures]
    df['power'] = df.V * currents
    df.plot()
    plt.show()
    print(df)
    if False:
        plt.plot(voltages, currents, 'r.')
        plt.xlabel("$V$")
        plt.ylabel("$I$")
        plt.show()

