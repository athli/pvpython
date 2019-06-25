"""

"""
import configparser

class SolarConfig(object):
    
    def __init__(self, filename):
        self.cfg = configparser.ConfigParser()
        self.load(filename)
        
    def load(self, filename):
        self.cfg.read(filename if isinstance(filename, (list, tuple)) else [filename, ], encoding='utf8')
    
    def __getitem__(self, key):
        """Act like the normal ConfigParser"""
        return self.cfg[key]
    
    def get(self, section, name):
        """
        Get an option, converted to int or float as possible
        """
        s = self.cfg[section][name]
        if '#' in s:
            s = s.split('#')[0]
        try:
            return eval(s)
        except:
            return s

    def section(self, section):
        """Return a dictionary of all items in a section"""
        dic = {}
        for (key, valstr) in self.cfg.items(section=section):
            dic[key] = self.get(section, key)
        return dic
            

if __name__ == '__main__':
    import os
    sc = SolarConfig(os.path.join(os.path.split(__file__)[0], 'solar_cell_models.ini'))
    print(sc['KU265'])
    print(sc.get('KU265', 'i_sc'))
    print(sc.section('KU265'))