import os.path
from setuptools import setup
version = '0.1.5'

setup(name='fview_SphereTrax',
      description='fview plugin to track angular velocity of a sphere',
      version=version,
      entry_points = {
    'motmot.fview.plugins':'fview_SphereTrax = fview_SphereTrax:SphereTrax_Class',
    },

      packages = ['fview_SphereTrax'],
      zip_safe=True,
      package_data = {'fview_SphereTrax':['fview_SphereTrax.xrc',
                                          'data/camera_cal.txt',
                                          'data/sphere_defaults.txt'],
                      },
      )
