
from setuptools import setup, find_packages
from setuptools.extension import Extension 
from Cython.Build import cythonize

import numpy


rst_sourcefiles = ['RSTPython.pyx', 
                   'RStarTree.c','RSTInterUtil.c','RSTInOut.c','RSTInstDel.c','RSTUtil.c','RSTQuery.c','RSTJoin.c','RSTFunctions.c']

rst_sourcefiles = ['./spykshrk/realtime/rst/' + src for src in rst_sourcefiles]

binary_rec_sourcefiles = ['./spykshrk/realtime/binary_record_cy.pyx']

pp_decode_sourcefiles = ['./spykshrk/franklab/pp_decoder/pp_clusterless_cy.pyx']

extensions = [Extension('spykshrk.realtime.rst.RSTPython', rst_sourcefiles, include_dirs=[numpy.get_include()]),
              Extension('spykshrk.realtime.binary_record_cy', binary_rec_sourcefiles, include_dirs=[numpy.get_include()]),
              Extension('spykshrk.franklab.pp_decoder.pp_clusterless_cy', pp_decode_sourcefiles, include_dirs=[numpy.get_include()], define_macros=[('CYTHON_TRACE', '1')])
              ]

setup (name = 'spykshrk_realtime',
       ext_modules = cythonize(extensions, compiler_directives={'linetrace': True}),
       packages=find_packages('./'),
       package_dir={'spykshrk':'./spykshrk'},
       install_requires=['scipy', 'iPython','mpi4py', 'numpy', 'Cython', 'pandas',
                         'trodesnetwork', 'seaborn', 'pyqtgraph', 'matplotlib'])
