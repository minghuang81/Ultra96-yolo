from distutils.core import setup, Extension
setup(name='convNetC', version='1.0', 
      ext_modules=[Extension('convNetC', 
                             ['convNetC_itf.c',
                              '../lib/conv_net.c',
                              '../lib/yolo.c',
                              '../model/convNetC_fct.c',
                              ])])