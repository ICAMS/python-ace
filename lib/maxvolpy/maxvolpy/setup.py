def configuration(parent_package=None, top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy.__config__ as npconf
    import os
    from os.path import exists, getmtime
    config = Configuration('maxvolpy', parent_package, top_path) 
    
    start_setup_dir = os.getcwd()
    cur_setup_dir = os.path.dirname(__file__)
    if len(cur_setup_dir) > 0:
        os.chdir(cur_setup_dir)
    if not exists('_maxvol.pyx') or \
            getmtime('_maxvol.pyx.src') > getmtime('_maxvol.pyx'):
        exec(open('_maxvol.pyx.src').read())
    os.chdir(start_setup_dir)

    config.add_extension('_maxvol',
            sources=['_maxvol.pyx'],
            extra_info=npconf.lapack_opt_info
            )
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration().todict())
