#!/usr/bin/python
#! /bliss/users/blissadm/python/bliss_python/suse82/bin/python


"""
Wrapper script to refine a single grain using 
all peaks in a dataset (so that hkl assignments
are not a problem).
"""

from ImageD11 import refinegrains, indexing
import logging, sys

def fitgrain(options):
    """
    Fits grains to a dataset using all peaks within tol
    """
    o = refinegrains.refinegrains(tolerance = options.tol)
    o.loadparameters(options.parfile)
    o.readubis(options.ubifile)
    o.loadfiltered(options.fltfile)
    o.generate_grains()
    o.refineubis(quiet = False)
    print o.gv[:10,:]
    o.parameterobj.varylist = options.varylist
    for p in options.fixlist:
        try:
            o.parameterobj.varylist.remove(p)
        except:
            pass
    logging.info("Varying " + str(o.parameterobj.varylist))
    o.fit(maxiters = options.steps)
    o.refineubis(quiet = False)
    o.saveparameters(options.newparfile)
    ul = [g.ubi for g in o.grains.values()]
    indexing.write_ubi_file(options.newubifile, ul)




if __name__=="__main__":
    
    console = logging.StreamHandler(sys.stdout)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)-8s : %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    console.setLevel(logging.DEBUG)
    root = logging.getLogger('')
    root.addHandler(console)
    root.setLevel(logging.DEBUG) # should we process everything...?

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-p",  "--parfile", action="store",
                      dest="parfile", type="string",
                      help="Name of parameter file")
    parser.add_option("-u",  "--ubifile", action="store",
                      dest="ubifile", type="string",
                      help="Name of ubi file")
    parser.add_option("-f",  "--fltfile", action="store",
                      dest="fltfile", type="string",
                      help="Name of flt file")
    parser.add_option("-P", "--newparfile", action="store",
                      dest="newparfile", type="string",
                      help="Name of new parameter file")
    parser.add_option("-U", "--newubifile", action="store",
                      dest="newubifile", type="string",
                      help="Name of new ubi file")
    parser.add_option("-v", "--vary", action="append",
                      dest="varylist", type="string",
                      default =    [ "y_center","z_center",
                                     "tilt_y","tilt_x","tilt_z","wedge",
                                     "t_x","t_y","distance"],
                      help="Parameters to vary"  )
    parser.add_option("-x", "--fiX", action="append",
                      dest="fixlist", type="string",
                      default = ['chi'],
                      help="Parameters to fix (overrides vary)")
    parser.add_option("-t", "--tol", action="store",
                      dest="tol", type="float",
                      default =   0.5,
                      help="Tolerance to use in peak assignment")
    parser.add_option("-s", "--steps", action="store",
                      dest="steps", type="int",
                      default =   1000,
                      help="Number of simplex iterations")
    
    parser.description = """
Fitgrain should attempt to fit one or more grains to a dataset
using the parameters specified on the command line.
    """
    
    options, args = parser.parse_args()

    for name in ["parfile" , "newparfile",
                 "ubifile", "newubifile",
                    "fltfile"]:
        if getattr(options, name) is None:
            parser.print_help()
            logging.error("Missing option "+name)
            import sys
            sys.exit()
        
    try:
        fitgrain(options)
    except:
        parser.print_help()
        import traceback
        logging.error("An error occurred, details follow")
        traceback.print_exc()
    
