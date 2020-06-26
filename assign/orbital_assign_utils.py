import numpy as np
import h5py as h5

def copy_check(chk_original, chk_copy):
    import subprocess as sp
    sp.call(['cp',chk_original, chk_copy])

def AngToBohr(pos):
    '''
    Returns the position given in 'pos'(specified in Angstrom) in units Bohr
    '''
    newPos = []
    for p in pos:
        newPos.append(p*1.889725989)
    return tuple(newPos)

def write_to_chk(oldchk,newchk,C,erkale=True,restr=True):
    '''
    Saves the sorted orbitals in a .chk file, using the 'oldchk' to get all other info
    '''
    copy_check(oldchk,newchk)
    f = h5.File(newchk,'a')
    if erkale: # use ERKALE format
        try:
            f['/C'][...] = C
        except:
            print("[!] failed to write orbitals to erkale .chk file")
    else: # use pyscf format
        try:
            f['/scf/mo_coeff'][...] = C
        except:
            print("[!] failed to write orbitals to pyscf .chk file")
    f.close()


def write_sorted_sizeDist(fname,orbList):
    '''
    Save the sorted sizeDist file
    '''

    print("[+] : writing sorted sizeDist file to {}".format(fname))
    
    # Q: do I need to re-number these? - I think so, we will want the 'new' index
    #       - actually, the orbital assign script doesn't read the orbital from the 
    #         sizeDist file. It simply counts as it goes based on the order in which
    #         the orbitals are list

    f = open(fname,'w')
    for orb in orbList:
        #print(orb)
        f.write("{} {} {} ({},{},{})\n".format(orb[0], orb[2], orb[3], orb[1][0],orb[1][1],orb[1][2]))
        
    f.close()

def write_orbital_dist(name, orbList, bins=100, display_plot=False):
    '''
    writes a file called 'name' which contains the cumulative distribution of local orbitals
    as a function of the metric value in orbList.
    '''
    
    # for tests, delete later
    import matplotlib.pyplot as plt

    metVals = []
    for orb in orbList:
        metVals.append(orb[4]) # this is the orb. centroid
    
    _hist, binLabels = np.histogram(metVals, bins=bins, range=(-0.001,np.amax(metVals)), density=False)
    count = np.cumsum(_hist)

    if display_plot:
        plt.plot(binLabels[1:], count)
        plt.title("Orbital count vs R")
        plt.show()

    f = open(name,'w')
    for i in range(count.shape[0]):
        f.write("%f %d\n" % (binLabels[1 + i], count[i]))
    f.close()

    return count


def orbitalSetStats(orbitals):
    '''
    prints localization stats. of the locality of the orbitals in 'orbitals'

    orbitals is a list with the entries as follows:
    orbitals = [[index, centroid position, sigma2, sigma4, distance from centroid to origin],
                [ '' ],
                  ... ,
                 [ '']]

    '''
    sum2ndMom = 0.0
    sum4thMom = 0.0

    max2ndMom = 0.0
    max4thMom = 0.0

    Norbs = len(orbitals)

    for orb in orbitals:
        sum2ndMom = sum2ndMom + orb[2]
        if (orb[2] > max2ndMom):
            max2ndMom = orb[2]
        sum4thMom = sum4thMom + orb[3]        
        if (orb[3] > max4thMom):
            max4thMom = orb[3]
    print "   - Num. Orbitals: ", Norbs
    if Norbs > 0:
        print "   - mean 2nd central moment: ", sum2ndMom/Norbs
        print "   - mean 4th central moment: ", sum4thMom/Norbs
        print "   - max 2nd central moment: ", max2ndMom
        print "   - max 4th central moment: ", max4thMom

def get_orbital_sizeDist(fname, startIndex=0):
    
    f = open(fname, 'r')
    
    orbitals = []
    
    index = startIndex
    
    for line in f:
        lineList = line.split()
        centL = lineList[3]
        cent = eval(centL)
        rad = 0.0
        orbitals.append((index, cent, eval(lineList[1]), eval(lineList[2]), rad))
        index = index + 1

    f.close()
    return orbitals, index

def stats_from_sizeDist(fname):
    '''
    Simple shortcut function for a common combination of function calls
    '''
    orbitals = get_orbital_sizeDist(fname)[0]
    orbitalSetStats(orbitals)
