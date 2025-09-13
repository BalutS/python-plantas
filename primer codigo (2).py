#Esta versión del código es una versión medio limpia para el profe Crispulo, guardada el 11 de marzo de 2025

# standard library
import collections
from enum import Enum
import math
import os.path
import pprint
#import statistics
import sys


# third party library
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#import seaborn as sns
from matplotlib.patches import Rectangle
#import matplotlib.ticker as ticker


# set true to plot scans with positive power in different color
# this is for powder bed fusion
PLOT_POWER = True
POWER_ZERO = 1
IGNORE_ZERO_POWER = True

# Element namedtuple
Element = collections.namedtuple('Element', ['x0', 'y0', 'x1', 'y1', 'z'])

# set true to add axis-label and title
FIG_INFO = True

# MARGIN RATIO
MARGIN_RATIO = 0.2

# zero tolerance for is_left check
ZERO_TOLERANCE = 1e-12

# global variables
pp = pprint.PrettyPrinter(indent=4)

### under construction
# plot polygon
HALF_WIDTH = 0.6 # FDM regular

HORIZONTAL_SHRINK_RATIO = (1 / 1000) * (1 / (600 / 25)) # wrench
DELTA_Z = 2e-5

LASER_POWER = 195
LASER_SPEED = 0.8
TRAVEL_SPEED = 0.8



class GcodeType(Enum):
    """ enum of GcodeType """

    FDM_REGULAR = 1
    FDM_STRATASYS = 2
    LPBF_REGULAR = 3
    LPBF_SCODE = 4

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class GcodeReader:
    """ Gcode reader class """

    def __init__(self, filename,filetype=GcodeType.FDM_REGULAR):
        if not os.path.exists(filename):
            print("{} does not exist!".format(filename))
            sys.exit(1)
        self.filename = filename
        self.filetype = filetype
        
        self.refSolida_file = "ref_solido.GCODE"
        self.refSolida_segs = None
        
        self.n_segs = 0  # number of line segments
        self.segs = None  # list of line segments [(x0, y0, x1, y1, z)]
        self.n_layers = 0  # number of layers
        self.seg_index = []
        self.xyzlimits = None
        #self.minDimensions = None
        self.segs = self._read(filename)
        self.xyzlimits = self._compute_xyzlimits(self.segs)
        self.minDimensions = self.get_specimenDimensions()
        print(self.minDimensions)
  
    def _read(self, filename):
        """
        read the file and populate self.segs, self.n_segs and
        self.seg_index_bars
        """
        if self.filetype == GcodeType.FDM_REGULAR:
            segs = self._read_fdm_regular(filename)
       
        else:
            print("file type is not supported")
            sys.exit(1)
        
        return segs
        
    def _compute_xyzlimits(self, seg_list):
        """ compute axis limits of a segments list """
        xmin, xmax = float('inf'), -float('inf')
        ymin, ymax = float('inf'), -float('inf')
        zmin, zmax = float('inf'), -float('inf')
        for x0, y0, x1, y1, z in seg_list:
            xmin = min(x0, x1) if min(x0, x1) < xmin else xmin
            ymin = min(y0, y1) if min(y0, y1) < ymin else ymin
            zmin = z if z < zmin else zmin
            xmax = max(x0, x1) if max(x0, x1) > xmax else xmax
            ymax = max(y0, y1) if max(y0, y1) > ymax else ymax
            zmax = z if z > zmax else zmax
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    
    def _read_fdm_regular(self, filename):
        """ read fDM regular gcode type """
        with open(filename, 'r') as infile:
            # read nonempty lines
            lines = (line.strip() for line in infile.readlines()
                     if line.strip())
                    
            new_lines = []
            
            i = 0
            
            for line in lines:
                    
                if line.startswith('G'):
                    idx = line.find(';')
                    if idx != -1:
                        line = line[:idx]
                        
                    new_lines.append(line)
            
            lines = new_lines
       
        #pp.pprint(lines) # for debug
        #print("size of lines: ")
        #print(len(lines))
        
        segs = []
        temp = -float('inf')
        gxyzef = [temp, temp, temp, temp, temp, temp, temp]
        d = dict(zip(['G', 'X', 'Y', 'Z', 'E', 'F', 'S'], range(7)))
        seg_count = 0
        z = -math.inf
        x0 = temp
        y0 = temp
        
        i = 0
        for line in lines:
            for token in line.split():
                gxyzef[d[token[0]]] = float(token[1:])
            if gxyzef[0] == 1 :
                if np.isfinite(gxyzef[3]):
                   z = gxyzef[3]
                if np.isfinite(gxyzef[1]) and np.isfinite(gxyzef[2]) and not np.isfinite(gxyzef[4]):
                   x0 = gxyzef[1]
                   y0 = gxyzef[2]
                else:
                   if np.isfinite(gxyzef[1]) and np.isfinite(gxyzef[2]) and (gxyzef[4] > 0):
                      segs.append((x0, y0, gxyzef[1], gxyzef[2], z))
                      #print("segmento: (%f, %f, %f)-->(%f, %f, %f), estruye %f" % (x0, y0, z, gxyzef[1], gxyzef[2], z, gxyzef[4]))
                      x0 = gxyzef[1]
                      y0 = gxyzef[2]
                      seg_count += 1
            gxyzef = [temp, temp, temp, temp, temp, temp, temp]
            i=i+1
            
        #self.n_segs = len(self.segs)
        segs = np.array(segs)
        
        #if filename != self.refSolida_file :
        self.n_segs = len(segs)
        self.seg_index = np.unique(segs[:,4])
        self.n_layers = len(self.seg_index)
        print("Número de segmentos: ", self.n_segs)
        print("Número de capas: ", self.n_layers-1)
        
        return segs
   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def get_specimenDimensions(self):
        n_zCoords = len(self.seg_index)
        mz = int(n_zCoords/2)
        
        mz_idx = self.seg_index[mz]
        
        mz_layerSegs = self.get_layerSegs(mz_idx, mz_idx)
        print(type(mz_layerSegs))
        
        mz_layerSegs = np.array(mz_layerSegs)
        
        minx = min(min(mz_layerSegs[:, 0]), min(mz_layerSegs[:,2]))
        miny = min(min(mz_layerSegs[:, 1]), min(mz_layerSegs[:,3]))
        
        maxx = max(max(mz_layerSegs[:, 0]), max(mz_layerSegs[:,2]))
        maxy = max(max(mz_layerSegs[:, 1]), max(mz_layerSegs[:,3]))
                
        minDimensions = [minx, miny, maxx, maxy]
        return minDimensions

    def get_layerSegs(self, min_layer, max_layer):
        temp = []
        for (x0, y0, x1, y1, z) in self.segs:
            if z >= min_layer and z <= max_layer:
                temp.append((x0, y0, x1, y1, z))
        return temp
    
    
    def remove_skirt(self, verbose = False):
        layer = 0.2
        
        if verbose :
           self.animate_minDimensions(layer, self.minDimensions)
       
        new_segs = [seg for seg in self.segs if not self.is_skirt(seg)]
        self.segs = new_segs
        
        if verbose :
           self.animate_minDimensions(layer, self.minDimensions)  
    

    def is_skirt(self, seg):
        
        minx = self.minDimensions[0]
        miny = self.minDimensions[1]
        maxx = self.minDimensions[2]
        maxy = self.minDimensions[3]
        
        return (seg[0] < minx or seg[1] < miny or 
                seg[2] < minx or seg[3] < miny or
                seg[0] > maxx or seg[1] > maxy or 
                seg[2] > maxx or seg[3] > maxy)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def search_minorArea(self, delta, step, ejeMenor, ejeMayor):
        
        minx = self.minDimensions[0]
        maxx = self.minDimensions[2]
        
        middleP = minx + ((maxx - minx)/2)
        middleD = delta/2
        
        limInf = middleP - middleD
        limSup = middleP + middleD
        
        ptoCortes = np.arange(limInf, limSup, step) 
        areaCortes = []
        
        minArea = np.inf
        minP = 0
        for p in ptoCortes:
            (areaP, nCutPoints, areaSolida) = self.apply_cutPoint(p, ejeMenor, ejeMayor)
            #print(p, areaP)
            areaCortes.append((p, areaP, nCutPoints, areaSolida))
            if areaP < minArea:
                minArea = areaP
                minP = p
        
        #print(len(ptoCortes))
        print("La menor área encontrada es: %f en el punto de corte %f" % (minArea, minP))   
        self.apply_cutPoint(minP, ejeMenor, ejeMayor, verbose = True)
        return (areaCortes, minP)
        

    def apply_cutPoint(self, xcorte, ejeMenor, ejeMayor, verbose = False):
        miny  = self.minDimensions[1]
        maxy  = self.minDimensions[3]
        cutSeg = [xcorte, miny, maxy] 
        
        cutPoints = []
        cutPoints = self.apply_cutSeg(cutSeg)
                
        if verbose : print("Encontro %d puntos de corte" % (len(cutPoints)) )
        
        if verbose : 
            self.animate_layer2(6, cutPoints, cutSeg)
            #self.animate_layers2(cutPoints, min_layer = 5, max_layer = 8)
        extremePoints = self.elispse_extremePoints(cutPoints, ejeMenor, ejeMayor)
        area_totalSolida = self.calcular_areaTotal_solida(extremePoints)
        if verbose : 
            self.figSolidRectangle(cutPoints)
            #self.plot_ellipses(cutPoints, extremePoints, ejeMenor, ejeMayor)
        if verbose : print('área solida total del corte: ', area_totalSolida)
        
        #minDist_y = self.estimate_spacingY(xcorte)
        minDist_y = 0.33799999999999386
        areaP = self.estimate_proportionalArea(cutPoints, area_totalSolida, minDist_y)
        if verbose : print('área proporcional del corte: ', areaP)
        
        #return 0
        return (areaP, len(cutPoints), area_totalSolida)

    def apply_cutSeg(self, cutSeg):
        cutPoints = []
        for (x0, y0, x1, y1, z) in self.segs:
            if x0 == x1:
               continue
            else :
                if y0 == y1:
                   
                    if (cutSeg[0] >= min(x0, x1)) and (cutSeg[0] <= max(x0, x1)) : #Si el segmento perpendicular corta el segmento de corte
                       cutPoints.append([cutSeg[0], y0, z, x0, x1, y0, y0])
                else:
                    if min(x0, x1) <=  cutSeg[0] and max(x0, x1) >= cutSeg[0] : 
                       mseg = (y1-y0)/(x1-x0)
                       y = mseg*(cutSeg[0] - x0) + y0
                       if (y >= cutSeg[1]) and (y <= cutSeg[2]):
                          cutPoints.append([cutSeg[0], y, z, x0, x1, y0, y1])
        #print("Cantidad de puntos de corte retornando: ", len(cutPoints))
        return cutPoints 

    def estimate_proportionalArea(self, cutPoints, areaSolida, minDist_y, verbose = False):
        y_coords = [point[1] for point in cutPoints]
        z_coords = [point[2] for point in cutPoints]
       
        miny = min(y_coords)
        maxy = max(y_coords)
               
        nPoints_y = round((maxy-miny)/minDist_y)
        nPoints_z = len(np.unique(z_coords))
        nCutPoints = len(cutPoints)
        
        #print("nPoints_z: ", nPoints_z)
        
        nGridPoints = nPoints_y * nPoints_z
        areaEstimada = (areaSolida * nCutPoints)/nGridPoints
        
        if verbose :
           print("Número de celdas de la rejilla: ", nGridPoints)
           print("Número de puntos en el corte: ", nCutPoints)
           print("área sólida: ", areaSolida)
           print("área proporcional: ", areaEstimada)
        
        return min(areaEstimada, areaSolida)

    def elispse_extremePoints(self, cutPoints, ejeMenor, ejeMayor):
        extremePoints = []
        for point in cutPoints:
            x = point[0]
            y = point[1]
            z = point[2]
            extreme_ejeMenor1 = [x, y, z+ejeMenor]
            extreme_ejeMenor2 = [x, y, z-ejeMenor]
            extreme_ejeMayor1 = [x, y+ejeMayor, z]
            extreme_ejeMayor2 = [x, y-ejeMayor, z]
            
            extremePoints.append(extreme_ejeMenor1)
            extremePoints.append(extreme_ejeMenor2)
            extremePoints.append(extreme_ejeMayor1)
            extremePoints.append(extreme_ejeMayor2)
        
        return extremePoints


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def animate_minDimensions(self, layer, minDimensions, animation_time=10):
    
        fig, ax = create_axis(projection='2d')
        xmin, xmax, ymin, ymax, _, _ = self.xyzlimits
        ax.set_xlim(add_margin_to_axis_limits(xmin, xmax))
        ax.set_ylim(add_margin_to_axis_limits(ymin, ymax))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Una sola capa con puntos límites de la probeta")
 
        temp = self.get_layerSegs(layer, layer)
        
        lens = np.array([abs(x0 - x1) + abs(y0 - y1) for x0, y0, x1, y1, z in
                         temp])
        times = lens / lens.sum() * animation_time
        
        
        for time, (x0, y0, x1, y1, z) in zip(times, temp):
            
            ax.plot(x0, y0, 'b.')
            ax.plot(x1, y1, 'b.')
            
            ax.plot([x0, x1], [y0, y1], 'y-')
    
            plt.pause(time)
            plt.draw()

        minx = minDimensions[0]
        miny = minDimensions[1]
        maxx = minDimensions[2]
        maxy = minDimensions[3]
        
        ax.plot(minx, miny, 'g*')
        ax.plot(maxx, maxy, 'm*')
            
        plt.show()
   
    def animate_layer(self, layer, cutSeg, animation_time=10):
       
        fig, ax = create_axis(projection='2d')
        xmin, xmax, ymin, ymax, _, _ = self.xyzlimits
        ax.set_xlim(add_margin_to_axis_limits(xmin, xmax))
        ax.set_ylim(add_margin_to_axis_limits(ymin, ymax))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Concentric Pattern")
 
        temp = self.get_layerSegs(layer, layer)
        
        lens = np.array([abs(x0 - x1) + abs(y0 - y1) for x0, y0, x1, y1, z in
                         temp])
        times = lens / lens.sum() * animation_time
        
        
        for time, (x0, y0, x1, y1, z) in zip(times, temp):
            
            ax.plot(x0, y0, 'b*')
            ax.plot(x1, y1, 'b*')
            
            ax.plot([x0, x1], [y0, y1], 'y-')
    
            plt.pause(time)
            plt.draw()

        if cutSeg :
            ax.plot([cutSeg[0], cutSeg[0]], [cutSeg[1], cutSeg[2]], 'r-') 

        plt.show()   
        
        
    def animate_layer2(self, layer, cutPoints = None, cutSeg = None, animation_time=10):
    
        fig, ax = create_axis(projection='2d')
        #xmin, xmax, ymin, ymax, _, _ = self.xyzlimits
        xmin = 80
        xmax = 140
        ymin = 102
        ymax = 115
        ax.set_xlim(add_margin_to_axis_limits(xmin, xmax))
        print(xmin)
        print(xmax)
        ax.set_ylim(add_margin_to_axis_limits(ymin, ymax))
              
        #ax.set_xlabel('x', fontsize=12)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
        ax.set_title("Line Pattern", fontsize=26)
        
        temp = self.get_layerSegs(self.seg_index[layer], self.seg_index[layer])
        
        lens = np.array([abs(x0 - x1) + abs(y0 - y1) for x0, y0, x1, y1, z in
                         temp])
        times = lens / lens.sum() * animation_time
        
        
        for time, (x0, y0, x1, y1, z) in zip(times, temp):
            
            ax.plot([x0, x1], [y0, y1], 'k-')
    
            plt.pause(time)
            plt.draw()

        if cutSeg :
            ax.plot([cutSeg[0], cutSeg[0]], [cutSeg[1], cutSeg[2]], 'r-') 
        for p in cutPoints:
            if p[2] == self.seg_index[layer]:
               ax.plot(p[0], p[1], 'g*')
               ax.plot([p[3], p[4]], [p[5], p[6]], 'm-')
        
        plt.savefig('probando_fig.png', dpi=400)
        
        
    def animate_layers2(self, cutPoints, min_layer = 0, max_layer = None):
   
        if max_layer == None: max_layer = self.n_layers-1
             
        fig, ax = create_axis(projection='3d')
        xmin, xmax, ymin, ymax, zmin, zmax = self.xyzlimits
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        if zmax > zmin : ax.set_zlim([zmin, zmax])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')
        ax.set_title("Specimen 3D")
         
        layer_segs = self.get_layerSegs(self.seg_index[min_layer], self.seg_index[max_layer])
         
        for (x0, y0, x1, y1, z) in layer_segs:
           #ax.plot(x0, y0, z, 'k.')
           #ax.plot(x1, y1, z, 'k.')
           ax.plot([x0, x1], [y0, y1], [z, z], 'k-')
           #plt.pause(0.00001)
           plt.draw()
           
        x_coords = []
        y_coords = []
        z_coords = []
        
        for p in cutPoints:
            x = p[0]
            y = p[1]
            z = p[2]
            if z >= self.seg_index[min_layer] and z <= self.seg_index[max_layer]:
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                
            
        ax.scatter(x_coords, y_coords, z_coords, color =['red'] ) 
        plt.show()

    
    def plot_ellipses(self, cutPoints, extremePoints, ejeMenor, ejeMayor):
        
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.set_title("Vista de corte")
        
        a=ejeMayor     #radius on the x-axis
        b=ejeMenor    #radius on the y-axis

        #cont = 0
        for point in cutPoints :
            y = point[1]
            z = point[2]
            t = np.linspace(0, 2*np.pi, 100)
            ax.plot(y, z, 'k*')
            plt.plot( y+a*np.cos(t) , z+b*np.sin(t))
            plt.grid(color='lightgray',linestyle='--')
            #break
        
        #y_coords = [point[1] for point in extremePoints]
        #z_coords = [point[2] for point in extremePoints]
        
        #plt.scatter(y_coords, z_coords)
        plt.show()
   

    def figSolidRectangle(self, cutPoints):
        x_coords = []
        y_coords = []
        z_coords = []
        
        for p in cutPoints:
            x_coords.append(p[0])
            y_coords.append(p[1])
            z_coords.append(p[2])
        
        miny = min(y_coords)
        maxy = max(y_coords)
        minz = min(z_coords)
        maxz = max(z_coords)
                
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')
        ax.add_patch(Rectangle((miny, minz), (maxy-miny), (maxz-minz), facecolor = 'k', fill=False))
 
        ax.scatter(y_coords, z_coords, color =['red'], s=5 ) 
        plt.show()
        


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
        
    
                
    
    def calcular_areaTotal_solida(self, extremePoints):
        y_coords = [point[1] for point in extremePoints]
        #z_coords = [point[2] for point in extremePoints]
        miny = min(y_coords)
        minz = min(self.seg_index) #De acuerdo a lo observado en el laboratorio
        maxy = max(y_coords)
        maxz = max(self.seg_index)
        a = maxy - miny
        b = maxz - minz
        return a*b
        
    def estimate_area(self, cutPoints, ejeMenor, ejeMayor):
        areaTotal = np.pi * ejeMenor * ejeMayor * len(cutPoints)
        return areaTotal
    
    def get_cutSeg(self, xcorte):
        n_zCoords = len(self.seg_index)
        mz = int(n_zCoords/2)
        
        mz_idx = self.seg_index[mz]
        
        mz_layerSegs = self.get_layerSegs(mz_idx, mz_idx)
            
        (minx, miny) = self.min_coordxy(mz_layerSegs)
        (maxx, maxy) = self.max_coordxy(mz_layerSegs)
                
        if xcorte < minx or xcorte > maxx :
            print("El punto de corte seleccionado no es correcto")
            return
                
        cutSeg = [xcorte, miny, maxy]
        return cutSeg
    
        
    def eliminate_skirt(self, cutPoints, cutSeg, skirtHeight, verbose = True):
        y_coords = [point[1] for point in cutPoints]
        z_coords = [point[2] for point in cutPoints]
        
        miny = cutSeg[1]
        maxy = cutSeg[2]
        
        skirt_left  =  [index for index, item in enumerate(y_coords) if item < miny]
        skirt_right =  [index for index, item in enumerate(y_coords) if item > maxy]
        skirt_idx = skirt_left + skirt_right
            
        if len(skirt_left) <= skirtHeight :
            for index in sorted(skirt_idx, reverse=True): del cutPoints[index]
    
        if verbose :
           plt.figure()
           ax = plt.gca()
           ax.set_xlabel('y')
           ax.set_ylabel('z')
           ax.set_title("Vista de puntos de corte sin skirt")
           y_coords = [point[1] for point in cutPoints]
           z_coords = [point[2] for point in cutPoints]
           plt.scatter(y_coords, z_coords)
           plt.show()
           
        
        return cutPoints
    
    
    

    def find_min_spacing(self, cutPoints): #Esto debe mejorar
        
        mdist = []
        layers = self.seg_index[-1:]
        for layer in layers:
            print(layer)
            temp = []    
            for p in cutPoints:
                if p[2] == layer:
                  temp.append(p[1])
            if temp:
               mdist.append(self.find_min_distance(temp))
        
        return min(mdist)
        
        
    def find_min_distance(self, lst):
        
        sorted_lst = sorted(set(lst))
        return min([n2 - n1 for n1, n2 in zip(sorted_lst, sorted_lst[1:])])
    
    def apply_cutSeg_layer(self, zcoord, cutSeg):
        #print("Layer: ", zcoord)
        layerSegs = self.get_layerSegs(zcoord, zcoord)
        cutPoints = self.cutPoints_layer(layerSegs, cutSeg, zcoord)
        return cutPoints
        
    
    def get_layerSegs(self, min_layer, max_layer):
        temp = []
        for (x0, y0, x1, y1, z) in self.segs:
            if z >= min_layer and z <= max_layer:
                temp.append((x0, y0, x1, y1, z))
        return temp

    def min_coordxy(self, segList):
        segArray = np.array(segList)
        minx = min(min(segArray[:, 0]), min(segArray[:,2]))
        miny = min(min(segArray[:, 1]), min(segArray[:,3]))
        return (minx, miny)
                 
    
    def max_coordxy(self, segList):
        segArray = np.array(segList)
        maxx = max(max(segArray[:, 0]), max(segArray[:,2]))
        maxy = max(max(segArray[:, 1]), max(segArray[:,3]))
        return (maxx, maxy)

   
# =============================================================================
       

def add_margin_to_axis_limits(min_v, max_v, margin_ratio=MARGIN_RATIO):
    """
    compute new min_v and max_v based on margin

    Args:
        min_v: minimum value
        max_v: maximum value
        margin_ratio:

    Returns:
        new_min_v, new_max_v
    """
    dv = (max_v - min_v) * margin_ratio
    return (min_v - dv, max_v + dv)


def create_axis(figsize=(8, 8), projection='3d'):
      """
      create axis based on figure size and projection
      returns fig, ax

      Args:
          figsize: size of the figure
          projection: dimension of figure

      Returns:
          fig, ax
      """
      projection = projection.lower()
      if projection not in ['2d', '3d']:
          raise ValueError
      if projection == '2d':
          fig, ax = plt.subplots(figsize=figsize)
      else:  # '3d'
          fig = plt.figure(figsize=figsize)
          ax = fig.add_subplot(111, projection='3d')
      return fig, ax

    
def command_line_runner(filename, filetype, ref_file):
    """ command line runner """
   
    delta = 7.62
    step = 0.1
    #Ejes menor y mayor de la elipse que representa el grosor del hilo de 
    #impresión en mm
    ejeMenor = 0.119 * 2
    ejeMayor = 0.191 * 2
        
    gcode_reader = GcodeReader(filename, filetype)
    
    gcode_reader.remove_skirt()
    (cutAreas, minCutPoint) = gcode_reader.search_minorArea(delta, step, ejeMenor, ejeMayor)
    
    

if __name__ == "__main__":
    print("Gcode Reader")
    filetype = GcodeType.FDM_REGULAR
    
    #Probeta compacta tomada como referencia para calcular área proporcional
    refFile = "compactSpecimen.xlsx" 
    
    filename = sys.argv[1]
    print("Input file: ", filename)
    command_line_runner(filename, filetype, refFile)
    
    