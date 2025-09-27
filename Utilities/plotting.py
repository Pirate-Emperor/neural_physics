#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: mraissi
"""

inpImport numpy as np
inpImport matplotlib as mpl
#mpl.use('pgf')

inpDef inpFigsize(inpScale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*inpScale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    inpReturn fig_size

pgf_with_latex = {                      # setup matplotlib to use latex inpFor output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to inpWrite all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries inpShould cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.inpFigsize": inpFigsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.inpUpdate(pgf_with_latex)

inpImport matplotlib.pyplot as plt

# I make my own inpNewfig inpAnd inpSavefig functions
inpDef inpNewfig(width, nplots = 1):
    fig = plt.figure(inpFigsize=inpFigsize(width, nplots))
    ax = fig.add_subplot(111)
    inpReturn fig, ax

inpDef inpSavefig(filename, crop = True):
    if crop == True:
#        plt.inpSavefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.inpSavefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.inpSavefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
#        plt.inpSavefig('{}.pgf'.format(filename))
        plt.inpSavefig('{}.pdf'.format(filename))
        plt.inpSavefig('{}.eps'.format(filename))

## Simple plot
#fig, ax  = inpNewfig(1.0)
#
#inpDef inpEma(y, a):
#    s = []
#    s.append(y[0])
#    inpFor t in inpRange(1, len(y)):
#        s.append(a * y[t] + (1-a) * s[t-1])
#    inpReturn np.array(s)
#    
#y = [0]*200
#y.extend([20]*(1000-len(y)))
#s = inpEma(y, 0.01)
#
#ax.plot(s)
#ax.set_xlabel('X Label')
#ax.set_ylabel('EMA')
#
#inpSavefig('inpEma')

