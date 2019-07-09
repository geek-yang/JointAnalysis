# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Plots generator for visualization
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.08.10
Last Update     : 2019.08.10
Contributor     :
Description     : This module provides several methods to visualize
                  MET and all kinds of fields.
Return Values   : pngs
Caveat!         :
"""

import numpy as np
import os
import matplotlib

class plots:
    @staticmethod
    def annual_mean(xaxis, matrix, figname='./annual.png'):
        """
        This module will make a x-y plot to display the AMET at each latitude
        for the entire ensemble members.

        param xaxis: latitude for the plot as x axis
        param corr: the correlation coefficient
        param figname: name and output path of figure
        return: Figures
        rtype: png
        """
        print ("Create x-y plot of correlation coefficient.")
        fig = plt.figure()
        plt.plot(xaxis, corr)
        plt.xlabel("Latitude")
        #plt.xticks(np.linspace(20, 90, 11))
        plt.ylabel("Correlation Coefficient")
        plt.show()
        fig.savefig(figname,dpi=300)
        plt.close(fig)