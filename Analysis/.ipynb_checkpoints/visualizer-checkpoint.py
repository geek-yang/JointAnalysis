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
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import iris
import iris.plot as iplt
import cartopy
import cartopy.crs as ccrs

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

    @staticmethod
    def geograph(latitude, longitude, field, label, ticks,
                 figname='./NorthPolar.png', gridtype = 'geographical',
                 boundary='northhem'):
        """
        This module will make a geographical plot to give a spatial view of fields.
        This module is built on iris and cartopy for the visualization of fields on
        both geographical and curvilinear grid.
        param lat: latitude coordinate for plot
        param lon: longitude coordinate for plot
        param field: input field for visualization
        param p_value: the significance level from t test. The significance level is set to be 99.5%.
        param gridtype: type of input spatial fields, it has two options
        - geographical (default) the coordinate is geographical, normally applied to atmosphere reanalysis
        - curvilinear the coordinate is curvilinear, normally applied to ocean reanalysis
        param figname: name and output path of figure
        param boundary: region for plot. It determines the boundary of plot area (lat,lon) and projection.
        - northhem (default) plot the north hemisphere from 20N-90N & 180W-180E, with the projection NorthPolarStereo.
        - atlantic plot the north Atlantic from 20N-90N & 90W-40E, with the projection PlateCarree
        return: figures
        rtype: png
        """
        print ("Create a NorthPolarStereo view of input fields.")
        if gridtype == 'geographical':
            print ("The input fields are originally on geographical grid")
            # first construct iris coordinate
            lat_iris = iris.coords.DimCoord(latitude, standard_name='latitude', long_name='latitude',
                                            var_name='lat', units='degrees')
            lon_iris = iris.coords.DimCoord(longitude, standard_name='longitude', long_name='longitude',
                                            var_name='lon', units='degrees')
            # assembly the cube
            cube_iris = iris.cube.Cube(field, long_name='geographical field', var_name='field',
                                       units='1', dim_coords_and_dims=[(lat_iris, 0), (lon_iris, 1)])
            if boundary == 'polar':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                #ax.set_extent([-180,180,20,90],ccrs.PlateCarree())
                ax.set_extent([-180,180,60,90],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                cs = iplt.contourf(cube_iris, cmap='coolwarm',levels=ticks, extend='both') #, vmin=ticks[0], vmax=ticks[-1]
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
                
            elif boundary == 'northhem':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180,180,20,90],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                cs = iplt.contourf(cube_iris, cmap='coolwarm',levels=ticks, extend='both') #, vmin=ticks[0], vmax=ticks[-1]
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)                
            elif boundary == 'atlantic':
                fig = plt.figure(figsize=(8,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([-90,40,20,85],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_iris,cmap='coolwarm',levels=ticks, extend='both')
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05, format="%.1f")
                cbar.set_label(label,size = 11)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 11)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            else:
                print ('This boundary is not supported by the module. Please check the documentation.')