# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Statistical Operator for Climate Data
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.08.10
Last Update     : 2019.08.10
Contributor     :
Description     : This module provides several methods to perform statistical
                  analysis on MET and all kinds of fields.
Return Values   : numpy arrays
Caveat!         :
"""

import numpy as np
import scipy
from scipy import stats
import os
import iris

class statistics:
    def __init__(self, var):
        """
        Statistical operations on climate data.
        param var: imput time series
        param outpath: the path for the output files
        """
        print("Input array should have dimensions (ensemble,year,month,lat)")
        self.var = var

    def anomaly(self, Dim_ens=True):
        """
        Remove seasonal cycling for monthly data.
        param Dim_ens: there are two modes for removing the seasonal cycling
        -True (default) input time series have ensemble dimension [ensemble,year,month,...]
        -False input time series do not have ensemble dimension
        param white_var: time series without seasonal cycling
        return: time series
        rtype: numpy.array
        """
        # white refers to the time series without seasonal cycling
        white_var = np.zeros(self.var.shape, dtype=float)

        #switch mode
        if Dim_ens == True:
            print('Ensemble member should always be the first dimension!')
            # check the dimension of input
            if self.var.ndim == 3:
                seansonal_cycle_var = np.mean(self.var, axis=1)
                e, t, m = white_var.shape
                for i in np.arange(t):
                        white_var[:,i,:] = self.var[:,i,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(e,t*m)
            elif self.var.ndim == 4:
                seansonal_cycle_var = np.mean(self.var, axis=1)
                e, t, m, y = white_var.shape
                for i in np.arange(t):
                        white_var[:,i,:,:] = self.var[:,i,:,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(e,t*m,y)
            else:
                raise IOError("This module can not work with any array with a \
                              dimension other than 3 or 4!")
        else:
            print ('The input data does not have the dimension of ensemble.')
            if self.var.ndim == 2:
                seansonal_cycle_var = np.mean(self.var, axis=0)
                t, m = white_var.shape
                for i in np.arange(t):
                        white_var[i,:] = self.var[i,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(t*m)
            elif self.var.ndim == 3:
                seansonal_cycle_var = np.mean(self.var, axis=0)
                t, m, y = white_var.shape
                for i in np.arange(t):
                        white_var[i,:,:] = self.var[i,:,:] - seansonal_cycle_var[:]
                # re-arrange into single time series - without month dimension
                white_var = white_var.reshape(t*m,y)
            else:
                raise IOError("This module can not work with any array with a \
                              dimension other than 2 or 3!")
        self._anomaly = white_var

        print ("The output anomaly time series only contains one dimension for time!")

        return self._anomaly

    def detrend(self, order=2, obj='anomaly', Dim_ens=True):
        """
        Detrend time series through polynomial fit.
        param series: input time series, either 1D or 2/3D
        param order: order of polynomial for fitting
        param obj: objects for detrending, two options available
        -'anomaly' (default) the time series of anomaly will be detrended
        -'original' the original input time series will be detrended
        return: time series
        rtype: numpy.array
        """
        if obj == 'anomaly':
            series = self._anomaly
        elif obj == 'original':
            print ("Make sure that the input time series has only 1 dimension for time!")
            series = self.var
        else:
            raise IOError("Please choose the right input mode for detrending!")
        # check the dimension of input
        if Dim_ens == True:
            print('Ensemble member should always be the first dimension!')
            # check the dimension of input
            if series.ndim == 2:
                poly_fit_var = np.zeros(series.shape, dtype=float)
                e, t = poly_fit_var.shape
                for i in np.arange(e):
                    polynomial = np.polyfit(np.arange(t), series[i,:], order)
                    poly_fit = np.poly1d(polynomial)
                    poly_fit_var[i,:] = poly_fit(np.arange(t))
            elif series.ndim == 3:
                poly_fit_var = np.zeros(series.shape, dtype=float)
                e, t, y = poly_fit_var.shape
                for i in np.arange(e):
                    for j in np.arange(y):
                        polynomial = np.polyfit(np.arange(t), series[i,:,j], order)
                        poly_fit = np.poly1d(polynomial)
                        poly_fit_var[i,:,j] = poly_fit(np.arange(t))
            else:
                raise IOError("This module can not work with any array with a \
                                dimension other than 2 or 3!")
        else:
            if series.ndim == 1:
                polynomial = np.polyfit(np.arange(len(series)), series, order)
                poly_fit = np.poly1d(polynomial)
                poly_fit_var = poly_fit(np.arange(len(series)))
            elif series.ndim == 2:
                poly_fit_var = np.zeros(series.shape, dtype=float)
                t, y = poly_fit_var.shape
                for i in np.arange(y):
                    polynomial = np.polyfit(np.arange(t), series[:,i], order)
                    poly_fit = np.poly1d(polynomial)
                    poly_fit_var[:,i] = poly_fit(np.arange(t))
            else:
                raise IOError("This module can not work with any array with a \
                              dimension other than 1 or 2!")                       
        self._polyfit = poly_fit_var
        self._detrend = series - self._polyfit

        return self._detrend
    
    def trend(self,obj='anomaly', Dim_ens=True):
        """
        Compute the trend for the given time series through least square fit.
        param series: input time series, either 1D or 2/3D
        param obj: objects for detrending, two options available
        -'anomaly' (default) the time series of anomaly will be detrended
        -'original' the original input time series will be detrended
        return: slope/linear trend
        rtype: numpy.array
        """
        if obj == 'anomaly':
            series = self._anomaly
        elif obj == 'original':
            print ("Make sure that the input time series has only 1 dimension for time!")
            series = self.var
        else:
            raise IOError("Please choose the right input mode for calculating the linear trend!")
        if Dim_ens == True:
            print('Ensemble member should always be the first dimension!')
            # check the dimension of input
            if series.ndim == 2:
                e, t = series.shape
                # create an array to store the slope coefficient and residual
                a = np.zeros(e,dtype = float)
                b = np.zeros(e,dtype = float)
                A = np.vstack([np.arange(t),np.ones(t)]).T
                for i in np.arange(e):
                    a[i], b[i] = np.linalg.lstsq(A,series[i,:])[0]
            elif series.ndim == 3:
                e, t, y = series.shape
                a = np.zeros((e,y),dtype = float)
                b = np.zeros((e,y),dtype = float)
                A = np.vstack([np.arange(t),np.ones(t)]).T
                for i in np.arange(e):
                    for j in np.arange(y):
                        a[i,j], b[i,j] = np.linalg.lstsq(A,series[i,:,j])[0]
            else:
                raise IOError("This module can not work with any array with a \
                               dimension other than 2 or 3!")
        else:
            if series.ndim == 1:
                t = len(series)
                # the least square fit equation is y = ax + b
                # np.lstsq solves the equation ax=b, a & b are the input
                # thus the input file should be reformed for the function
                # we can rewrite the line y = Ap, with A = [x,1] and p = [[a],[b]]
                A = np.vstack([np.arange(t),np.ones(t)]).T
                # start the least square fitting
                # return value: coefficient matrix a and b, where a is the slope
                a, b = np.linalg.lstsq(A,series)[0]
            elif series.ndim == 2:
                t, y = series.shape
                a = np.zeros((y),dtype = float)
                b = np.zeros((y),dtype = float)
                A = np.vstack([np.arange(t),np.ones(t)]).T
                for i in np.arange(y):
                    a[i], b[i] = np.linalg.lstsq(A,series[:,i])[0]
            else:
                raise IOError("This module can not work with any array with a \
                              dimension other than 1 or 2!")
                
        self._a = a
        return self._a

    def lowpass(self, window=60, obj='anomaly', Dim_ens=True):
        """
        Apply low pass filter to the time series. The function gives running mean
        for the point AT The End Of The Window!!
        param series: input time series, either 1D or 2/3D
        param window: time span for the running mean
        param obj: object for detrending, two options available
        -'anomaly' (default) apply low pass filter to the time series of anomaly
        -'original' apply lowpass filter to the original input time series
        -'detrend' apply lowpass filter to the detrended time series
        return: time series
        rtype: numpy.array
        """
        if obj == 'anomaly':
            series = self._anomaly
        elif obj == 'original':
            series = self.var
        elif obj == 'detrend':
            series = self._detrend
        # check the dimension of input
        if Dim_ens == True:
            print('Ensemble member should always be the first dimension!')
            # check the dimension of input
            if series.ndim == 2:
                e, t  = series.shape
                running_mean = np.zeros((e, t-window+1), dtype=float)
                for i in np.arange(t-window+1):
                    running_mean[:,i] = np.mean(series[:,i:i+window],1)
            elif series.ndim == 3:
                e, t, y = series.shape
                running_mean = np.zeros((e, t-window+1, y), dtype=float)
                for i in np.arange(t-window+1):
                    running_mean[:,i,:] = np.mean(series[:,i:i+window,:],1)
            else:
                raise IOError("This module can not work with any array with a \
                               dimension other than 2 or 3!")
        else:
            if series.ndim == 1:
                t = len(series)
                running_mean = np.zeros(t-window+1, dtype=float)
                for i in np.arange(t-window+1):
                    running_mean[i] = np.mean(series[i:i+window])
            elif series.ndim == 2:
                t, y = series.shape
                running_mean = np.zeros((t-window+1, y), dtype=float)
                for i in np.arange(t-window+1):
                    running_mean[i,:] = np.mean(series[i:i+window,:],1)
            else:
                raise IOError("This module can not work with any array with a \
                              dimension other than 1 or 2!")
                
        self._lowpass = running_mean

        return self._lowpass

    @staticmethod
    def seasons(series, span='DJF', Dim_month=False):
        """
        Extract time series for certain months from given series.
        The given time series should include the time series of all seasons, starting
        from January to December.
        The module extracts 3 month per year based on given argument to incoorporate
        with lead / lag regressions with following modules.
        param series: input time series containing the data for all seasons.
        param span: Target months for data extraction. Following options are available:
        - DJF (default) December, January and February (winter)
        - JJA June, July, August (summer)
        - NDJ November, December and January
        - OND October, November, December
        - SON September, October, November (autumn)
        - MJJ May, June, July
        - AMJ April, May, June
        - MAM March, April, May (spring)
        param Dim_month: A check whether the time series include the dimension of month.
        return: time series
        rtype: numpy.array
        """
        # check if the input time is in the pre-defined month list
        month_list = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                      'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        if span not in month_list:
            raise IOError("The input month span does not include 3 contineous calander months!")
        # rearange the input series
        if Dim_month == True:
            if series.ndim == 2:
                t, m = series.shape
                series = series.reshape(t*m)
            elif series.ndim == 3:
                t, m, y = series.shape
                series = series.reshape(t*m, y)
            elif series.ndim == 4:
                t, m, y, x = series.shape
                series = series.reshape(t*m, y, x)
            else:
                raise IOError("This module can not work with any array with a \
                              dimension higher than 4!")
        else:
            pass
        # select the months for extraction
        month_1 = 0
        # months combinations except 'DJF' 'NDJ'
        if span == 'JJA':
            month_1 = 6
        elif span == 'OND':
            month_1 = 10
        elif span == 'SON':
            month_1 = 9
        elif span == 'ASO':
            month_1 = 8
        elif span == 'JAS':
            month_1 = 7
        elif span == 'MJJ':
            month_1 = 5
        elif span == 'AMJ':
            month_1 = 4
        elif span == 'MAM':
            month_1 = 3
        elif span == 'FMA':
            month_1 = 2
        elif span == 'JFM':
            month_1 = 1
        month_2 = month_1 + 1
        month_3 = month_1 + 2
        # now we deal with the exception
        if span == 'DJF':
            month_1 = 1
            month_2 = 2
            month_3 = 12
        elif span == 'NDJ':
            month_1 = 1
            month_2 = 11
            month_3 = 12
        # seperate summer and winter from the rest of the months
        if series.ndim == 1:
            t = len(series)
            series_season = np.zeros(t//4,dtype=float)
            series_season[0::3] = series[month_1-1::12]
            series_season[1::3] = series[month_2-1::12]
            series_season[2::3] = series[month_3-1::12]
        elif series.ndim == 2:
            t, y = series.shape
            series_season = np.zeros((t//4,y),dtype=float)
            series_season[0::3,:] = series[month_1-1::12,:]
            series_season[1::3,:] = series[month_2-1::12,:]
            series_season[2::3,:] = series[month_3-1::12,:]
        elif series.ndim == 3:
            t, y, x = series.shape
            series_season = np.zeros((t//4,y,x),dtype=float)
            series_season[0::3,:,:] = series[month_1-1::12,:,:]
            series_season[1::3,:,:] = series[month_2-1::12,:,:]
            series_season[2::3,:,:] = series[month_3-1::12,:,:]
        else:
            raise IOError("This module can not work with any array with a \
                           dimension higher than 3!")
        return series_season

class spatial:
    def __init__(self, var):
        """
        Statistical operations on climate data.
        param var: imput time series
        param outpath: the path for the output files
        """
        print("Input array should have dimensions (year,month,lat,lon)")
        self.var = var

    def anomaly(self):
        """
        Remove seasonal cycling for monthly data.
        param Dim_ens: there are two modes for removing the seasonal cycling
        -True (default) input time series have ensemble dimension [ensemble,year,month,...]
        -False input time series do not have ensemble dimension
        param white_var: time series without seasonal cycling
        return: time series
        rtype: numpy.array
        """
        # white refers to the time series without seasonal cycling
        white_var = np.zeros(self.var.shape, dtype=float)

        #switch mode
        print ('The input data does not have the dimension of ensemble.')
        if self.var.ndim == 4:
            seansonal_cycle_var = np.mean(self.var, axis=0)
            t, m, y, x = white_var.shape
            for i in np.arange(t):
                white_var[i,:,:,:] = self.var[i,:,:,:] - seansonal_cycle_var[:]
            # re-arrange into single time series - without month dimension
            white_var = white_var.reshape(t*m,y,x)
        else:
            raise IOError("This module can only work with an array with a \
                              dimension [year,month,lat,lon]")
        self._anomaly = white_var

        return self._anomaly

    def trend(self,obj='anomaly'):
        """
        Compute the trend for the given time series through least square fit.
        param series: input time series (time,lat,lon)
        param obj: objects for detrending, two options available
        -'anomaly' (default) the time series of anomaly will be detrended
        -'original' the original input time series will be detrended
        return: slope/linear trend
        rtype: numpy.array
        """
        if obj == 'anomaly':
            series = self._anomaly
        elif obj == 'original':
            print ("Make sure that the input time series has only 1 dimension for time!")
            series = self.var
        else:
            raise IOError("Please choose the right input mode for calculating the linear trend!")
            # check the dimension of input
        if series.ndim == 3:
            t, y, x = series.shape
            a = np.zeros((y,x),dtype = float)
            b = np.zeros((y,x),dtype = float)
            A = np.vstack([np.arange(t),np.ones(t)]).T
            for i in np.arange(y):
                for j in np.arange(x):
                    a[i,j], b[i,j] = np.linalg.lstsq(A,series[:,i,j])[0]
        else:
            raise IOError("This module can not work with an array with a \
                           dimension [time,lat,lon]!")
                
        self._a = a
        return self._a