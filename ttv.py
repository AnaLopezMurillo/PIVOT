"""
    PIVOT Package - Planetary Interactions: Variations of Timing
"""
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import batman
import emcee
from scipy import stats
import corner
import matplotlib.gridspec as gridspec
import celerite2
from celerite2 import terms
import os
import shutil

# globals
plots_path = ''

#### ---------- Setup Functions ---------- ####

def reset():
    """
        Deletes the plots and csv directories in home directory to prepare for re-run of ttv calculations. 
        
        CANNOT UNDO THIS ACTION. Recommended to only run if completely re-doing the TTV analysis.
    """
     # delete all used diretories (plots)
    plots_path = './plots/'
    csv_path = './csvs/'
    paths = [plots_path, csv_path]

    for path in paths:
        try:
            shutil.rmtree(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception as e:
            print(e)
            print("Folder " + path + " not found! Making folder...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("Folder made!")
    print("Workspace reset.")
    return True

def runmed(a,b,size):
    """Running Median Function: Bins data for plotting.
    
        :param a: time array
        :param b: flux array
        :param size: size of the bin (in days)
        
        :return: Binned time, binned flux data"""
    meda = []
    medb = []
    for i in np.arange(np.min(a),np.max(a),size):
        l = np.where((a > i) & (a<(i+size)))
        meda.append(np.median(a[l]))
        medb.append(np.median(b[l]))
    return meda,medb

def download_lightkurve(system_name, instrument):
    """
        Downloads all lightkurve data for a given system.

        Gives priority to shorter cadence data if available

        :param system_name: Lightkurve-applicable name of the system to download
        :param instrument: "Kepler" or "K2SFF"

        :return: Pandas DataFrame of the requested planet's data
    """
    try:
        search_result_short = lk.search_lightcurve(target=system_name, mission=instrument, cadence='short')
        search_result_long = lk.search_lightcurve(target=system_name, mission=instrument, cadence=1800) # putting stand-in for 'long' bc when putting in long the quarters check was not working

        lcs, lcs_long = search_result_short.download_all(), search_result_long.download_all()
        lcs_total = lcs

        kepler_flag = (instrument == "Kepler")
        k2sff_flag = (instrument == "K2SFF")

        # if there is no short data, save only the long LCs
        if lcs is None:
             lcs_total = lcs_long 

        if kepler_flag and lcs_long is not None and lcs is not None:
            quarters_short = lcs.quarter
            for lc in lcs_long:
                    if lc.quarter not in quarters_short:
                            lcs_total.append(lc)
            lcs_total = sorted(lcs_total, key=lambda x: x.quarter)
        elif k2sff_flag and lcs_long is not None and lcs is not None:
            campaign_short = lcs.campaign
            for lc in lcs_long:
                    if lc.campaign not in campaign_short:
                            lcs_total.append(lc)
            lcs_total = sorted(lcs_total, key=lambda x: x.campaign)

        lcs_total = LightCurveCollection(lcs_total)     # redefine as an LCCOllection after list conversion

        lc = lcs_total.stitch().remove_outliers().normalize()
        lc.scatter()

        target_name = lc.meta['KEPLERID']

        time_array = (np.array(lc.time.value))
        data = (np.array(lc["flux"]))
        err = (np.array(lc['flux_err']))
        df = pd.DataFrame({'time': time_array, 'data': data, 'err': err})

        # save the resulting lightcurve as a csv
        try:
            lc_path = './' + str(target_name) + '/' + str(instrument) + '_lc.csv'
            os.makedirs(os.path.dirname(lc_path), exist_ok=True)
            df.to_csv(lc_path, index=False)
            print(str(instrument) + ' data successfully saved to ' + str(lc_path))
        except Exception as e:
             print("Error ocurred while making lightcurve path!")
        
        print(str(instrument) + ' data downloaded successfully!')
        return df

    except Exception as e:
        print("Error ocurred while trying to find data for planet using " + str(instrument) + ".\n")
        return None


#### ---------- Transit Analysis Functions ---------- ####

def _transit(t, t0, transit_params):
    """Helper for the main TTV analysis code. 
    
    Runs the BATMAN transit model using the given planetary parameters, and returns resultant flux from the model.

    @param t: Time array for the transit
    @param t0: Time of transit midpoint for the transit
    @param transit_params: Tuple of the transit parameters in the style of:

        *per: orbital period (days)*

        *rp: planet radius (in units of stellar radii)*

        *axis: semi-major axis (in units of stellar radii)*

        *inc: orbital incliniation (degrees)*

        *ecc: eccentricity*

        *w: longitude of periastron (degrees)*

        *u1, u2: limb darkening coefficients. Please make sure to pass in correct coefficients for either TESS or Kepler/K2 bands.*
    
    """
    per, rp, axis, inc, ecc, w, u1, u2 = transit_params
    params = batman.TransitParams()
    params.t0 = t0                 #time of inferior conjunction
    params.per = per                 #orbital period  FIXED
    params.rp = rp                     #planet radius (in units of stellar radii)
    params.a = axis                     #semi-major axis (in units of stellar radii)
    params.inc = inc                     #orbital inclination (in degrees)
    params.ecc = ecc                      #eccentricity
    params.w = w                       #longitude of periastron (in degrees)
    params.u = [u1, u2]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, t)    #initializes model
    flux = m.light_curve(params)          #calculates light curve
    return flux 

# MCMC functions
def set_params(theta, time, gp, err):
    t0, logsigma, logrho, logQ = theta
    gp.kernel = terms.SHOTerm(sigma=np.exp(logsigma), rho=np.exp(logrho), Q=np.exp(logQ)) 
    gp.compute(time, yerr=err, quiet=True)
    return gp

def ln_likelihood(theta, time, gp, data, err, transit_params):
    t0, logsigma, logrho, logQ = theta
    try: 
        gp = set_params(theta, time, gp, err)
    except ZeroDivisionError:
        return -np.inf
    model = _transit(time, t0, transit_params)
    return gp.log_likelihood(data-model)

## let's add some weak priors to stop the parameters from going wild
def ln_prior(theta, t0_guess, t0_explore):
    """
        t0_explore: float saying how large to set the parameter space exploration of the planet
    """
    # TO-DO 2/14/25: make some sort of vetting to see if the window is too large (np.max-np.min) on t0 prior? have to restart the chain probably if we do this though...
    t0, logsigma, logrho, logQ = theta
    sigma = np.exp(logsigma)
    rho = np.exp(logrho)
    Q = np.exp(logQ)
    if np.abs(t0-t0_guess)>t0_explore or (Q == 0) or (rho == 0) or (sigma == 0):
        return -np.inf
    sigma_sigma2 = 20.0
    lp1 = -0.5*np.sum((sigma)**2/sigma_sigma2 + np.log(2*np.pi*sigma_sigma2))
    sigma_q2 = 20.0
    lp3 = -0.5*np.sum((Q)**2/sigma_q2 + np.log(2*np.pi*sigma_q2))
    return lp1+lp3

def ln_probability(theta, time, gp, data, err, t0_guess, transit_params, t0_explore):
    lp = ln_prior(theta, t0_guess, t0_explore)
    if not np.isfinite(lp):
        return -np.inf
    lnlike = ln_likelihood(theta, time,gp,data,err, transit_params)
    if not np.isfinite(lnlike):
        return -np.inf
    return lp + lnlike

def ttv_algo(time_array, data, yerr, planet, instrument, window_mult=2, flag_bounds=None):
    '''
    Main algorithm of TTV package. Iterates through the time dataset and fits each transit iteratively using BATMAN and a Gaussian Process MCMC

    :param planet: Dictionary object with all planetary parameters in following format. Only either `u1, u2` or `u1_k2, u2_k2` are necessary if only running analysis using one instrument.

        ```planet = 
            {
            "planet_name": '',  # string
            "per": ,            # Period (days)
            "rp": ,             # Rp/R*
            "axis": ,           # Axis
            "t0": ,             # Time of Conjunction (days)
            "inc": ,            # Inclinication (degrees)
            "ecc": ,            # Eccentricity
            "w": ,              # Argument of periastron
            "td": ,             # Transit Duration (days)
            "u1": ,             # TESS Limb-darkening parameter 1
            "u2": ,             # TESS Limb-darkening parameter 2
            "u1_k2": ,          # K2/Kepler Limb-darkening parameter 1
            "u2_k2": ,          # K2/Kepler Limb-darkening parameter 2
            "t0_explore": ,     # Float detailing how large to set the t0 parameter exploration (days)
            }
        ```
    :param instrument: string of either 'K2', 'Kepler', or 'TESS'. Determines which limb-darkening parameters are used in the transit BATMAN fit
    :param window_mult: optional parameter detailing how large to make the fit window. Default is 2 times the transit duration.
    :param flag_bounds: optional parameter detailing what time bounds to skip (ex. if break in data because of TESS data dump)
    :return: None
    '''
    if instrument == 'TESS':
            u1, u2 = planet["u1"], planet["u2"]
            t0_planet = planet["t0"] - 2457000
            print(t0_planet)
    elif instrument == 'K2' or 'Kepler':
            u1, u2 = planet["u1_k2"], planet["u2_k2"]
            t0_planet = planet["t0"] - 2454833
    else: 
         raise ValueError("Instrument parameter is not 'TESS', 'K2', or 'Kepler.")

    t0, per, td, t0_explore, planet_name = t0_planet, planet["per"], planet["td"], planet["t0_explore"], planet["planet_name"]
    transit_params = planet["per"], planet["rp"], planet["axis"], planet["inc"], planet["ecc"], planet["w"], u1, u2

    oc = []
    tt_exp = []
    tt_obs = []
    tt_obs_lower = []
    tt_obs_upper = []

    model_gp = []
    mu_gp = []

    terr = []
    chi2 = []
    transit_num = 0
    rho = 1.4

    for i in np.arange(-10000, 10000, 1):
        t0_guess = t0+(per*i)
                  # t0  #logsigma #logrho  #logQ
        initial = [t0_guess, -4,  np.log(rho),  0.7]

        window = window_mult*td
        ll = np.where((time_array > t0_guess - window) & (time_array < t0_guess + window))
        ll2 = np.where((time_array > t0_guess - 0.1) & (time_array < t0_guess + 0.1))
        
        if (flag_bounds is not None) and (t0_guess in flag_bounds):
            continue

        if (np.size(ll[0]) > 0 and np.size(ll2[0]) > 1): 
            y = np.array(data[ll[0]])
            t = np.array(time_array[ll[0]])
            err = None

            print(type(yerr))

            if not (isinstance(yerr, float)):
                 err = yerr[ll[0]]
            else:
                err = yerr

            transit_num+=1

            # Non-periodic component
            term2 = terms.SHOTerm(sigma=np.exp(initial[1]), rho=np.exp(initial[2]), Q=np.exp(initial[3]))
            ## just use the quasi-periodic one
            kernel = term2

            # Setup the GP
            gp = celerite2.GaussianProcess(kernel, mean=0.0)
            gp.compute(t, yerr=err)
            print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))

            ln_probability(initial,t,gp,y,err, t0_guess, transit_params, t0_explore)

            # set up MCMC sampler
            max_n = 15000
            nwalkers = 30
            initial_positions = initial + 0.001 * np.random.randn(nwalkers, len(initial))
            sampler = emcee.EnsembleSampler(nwalkers, len(initial), ln_probability, args=(t, gp, y, err, t0_guess, transit_params, t0_explore), threads=8)

            # We'll track how the average autocorrelation time estimate changes
            index = 0
            autocorr = np.empty(max_n)
            autcorrreq = 50
            autcorr_change_frac = 0.1
            old_tau = np.inf

            for sample in sampler.sample(initial_positions, iterations=max_n, progress=True):
                # Check convergence every 1000 steps
                if sampler.iteration % 1000:
                    continue
                
                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * autcorrreq < sampler.iteration)
                converged &= np.all(np.abs((old_tau - tau) / tau) < autcorr_change_frac)
                if converged:
                    print(np.max(tau*autcorrreq)//1,sampler.iteration,(100*np.max(np.abs(old_tau - tau) / tau))//1 )
                    print(" ")
                    break
                else:
                    print(np.max(tau*autcorrreq)//1,sampler.iteration,(100*np.max(np.abs(old_tau - tau) / tau))//1 )
                old_tau = tau

            ## burn and flat samples finding
            burn = sampler.iteration//5
            flat_samples = sampler.get_chain(discard=burn, thin=3, flat=True)
            mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])

            samples = sampler.get_chain(discard=burn)
            flats = np.concatenate(samples)

            # append the fit values 
            result = np.median(flat_samples,axis=0)
            set_params(result, t, gp, err)
            model = _transit(t,result[0], transit_params)
            actual = _transit(t, t0_guess, transit_params)
            mu, variance = gp.predict(y-model, t, return_var=True)
            sigma = np.sqrt(variance)
            tt_exp.append(t0_guess)
            tt_obs.append(mcmc[1])
            tt_obs_lower.append(mcmc[1]-mcmc[0])
            tt_obs_upper.append(mcmc[2]-mcmc[1])
            oc.append((result[0]-t0_guess)*24*60)
            terr.append(24*60*(np.percentile(flats[:,0], 84) - np.percentile(flats[:,0], 16))/2)

            print((result[0]-t0_guess)*24*60, 24*60*np.std(flats[:,0]), 24*60*(np.percentile(flats[:,0], 84) - np.percentile(flats[:,0], 16))/2)
            print("chi2: " + str(chi2))

            ## save plots to folder, redefine plots path
            plots_path = './plots/' + str(planet_name) + '/' + str(instrument) + '/' + str(planet_name) + '_' + str(transit_num) + '/' + str(transit_num)

            # walker plot
            ndim = 4
            labels = [r'$T_0$',r'$\ln{\sigma}$',r'$\ln{\rho}$',r'$\ln{Q}$']
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True, facecolor="white")
            for i in [0, 1, 2, 3]:
                ax = axes[i] 
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            save_plot(plots_path + '_' + str(round(oc[len(oc)-1], 2)) + '_walker.png')
            plt.show()

            # corner plot
            flat_samples = sampler.get_chain(discard=burn, flat=True)
            fig = corner.corner( 
                flat_samples,show_titles=True,labels=labels,quantiles=(0.16, 0.84),
                fill_contours=True, plot_datapoints=False,title_kwargs={"fontsize": 9},title_fmt='.3f',
                hist_kwargs={"linewidth": 2.5},levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],
                title_quantiles=[0.16, 0.5, 0.84]
            )
            save_plot(plots_path + '_' + str(round(oc[len(oc)-1], 2)) + '_corner.png')    # CAMBIAR A USAR EL RCHI2 EN EL TITULO PORFIS
            plt.show()

            # data model plot
            fontsize = 15
            scatter_size = 40
            fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(14, 6), gridspec_kw={'height_ratios':[3,1]})
            fig.subplots_adjust(wspace=0, hspace=0)
            print('Transit num: ' + str(transit_num))
            marker = "."
            if (np.size(ll) < 2500):
                marker="o"
            ax1.scatter(t,y-mu, marker=marker,alpha=0.5, label='Data', s=scatter_size)

            if np.size(ll) > 50:
                bint,binfl = runmed(t,y-mu,0.33/24.)
                bint_err, binfl_err = runmed(t, y-(model+mu), 0.33/24.)
                ax1.scatter(bint, binfl,marker='o',alpha=0.7,color='r', label='Binned Data', s=scatter_size, zorder=2)
                ax2.scatter(bint_err, binfl_err, marker='o', alpha=0.7, color='r', s=scatter_size, zorder=2)

            ax1.plot(t,model,color='black', label='Model', zorder=3, linewidth=2)
            ax1.plot(t,actual,color='r', zorder=4, alpha=0.7, linewidth=2, linestyle='dashed', label='Without TTV')
            ax1.set_ylabel('Normalized Flux', fontsize=fontsize)
            # plotting the observed - predicted 
            # ax2.errorbar(t, y-(model+mu), yerr=sigma, fmt='o', ms=6, elinewidth=1., alpha=0.5, zorder=1)
            ax2.scatter(t, y-(model+mu), marker='o', alpha=0.5, zorder=1, s=scatter_size)
            ax2.axhline(0, color='black', linestyle='--', zorder=3)
            ax2.set_ylabel('Residuals', fontsize=fontsize)
            ax2.set_xlabel('Time (days)', fontsize=fontsize)
            ax1.legend(fontsize=fontsize)
            ax1.tick_params(labelsize=(fontsize)-2, length=10)
            ax2.tick_params(labelsize=(fontsize)-2, length=10)
            plt.tight_layout
            save_plot(plots_path + '_' + str(round(oc[len(oc)-1], 2)) + '_oc.png')
            # plt.savefig(plots_path + '_oc.png')
            plt.show()

            model_gp.append(model)
            mu_gp.append(mu)

            ## save to csv as we go
            df = pd.DataFrame({'time': tt_exp, 'oc': oc, 'err': terr})
            df.to_csv('csvs/' + str(planet_name) + str(instrument) +'_test.csv', index=False)

            ## save the o-c diagram as we go
            oc_path = './plots/' + str(planet_name) + '_' + str(instrument)
            plot_ttv(tt_exp, oc, terr, path=oc_path)
            
    return df

def fix_trendline(tt, oc):
    """Computes the trendline for the O-C results using numpy.polyfit()

    :param tt: Transit midpoint times for planet
    :param oc: O-C results (in minutes) for planet
    
    :return: 
        Residuals: Array of points fixed along a y=mx+b trendline
        [m, b]: slope and y-intercept values from the trendline \n
        [err_m, err_b]: Errors of the m and b values from the residuals of the least-squares fit given using `np.polyfit`
    """
    fit = np.polyfit(tt, oc, 1, full=False, cov=True)
    coeff, var = fit    # var is the covariance matrix
    m, b  = coeff[0], coeff[1]
    err_m, err_b = np.sqrt(var[0][0]), np.sqrt(var[1][1]) # convert to standard deviation --> standard dev is sqrt(var)
    new_oc = [m*t + b for t in tt]
    residuals = oc - new_oc
    return (residuals, [m, b], [err_m, err_b])
     

def square(list):
    """
        Helper function for rchi2 calculation: squares each element in the list
    """
    return [j ** 2 for j in list]

def divide_list(l1, l2):
    res = []
    for i in range(len(l1)):
        res.append(l1[i]/l2[i])
    return res

def rchi2(oc, err):
    """ 
        Returns the rchi^2 of the O-C results for a given planet.
    """
    return np.sum(divide_list(square(oc), square(err))) / (len(oc) - 1)

#### ---------- Plotting Helper Functions ---------- ####

def plot_ttv(tt, oc, err, path=None):
        """ 
            Plots the O-C results over time for a planet.
        """
        if path is not None:
            plots_path = path
        else:
             plots_path = './'
        plt.figure(facecolor="white", figsize=(10, 4))
        plt.errorbar(tt,oc,yerr=err, fmt='o', mfc='white', ms=10, elinewidth=1., zorder=2)
        plt.axhline(y=0.0, linestyle='--', color='black', zorder=1)
        plt.scatter(tt,oc, color='b')
        plt.ylabel('O-C (minutes)')
        plt.xlabel('Time (days)')
        plt.title('O-C Diagram')
        save_plot(plots_path + '_OC.png')


def save_plot(filepath, figure=None):
    """
        Helper function to save a given plot for a certain filepath. Used for placing plots into correct directories.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if figure is None:
        plt.savefig(filepath)
    else:
        figure.savefig(filepath)