import numpy as np
from typing import List
import pandas as pd
from datetime import datetime, timedelta
from functions_general import *

def SEIR(start_org_date,
                  start_simu_date,
                  end_date,
                  i0: float,
                  R0: float,
                  eps: float,
                  mu: float,
                  ifr: List[float],
                  C: List[float],
                  Delta: int,
                  daily_steps: int,
                  basin: str):
    """
    SEIR model with mobility data used to modulate the force of infection.
    Parameters
    ----------
        @param i0 (int): initial number of infected
        @param r0 (int): initial number of recovered
        @param Nk (List[float]): number of individuals in different groups
        @param T (int): simulation steps
        @param R0 (float): basic reproductive number
        @param eps (float): inverse of latent period
        @param mu (float): inverse of infectious period
        @param ifr (List[float]): infection fatality rate by groups
        @param C (List[List[float]]): contact matrix

    Return
    ------
        @return: dictionary of compartments and deaths

            S: 0, E: 1, I: 2, R: 3, D: 4
    """
    ifr = np.array(ifr)
    T = (end_date - start_org_date).days

    country_dict = import_country(basin)
    Nk = country_dict['Nk']

    ngroups = len(Nk)  # Dynamic ngroups based on population data
    ncomp = 4

    dates = [start_org_date + timedelta(days=d) for d in range((end_date - start_org_date).days)]
    Cs = {}
    for i in dates:
        Cs[i] = imply_reductions(C, country_dict, i)

    # intial beta
    beta = get_beta(R0, mu, Nk, Cs[dates[0]])

    # initialize compartments and set initial conditions (S: 0, E: 1, I: 2, R: 3)
    compartments = np.zeros((ncomp, ngroups, T))
    deaths, infections = np.zeros((ngroups, T)), np.zeros((ngroups, T))

    # distribute intial infected and recovered among groups
    compartments[2, :, 0] = (i0 * Nk/sum(Nk) * (1 / mu) / ((1 / mu) + (1 / eps))).astype(int)
    compartments[1, :, 0] = (i0 * Nk/sum(Nk) * (1 / eps) / ((1 / mu) + (1 / eps))).astype(int)
    compartments[3, :, 0] = np.zeros(ngroups)
    compartments[0, :, 0] = (Nk - (compartments[1, :, 0] + compartments[2, :, 0] + compartments[3, :, 0])).astype(int)

    dt = 1./ daily_steps

    # simulate
    for t in np.arange(1, T, 1):
        compartments_next_day = compartments[:, :, t - 1].copy()
        new_R_day = np.zeros(ngroups, dtype=int)
        for _ in range(int(daily_steps)):
            force_inf = np.sum(beta * Cs[dates[t]] * compartments_next_day[2, :] / Nk, axis=1)

            # compute transitions
            new_E = np.random.binomial(compartments_next_day[0, :].astype(int), 1. - np.exp(-force_inf * dt))
            new_I = np.random.binomial(compartments_next_day[1, :].astype(int), 1. - np.exp(-eps * dt))
            new_R = np.random.binomial(compartments_next_day[2, :].astype(int), 1. - np.exp(-mu * dt))

            # S
            compartments_next_day[0, :] = compartments_next_day[0, :] - new_E
            # E
            compartments_next_day[1, :] = compartments_next_day[1, :] + new_E - new_I
            # I
            compartments_next_day[2, :] = compartments_next_day[2, :] + new_I - new_R
            # R
            compartments_next_day[3, :] = compartments_next_day[3, :] + new_R

            # store new_R for deaths computation
            new_R_day += new_R

        compartments[:, :, t] = compartments_next_day

        # compute deaths
        if (t - 1) + Delta < deaths.shape[1]:
            deaths[:, (t - 1) + int(Delta)] += np.random.binomial((new_R_day), ifr)
        infections[:, (t - 1)] = new_I

    cut_index = int((start_simu_date - start_org_date).days/7)

    deaths_sum = deaths.sum(axis=0)
    df_deaths = pd.DataFrame(data={"deaths": deaths_sum}, index=pd.to_datetime(dates))
    deaths_week = df_deaths.resample("W").sum()

    infections_sum = infections.sum(axis=0)
    df_infections = pd.DataFrame(data={"infections": infections_sum}, index=pd.to_datetime(dates))
    infections_week = df_infections.resample("W").sum()

    weekly_deaths = list(deaths_week.deaths.values)[cut_index:]
    weekly_infections = list(infections_week.infections.values)[cut_index:]

    df_deaths_by_group = pd.DataFrame(deaths.T, index=pd.to_datetime(dates))
    weekly_deaths_by_group = []
    for group in range(ngroups):
        df_deaths_group = df_deaths_by_group.iloc[:, group]
        deaths_week_group = df_deaths_group.resample("W").sum()
        weekly_deaths_by_group.append(list(deaths_week_group.values)[cut_index:])

    df_infections_by_group = pd.DataFrame(infections.T, index=pd.to_datetime(dates))
    weekly_infections_by_group = []
    for group in range(ngroups):
        df_infections_group = df_infections_by_group.iloc[:, group]
        infections_week_group = df_infections_group.resample("W").sum()
        weekly_infections_by_group.append(list(infections_week_group.values)[cut_index:])

    return [weekly_deaths,
            weekly_infections,
            weekly_deaths_by_group,
            weekly_infections_by_group]