import numpy as np
import pandas as pd

def import_country(country, path_to_data=r"../data"):
    """
    This function returns all data needed for a specific country
        :param country (string): name of the country
        :param path_to_data (string): path to the countries folder
        :return dict of country data (country_name, work, school, home, other_locations, Nk, epi_data)
    """
    # import demographic
    Nk = pd.read_csv(path_to_data + "/regions/" + country + "/demographic/pop.csv")['population'].values

    # import restriction
    all_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/all_reductions.csv")
    # school_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/school.csv")
    # work_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/work.csv")
    # oth_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/other_loc.csv")

    # import epidemiological data
    deaths = pd.read_csv(path_to_data + "/regions/" + country + "/epidemic/weekly_deaths.csv")

    # create dict of data
    country_dict = {"country": country,
                    # "contacts_home": contacts_home,
                    # "contacts_work": contacts_work,
                    # "contacts_school": contacts_school,
                    # "contacts_other_locations": contacts_other_locations,
                    "all_red": all_reductions,
                    # "school_red": school_reductions,
                    # "work_red": work_reductions,
                    # "oth_red": oth_reductions,
                    "Nk": Nk,
                    "deaths": deaths}

    return country_dict

def get_beta(R0, mu, Nk, C):
    """
    Compute the transmission rate beta for a SEIR model with age groups
    Parameters
    ----------
        @param R0: basic reproductive number
        @param mu: recovery rate
        @param Nk: number of individuals in different age groups
        @param C: contact rate matrix
    Return
    ------
        @return: the transmission rate beta
    """
    # get seasonality adjustment
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]

    return R0 * mu / (np.max([e.real for e in np.linalg.eig(C_hat)[0]]))

def imply_reductions(C, country_dict, date):
    # get year-week
    if date.isocalendar()[1] < 10:
        year_week = str(date.isocalendar()[0]) + "-0" + str(date.isocalendar()[1])
    else:
        year_week = str(date.isocalendar()[0]) + "-" + str(date.isocalendar()[1])

    all_reductions = country_dict["all_red"]

    omega_all = all_reductions.loc[all_reductions.date == date.strftime('%Y-%m-%d')]["all_red"].values[0]

    return omega_all * C

def imply_reductions_each_place(country_dict, date):
    #print("date", date)
    # get baseline contacts matrices
    home = country_dict["contacts_home"]
    work = country_dict["contacts_work"]
    school = country_dict["contacts_school"]
    oth_loc = country_dict["contacts_other_locations"]

    # get year-week
    if date.isocalendar()[1] < 10:
        year_week = str(date.isocalendar()[0]) + "-0" + str(date.isocalendar()[1])
    else:
        year_week = str(date.isocalendar()[0]) + "-" + str(date.isocalendar()[1])

    #print("year_week", year_week)
    # get work / other_loc reductions
    work_reductions = country_dict["work_red"]
    comm_reductions = country_dict["oth_red"]
    school_reductions = country_dict["school_red"]
    school_reductions["datetime"] = pd.to_datetime(school_reductions["datetime"])

    #if year_week <= "2021-30":
    omega_w = work_reductions.loc[work_reductions.year_week == year_week]["work_red"].values[0]
    omega_c = comm_reductions.loc[comm_reductions.year_week == year_week]["oth_red"].values[0]
    C1_school = school_reductions.loc[school_reductions.datetime == date]["C1M_School.closing"].values[0]

        # check we are not going below zero
    if C1_school < 0:
        C1_school = 0

    omega_s = (3 - C1_school) / 3

    return home + (omega_s * school) + (omega_w * work) + (omega_c * oth_loc)