#!/usr/bin/env python3

from collections import defaultdict
import concurrent.futures
import logging
import glob
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from rich.logging import RichHandler


# Set up logging with rich
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


def read_met(metfile: str) -> pd.DataFrame:
    metfile_types_conversion = {
        "Day": "int16",
        "Hr": "int16",
        "Min": "int16",
        "Pressure": "float32",
        "Temp": "float32",
        "RH": "float32",
        "WndSpd": "float32",
        "WndDir": "float32",
        "Precip": "float32",
        "Vis": "int16",
        "Aer": "int16",
        "Cloud1": "int16",
        "Cloud2": "int16",
        "Cloud3": "int16",
        "Cloud4": "int16",
        "Cloud5": "int16",
        "Cloud6": "int16",
        "Cloud7": "int16",
        "Global": "float32",
        "Direct": "float32",
        "Diffuse": "float32",
        "LWdown": "float32",
        "Zenith": "float32",
        "Azimuth": "float32",
    }
    log.debug(f"Reading met file: {metfile}")
    met = pd.read_csv(metfile, sep=" ", header=None, skiprows=5)
    met.columns = metfile_types_conversion.keys()
    met = met.astype(metfile_types_conversion)
    return met

def add_year_column(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    year = start_year
    prev_day = df.iloc[0]["Day"]
    years = []

    for day in df["Day"]:
        if (
            day < prev_day
        ):  # The day is less than the previous day, indicating a new year
            if is_leap_year(year) and prev_day == 366:
                year += 1
            elif not is_leap_year(year) and prev_day == 365:
                year += 1
            else:
                raise ValueError('Invalid "Day" value encountered in DataFrame.')
        years.append(year)
        prev_day = day

    df["Year"] = years
    # logging print unique years from df['Year']
    log.debug(f"Unique years in met file: {df['Year'].unique()}")
    return df


def set_met_index_to_datetime(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    # Add the 'Year' column
    df_with_year = add_year_column(df, start_year)

    # Convert the 'Year', 'Day', 'Hr', and 'Min' columns to datetime
    df_with_year["Datetime"] = (
        pd.to_datetime(df_with_year["Year"] * 1000 + df_with_year["Day"], format="%Y%j")
        + pd.to_timedelta(df_with_year["Hr"], unit="h")
        + pd.to_timedelta(df_with_year["Min"], unit="m")
    )

    # Set the index to the 'Datetime' column
    df_with_year.set_index("Datetime", inplace=True)

    # Drop unnecessary columns
    df_with_year.drop(["Year", "Day", "Hr", "Min"], axis=1, inplace=True)

    return df_with_year


def read_fsd(file_path: str) -> float:
    with open(file_path, "r") as f:
        data = [float(line.strip()) for line in f]
    return data

def parse_surface_mesh(meshfile: str) -> Tuple[np.ndarray, np.ndarray]:
    facets = []
    nodes = []
    with open(meshfile, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] == 'E3T':
                node_numbers = [int(tokens[i]) for i in range(2, 5)]
                material_id = int(tokens[5])
                facets.append((material_id, ) + tuple(node_numbers))
            elif tokens[0] == 'ND':
                nodes.append(tuple(float(tokens[i]) for i in range(2, 5)))
    return np.array(facets), np.array(nodes)

def parse_temperature_file(temperature_file):
    temperatures = []
    with open(temperature_file, 'r') as f:
        for i, line in enumerate(f):
            temperature = float(line.strip())
            temperatures.append(temperature)
    return np.array(temperatures)


# def serial_read_and_average_fsd(mesh_file: str, temperature_files: List[str]) -> pd.DataFrame:
#     facets, nodes = parse_surface_mesh(mesh_file)
#     temp_sums = pd.Series(0.0, index=np.unique(facets[:, 0]))
#     temp_counts = pd.Series(0, index=np.unique(facets[:, 0]))
#
#     for temperature_file in temperature_files:
#         temperatures = parse_temperature_file(temperature_file)
#
#         for facet in facets:
#             material_id = facet[0]
#             node_temps = [temperatures[node_index - 1] for node_index in facet[1:]]  # Subtract 1 from node_index
#             temp_sums[material_id] += sum(node_temps)
#             temp_counts[material_id] += len(node_temps)
#
#     return temp_sums / temp_counts
#
#
# def parallel_read_and_average_fsd(mesh_file: str, temperature_files: List[str]) -> pd.DataFrame:
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         future = executor.submit(serial_read_and_average_fsd, mesh_file, temperature_files)
#         return future.result()

def build_fsd_dataframe(
    file_averages: Dict[str, pd.Series], start_year: int, column_prefix: str = "Temperature"
) -> pd.DataFrame:
    column_prefix += "_Ave_MatID"
    df = pd.concat(file_averages, axis=1)
    df.columns = df.columns.map(lambda x: fsd_filename_to_datetime(x, start_year))
    df = df.T
    df.columns = df.columns.map(lambda x: f"{column_prefix}_{x}")
    return df

def serial_read_and_average_fsd(mesh_file: str, temperature_file: str) -> pd.Series:
    facets, nodes = parse_surface_mesh(mesh_file)
    temp_sums = pd.Series(0.0, index=np.unique(facets[:, 0]))
    temp_counts = pd.Series(0, index=np.unique(facets[:, 0]))

    temperatures = parse_temperature_file(temperature_file)

    for facet in facets:
        material_id = facet[0]
        node_temps = [temperatures[node_index - 1] for node_index in facet[1:]]  # Subtract 1 from node_index
        temp_sums[material_id] += sum(node_temps)
        temp_counts[material_id] += len(node_temps)

    return np.round(temp_sums / temp_counts, 5)


def parallel_read_and_average_fsd(mesh_file: str, temperature_file: str) -> pd.Series:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(serial_read_and_average_fsd, mesh_file, temperature_file)
        return future.result()


def compute_averages(
    mesh_file: str,
    temperature_files: List[str],
    start_year: int,
    column_prefix: str = "Temperature",
    num_processors: int = 1,
) -> pd.DataFrame:
    def process_file(temperature_file):
        if num_processors == 1:
            return serial_read_and_average_fsd(mesh_file, temperature_file)
        else:
            return parallel_read_and_average_fsd(mesh_file, temperature_file)

    file_averages = {
        temperature_file: process_file(temperature_file)
        for temperature_file in temperature_files
    }

    avg_temps_df = build_fsd_dataframe(
        file_averages, start_year=start_year, column_prefix=column_prefix
    )

    return avg_temps_df


def fsd_filename_to_datetime(filename: str, start_year: int) -> pd.Timestamp:
    # Extract the day of the year from the filename
    day_of_year_and_hour = int(filename.split("_")[-1].split(".")[0])
    day_of_year = day_of_year_and_hour // 1000
    hour = day_of_year_and_hour % 100

    # Convert the day of the year to a Timestamp
    date = pd.to_datetime(start_year * 1000 + day_of_year, format="%Y%j")
    return date + pd.to_timedelta(hour, unit="h")


if __name__ == "__main__":
    met = read_met('data/TestPlot.met')
    met_dt = set_met_index_to_datetime(met, 2022)
    print(f"{met_dt.head()=}")
    fsd_data = serial_read_and_average_fsd('data/Scenario1.2dm',
                                           glob.glob('data/file_sock300*.fsd'))

    print(f"{fsd_data.head()=}")
    temps_df = build_fsd_dataframe(fsd_data, 2022, 'Temperature')
    print(f"{temps_df.head()=}")

