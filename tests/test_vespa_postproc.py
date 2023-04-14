import datetime
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tempfile

DATA_DIR = Path('./data')

from vespa_postproc import (
    add_year_column,
    build_fsd_dataframe,
    compute_averages,
    fsd_filename_to_datetime,
    parallel_read_and_average_fsd,
    parse_surface_mesh,
    parse_temperature_file,
    read_fsd,
    read_met,
    serial_read_and_average_fsd,
    set_met_index_to_datetime,
)

def test_read_met():
    file_path = Path.cwd() / "data" / "TestPlot.met"
    met = read_met(file_path)
    assert type(met) == pd.DataFrame
    assert met.Day.dtype == "int16"
    assert len(met.columns) == 24

# Add the 'Year' column tests


def test_add_year_column_single_year():
    df = pd.DataFrame({"Day": [1, 2, 3], "Hr": [0, 1, 2], "Min": [0, 0, 0]})

    expected = pd.DataFrame(
        {
            "Day": [1, 2, 3],
            "Hr": [0, 1, 2],
            "Min": [0, 0, 0],
            "Year": [2022, 2022, 2022],
        }
    )

    result = add_year_column(df, 2022)
    pd.testing.assert_frame_equal(result, expected)


def test_add_year_column_cross_year():
    df = pd.DataFrame({"Day": [365, 1, 2], "Hr": [0, 1, 2], "Min": [0, 0, 0]})

    expected = pd.DataFrame(
        {
            "Day": [365, 1, 2],
            "Hr": [0, 1, 2],
            "Min": [0, 0, 0],
            "Year": [2022, 2023, 2023],
        }
    )

    result = add_year_column(df, 2022)
    pd.testing.assert_frame_equal(result, expected)


def test_add_year_column_leap_year():
    df = pd.DataFrame({"Day": [366, 1, 2], "Hr": [0, 1, 2], "Min": [0, 0, 0]})

    expected = pd.DataFrame(
        {
            "Day": [366, 1, 2],
            "Hr": [0, 1, 2],
            "Min": [0, 0, 0],
            "Year": [2020, 2021, 2021],
        }
    )

    result = add_year_column(df, 2020)
    pd.testing.assert_frame_equal(result, expected)


# Set index to datetime tests
def test_set_datetime_index_single_year():
    df = pd.DataFrame({"Day": [1, 2, 3], "Hr": [0, 1, 2], "Min": [0, 0, 0]})

    expected = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2022-01-01 00:00:00", "2022-01-02 01:00:00", "2022-01-03 02:00:00"]
            ),
        }
    ).set_index("Datetime")

    result = set_met_index_to_datetime(df, 2022)
    pd.testing.assert_frame_equal(result, expected)


def test_set_datetime_index_cross_year():
    df = pd.DataFrame({"Day": [365, 1, 2], "Hr": [0, 1, 2], "Min": [0, 0, 0]})

    expected = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2022-12-31 00:00:00", "2023-01-01 01:00:00", "2023-01-02 02:00:00"]
            ),
        }
    ).set_index("Datetime")

    result = set_met_index_to_datetime(df, 2022)
    pd.testing.assert_frame_equal(result, expected)


def test_set_datetime_index_leap_year():
    df = pd.DataFrame({"Day": [366, 1, 2], "Hr": [0, 1, 2], "Min": [0, 0, 0]})

    expected = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2020-12-31 00:00:00", "2021-01-01 01:00:00", "2021-01-02 02:00:00"]
            ),
        }
    ).set_index("Datetime")

    result = set_met_index_to_datetime(df, 2020)
    pd.testing.assert_frame_equal(result, expected)


def test_read_fsd():
    file_path = Path.cwd() / "data" / "file_sock300_181012.fsd"
    fsd = read_fsd(file_path)
    assert len(fsd) == 1942

@pytest.fixture
def mesh_data():
    return '''E3T 1 1 2 3 5
E3T 2 1 5 3 1
E3T 3 3 6 3 2
ND 1 0.5 0.5 0.5
ND 2 0.7 0.7 0.2
ND 3 0.3 0.6 0.4
ND 4 0.1 0.9 0.7
ND 5 0.8 0.2 0.1
ND 6 0.4 0.4 0.9
'''

def test_parse_surface_mesh(mesh_data):

    # Create a temporary file with the mesh data
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(mesh_data)
        mesh_file = f.name

    facets, nodes = parse_surface_mesh(mesh_file)

    # Expected output
    expected_facets = np.array([
        [5, 1, 2, 3],
        [1, 1, 5, 3],
        [2, 3, 6, 3]
    ])
    expected_nodes = np.array([
        [0.5, 0.5, 0.5],
        [0.7, 0.7, 0.2],
        [0.3, 0.6, 0.4],
        [0.1, 0.9, 0.7],
        [0.8, 0.2, 0.1],
        [0.4, 0.4, 0.9]
    ])

    # Assert that the output matches the expected values
    assert np.array_equal(facets, expected_facets)
    assert np.array_equal(nodes, expected_nodes)

@pytest.fixture
def temperature_files_data():
    return [
        '10.0\n20.0\n30.0\n40.0\n50.0\n60.0\n',
        '12.0\n22.0\n32.0\n42.0\n52.0\n62.0\n',
    ]

def test_parse_temperature_file():
    temperature_file = 'temperature.fsd'
    with open(temperature_file, 'w') as f:
        f.write("""10.5
15.7
20.3
25.1
30.8
35.4
""")

    # Call the function and get the result
    temperatures = parse_temperature_file(temperature_file)

    # Expected output
    expected_temperatures = np.array([ 10.5, 15.7, 20.3, 25.1, 30.8, 35.4 ])

    # Assert that the output matches the expected values
    assert np.array_equal(temperatures, expected_temperatures)

def test_serial_read_and_average_fsd():
    mesh_file = "./data/Scenario1_mats.2dm"
    temperature_file = "./data/file_sock300_181012.fsd"
    # [43.402247539002865, 45.91624048917751, 37.7665000634218]
    expected = pd.Series({1: 43.402247539002865,
                          2: 45.91624048917751,
                          5: 37.7665000634218})
    result = serial_read_and_average_fsd(mesh_file, temperature_file)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_parallel_read_and_average_fsd():
    mesh_file = "./data/Scenario1_mats.2dm"
    temperature_file = "./data/file_sock300_181012.fsd"
    expected = pd.Series({1: 43.402247539002865,
                          2: 45.91624048917751,
                          5: 37.7665000634218})
    result = parallel_read_and_average_fsd(mesh_file, temperature_file)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_compute_averages():
    mesh_file = "./data/Scenario1_mats.2dm"
    temperature_files = [
        "./data/file_sock300_181012.fsd",
        "./data/file_sock300_181013.fsd",
        "./data/file_sock300_181014.fsd",
    ]
    start_year = 2022
    file_averages = {
        "./data/file_sock300_181012.fsd": pd.Series({1: 43.40225,
                                                     2: 42.57872,
                                                     5: 43.08382}),
        "./data/file_sock300_181013.fsd": pd.Series({1: 45.91624,
                                                     2: 44.43341,
                                                     5: 44.60421}),
        "./data/file_sock300_181014.fsd": pd.Series({1: 37.7665,
                                                     2: 37.466,
                                                     5: 37.93235}),
    }
    result = compute_averages(mesh_file, temperature_files, start_year)
    print(f"---------------------")
    print(f"{result.head()=}")

    expected_index = pd.to_datetime(["2022-06-30 12:00", "2022-06-30 13:00",
                                     "2022-06-30 14:00"])
    expected_columns = ["Temperature_Ave_MatID_1", "Temperature_Ave_MatID_2", "Temperature_Ave_MatID_5"]
    expected_data = [
        [43.40225, 45.91624, 37.7665],
        [42.57872, 44.43341, 37.466],
        [43.08382, 44.60421, 37.93235]
    ]
    expected_result = pd.DataFrame(expected_data, index=expected_index, columns=expected_columns)

    print(f"---------------------")
    print(f"{expected_result.head()=}")

    pd.testing.assert_frame_equal(result, expected_result)

@pytest.fixture
def file_averages():
    return {
        "file_sock300_283010.fsd": pd.Series({1: 31.0, 2: 41.0, 5: 21.0}),
        "file_sock300_283011.fsd": pd.Series({1: 32.0, 2: 42.0, 5: 22.0}),
        "file_sock300_283012.fsd": pd.Series({1: 33.0, 2: 43.0, 5: 23.0}),
    }


def test_build_fsd_dataframe(file_averages):
    start_year = 2018
    result = build_fsd_dataframe(file_averages, start_year=start_year)

    expected_index = pd.to_datetime(["2018-10-10 10:00", "2018-10-10 11:00",
                                     "2018-10-10 12:00"])
    expected_columns = ["Temperature_Ave_MatID_1", "Temperature_Ave_MatID_2", "Temperature_Ave_MatID_5"]
    expected_data = [
        [31.0, 41.0, 21.0],
        [32.0, 42.0, 22.0],
        [33.0, 43.0, 23.0]
    ]

    expected_result = pd.DataFrame(expected_data, index=expected_index, columns=expected_columns)

    pd.testing.assert_frame_equal(result, expected_result)

# def test_compute_averages():
#     mesh_file = "dummy_mesh_file"
#     temperature_files = [
#         "file_sock300_283010.fsd",
#         "file_sock300_283011.fsd",
#         "file_sock300_283012.fsd",
#     ]
#     start_year = 2018
#     column_prefix = "Temperature"
#
#     # Mock serial_read_and_average_fsd function
#     mock_serial_read_and_average_fsd = MagicMock(
#         side_effect=[
#             pd.Series({1: 31.0, 2: 41.0, 5: 21.0}),
#             pd.Series({1: 32.0, 2: 42.0, 5: 22.0}),
#             pd.Series({1: 33.0, 2: 43.0, 5: 23.0}),
#         ]
#     )
#
#     with unittest.mock.patch("vespa_postproc.serial_read_and_average_fsd", mock_serial_read_and_average_fsd):
#         result = compute_averages(
#             mesh_file,
#             temperature_files,
#             start_year,
#             column_prefix=column_prefix,
#             num_processors=1,
#         )
#
#     expected_index = pd.to_datetime(["2018-10-10 10:00", "2018-10-10 11:00",
#                                      "2018-10-10 12:00"])
#     expected_columns = ["Temperature_Ave_MatID_1", "Temperature_Ave_MatID_2", "Temperature_Ave_MatID_5"]
#     expected_data = np.array([
#         [31.0, 41.0, 21.0],
#         [32.0, 42.0, 22.0],
#         [33.0, 43.0, 23.0]
#     ])
#
#     expected_result = pd.DataFrame(expected_data, index=expected_index, columns=expected_columns)
#
#     pd.testing.assert_frame_equal(result, expected_result)

def test_fsd_filename_to_datetime():
    filename = "file_sock300_180000.fsd"
    expected = pd.to_datetime("2018-06-29 00:00:00")
    result = fsd_filename_to_datetime(filename, 2018)
    assert result == expected