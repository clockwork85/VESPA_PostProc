#!/usr/bin/env python3

import base64
import concurrent.futures
from datetime import date, datetime, time
import logging
import glob
import io
import re
import threading
from typing import List, Dict, Tuple, Union

from bokeh.io import save
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Span, Title
from bokeh.palettes import Category10, Category20, Set2_8, Set3_10, viridis
from bokeh.plotting import figure, Figure
import hvplot.pandas
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, to_rgba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from panel.template import  MaterialTemplate, FastGridTemplate
from panel.template.theme import DarkTheme
import param
from PIL import Image
import pyvista as pv
from rich.logging import RichHandler

from flir_color_palettes import flir_cmap, rainbow1234_cmap, white_hot_cmap, \
    black_hot_cmap, artic_cmap, lava_cmap, yellow_cmap

# Set up logging with rich
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")

class FairyLayout:

    sublabel_color = '#6F88FC' # royal blue
    label_color =    '#45E3FF' # light blue
    slider_color =   '#FF7582' # white
    knob_color =     '#FF7582' # white
    border_color =   '#A163F7' # white




raw_css= '''
    .custom-date-range-slider .bk .bk-slider-title {
        font-size: 17px;
        margin-bottom: 12px;
        margin-left: 2px;
    }
    .custom-date-slider .bk .bk-slider-title {
        font-size: 17px;
        margin-bottom: 12px;
        margin-left: 70px;
    }
    .custom-static-text .bk-clearfix {
        font-size: 25px;
        margin-left: 45px;
    }
    .custom-btn .bk-btn {
        font-size: 15px;
    }
    '''

pn.extension(raw_css=[raw_css])





def image_to_base64(image_path: str) -> base64:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
        "Vis": "int16",
        "Aer": "int16",
        "Precip": "float32",
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

    col_conversion = {
        'Temp': 'Air Temperature',
        'Press': 'Pressure',
        'WndSpd': 'Wind Speed',
        'WndDir': 'Wind Direction',
        'Precip': 'Precipitation',
        'LWdown': 'Longwave Downwelling',
        'RH': 'Relative Humidity',
        'Direct': 'Direct Radiation',
        'Global': 'Global Radiation',
        'Diffuse': 'Diffuse Radiation',
    }

    log.debug(f"Reading met file: {metfile}")

    # Determine the separator (space or tab) by checking the first line
    with open(metfile, "r") as file:
        first_line = file.readline()
        if "\t" in first_line:
            separator = "\t"
        else:
            separator = " "

    met = pd.read_csv(metfile, sep=separator, header=None, skiprows=5)

    # Tab separated files have extra columns at the end
    if met.isna().all().any():
        print(f"Removing extra columns from {metfile}")
        met = met.dropna(axis=1, how="all")

    met.columns = metfile_types_conversion.keys()
    met = met.astype(metfile_types_conversion)
    met.rename(columns=col_conversion, inplace=True)
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
    with open(meshfile, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] == "E3T":
                node_numbers = [int(tokens[i]) for i in range(2, 5)]
                material_id = int(tokens[5])
                facets.append(tuple(node_numbers) + (material_id,))
            elif tokens[0] == "ND":
                nodes.append(tuple(float(tokens[i]) for i in range(2, 5)))
    return np.array(nodes), np.array(facets)


def parse_fsd_file(fsd_file):
    fsd_data = []
    with open(fsd_file, "r") as f:
        for i, line in enumerate(f):
            fsd_datum = float(line.strip())
            fsd_data.append(fsd_datum)
    return np.array(fsd_data)


def serial_read_and_average_fsd(
    mesh_file: str, fsd_file: str, is_nodal=True
) -> pd.Series:
    log.debug(f'Processing fsd file: {fsd_file}')
    nodes, facets = parse_surface_mesh(mesh_file)
    # set each material ID to 0.0
    temp_sums = pd.Series(0.0, index=np.unique(facets[:, -1]))
    temp_counts = pd.Series(0, index=np.unique(facets[:, -1]))

    fsd_data = parse_fsd_file(fsd_file)

    for facet_idx, facet in enumerate(facets):
        material_id = facet[-1]
        if is_nodal:
            # Subtract 1 from node_index
            data = [fsd_data[node_index - 1] for node_index in facet[:3]]
        else:
            data = [fsd_data[facet_idx]]

        temp_sums[material_id] += sum(data)
        temp_counts[material_id] += len(data)

    return np.round(temp_sums / temp_counts, 5)


def parallel_read_and_average_fsd(
    mesh_file: str, data_files: List[str], is_nodal=True, num_processors: int = 1
) -> pd.Series:

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:

        futures = {executor.submit(serial_read_and_average_fsd, mesh_file, fsd_file, is_nodal): fsd_file for fsd_file in data_files}
        results = []

        processed_files = 0
        total_files = len(data_files)

        for future in concurrent.futures.as_completed(futures):
            fsd_file = futures[future]
            try:
                result = future.result()
                processed_files += 1
                print(f'File {fsd_file} processed ({processed_files}/{total_files})')
                results.append(result)
            except Exception as e:
                print(f'File {fsd_file} encountered an error: {e}')
                results.append(None)

        return results

def compute_fsd_averages_by_material_id(
    mesh_file: str,
    data_files: List[str],
    start_year: int,
    column_prefix: str = "Temperature",
    is_nodal: bool = True,
    num_processors: int = 1,
) -> pd.DataFrame:

    if num_processors == 1:
        print('Running serially')
        file_averages = {
            data_file: serial_read_and_average_fsd(mesh_file, data_file, is_nodal)
            for data_file in data_files
        }
    else:
        print(f'Running in parallel with {num_processors} processors')
        parallel_results = parallel_read_and_average_fsd(mesh_file, data_files, is_nodal, num_processors)
        file_averages = dict(zip(data_files, parallel_results))

    avg_df = build_fsd_dataframe(
        file_averages, start_year=start_year, column_prefix=column_prefix
    )

    return avg_df

def build_image_dataframe(images: List[str], start_year: int, column_prefix: str = "Imagery", store_data: bool = False) \
        -> pd.DataFrame:

    def image_data(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            img = Image.open(f)
            return np.array(img)

    datetime_file_dict = {
        image_filename_to_datetime(filename, start_year): (
            image_data(filename) if store_data else filename
        ) for filename in images
    }
    image_series = pd.Series(datetime_file_dict)
    image_series.sort_index(inplace=True)

    image_series.name = column_prefix

    return image_series.to_frame()

def build_vtk_temperature_dataframe(surface_temp_files: List[str], start_year: int,
                                    column_prefix: str = "Surface Temperature", store_data: bool=False) -> \
        pd.DataFrame:
    datetime_file_dict = { fsd_filename_to_datetime(filename, start_year): (
            read_fsd(filename) if store_data else filename
        ) for filename in surface_temp_files
    }
    temp_series = pd.Series(datetime_file_dict)
    temp_series.sort_index(inplace=True)
    temp_series.name = column_prefix

    return temp_series.to_frame()

def build_vtk_flux_dataframe(surface_flux_files: List[str], start_year: int,
                                    column_prefix: str = "Surface Flux") -> \
        pd.DataFrame:
    datetime_file_dict = { fsd_filename_to_datetime(filename, start_year): filename
                           for filename in surface_flux_files }
    temp_series = pd.Series(datetime_file_dict)
    temp_series.sort_index(inplace=True)

    temp_series.name = column_prefix

    return temp_series.to_frame()

def build_fsd_dataframe(
    file_averages: Dict[str, pd.Series],
    start_year: int,
    column_prefix: str = "Temperature",
) -> pd.DataFrame:
    column_prefix += "_Ave_MatID"
    df = pd.concat(file_averages, axis=1)
    df.columns = df.columns.map(lambda x: fsd_filename_to_datetime(x, start_year))
    df = df.T
    df.columns = df.columns.map(lambda x: f"{column_prefix}_{x}")
    return df


def fsd_filename_to_datetime(filename: str, start_year: int) -> pd.Timestamp:
    # Extract the day of the year from the filename
    day_of_year_and_hour = int(filename.split("_")[-1].split(".")[0])
    day_of_year = day_of_year_and_hour // 1000
    hour = day_of_year_and_hour % 100

    # Convert the day of the year to a Timestamp
    date = pd.to_datetime(start_year * 1000 + day_of_year, format="%Y%j")
    return date + pd.to_timedelta(hour, unit="h")

def image_filename_to_datetime(filename: str, start_year: int) -> pd.Timestamp:
    date_pattern = r'\d{6}'
    day_of_year_and_hour = int(re.search(date_pattern, filename).group(0))
    day_of_year = day_of_year_and_hour // 1000
    hour = day_of_year_and_hour % 100

    date = pd.to_datetime(start_year * 1000 + day_of_year, format="%Y%j")
    return date + pd.to_timedelta(hour, unit="h")


def merge_dataframes_on_datetime(
    *dataframes_and_series: Union[pd.DataFrame, pd.Series], how: str = "inner"
) -> pd.DataFrame:
    if len(dataframes_and_series) < 2:
        raise ValueError("At least two DataFrames are required for merging.")

    dataframes = [
        obj.to_frame() if isinstance(obj, pd.Series) else obj
        for obj in dataframes_and_series
    ]

    merged_data = dataframes[0].join(dataframes[1:], how=how)
    return merged_data

# Images

def image_histogram(image_path_or_data: Union[str, np.ndarray], bins: int = 5):

    if isinstance(image_path_or_data, str):
        image = Image.open(image_path_or_data)
    else:
        image = Image.fromarray(image_path_or_data)

    if image.mode != 'L':
        image = image.convert('L')
    image_data = np.array(image)
    df = pd.DataFrame({
        'Grayscale': image_data.flatten(),
    })

    histogram = df.hvplot.hist(y='Grayscale', ylabel='Total Irradiance (W/m^2)', yformatter='%.1e', bins=bins, color='gray',
                               width=500, height=500)
    return histogram

# Mesh Stuff
def read_2dm_pyvista(mesh_file: str) -> pv.PolyData:
    points = []
    facets = []
    mats = []

    with open(mesh_file, 'r') as f:
        for line in f:
            if line.startswith('ND'):
                _, nnum, x, y, z = line.split()
                points.append([float(x), float(y), float(z)])
            elif line.startswith('E3T'):
                _, fnum, n1, n2, n3, mat = line.split()
                facets.extend([3, int(n1) - 1, int(n2) - 1, int(n3) - 1])
                mats.append(int(mat))
    mesh = pv.PolyData()
    mesh.points = np.array(points)
    mesh.faces = np.array(facets)
    log.debug(f"{len(mesh.points)=} points")
    log.debug(f"{mesh.n_cells=}")
    mesh.cell_data['MatID'] = np.array(mats, dtype=np.int32)

    return mesh, len(mesh.points), mesh.n_cells

def plot_mesh(mesh: pv.PolyData, cmap) -> pv.Plotter:
    plotter = pv.Plotter(window_size=(1000, 1000))
    plotter.add_mesh(mesh, scalars='MatID', cmap=cmap, show_edges=False, show_scalar_bar=True)
    plotter.view_xy()
    return plotter

def numpy_datetime64_to_datetime(dt64: np.datetime64) -> datetime:
    if isinstance(dt64, datetime):
        return dt64
    if isinstance(dt64, date):
        return datetime.combine(dt64, datetime.min.time())
    ts = (dt64.astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00', 's')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def create_met_plot(source: ColumnDataSource, met_column: str,
                    legend_label: str, y_axis_label: str, span_line: Span, width: int, height: int) -> figure:
    title_text = f"{met_column.capitalize()} vs. Date"
    met_plot = figure(x_axis_type='datetime',
                      x_axis_label='Date', y_axis_label=y_axis_label, plot_width=width,
                      plot_height=height
                      )
    met_plot.line(x='index', y=met_column, source=source, legend_label=legend_label)
    met_plot.title = Title(text=title_text, align='center', text_font_size='18px')
    if span_line:
        met_plot.add_layout(span_line)
    met_plot.legend.location = 'top_left'
    met_plot.legend.click_policy = 'hide'

    return met_plot

def create_variable_plot(source: ColumnDataSource, columns: List[str], title: str,
                         y_axis_label: str, span_line: Span, colors, width: int, height: int) -> figure:
    title_text = f"{title}"
    var_plot = figure(x_axis_type='datetime', x_axis_label='Date',
                      y_axis_label=y_axis_label, plot_width=width, plot_height=height)
    for column in columns:
        matid = int(column.split('_')[-1])
        legend_label = f'Material ID {matid}'
        var_plot.line(x='index', y=column, source=source, line_width=1,
                      color=colors[matid-1], legend_label=legend_label)

    var_plot.title = Title(text=title_text, align='center', text_font_size='18px')
    if span_line:
        var_plot.add_layout(span_line)
    var_plot.legend.location = 'top_left'
    var_plot.legend.click_policy = 'hide'
    return var_plot

class DashboardModel(param.Parameterized):
    source: ColumnDataSource = param.Parameter()
    spatial_df: pd.DataFrame = param.Parameter()

    def __init__(self, source: ColumnDataSource, spatial_df: pd.DataFrame,
                 mesh_file: str):
        super().__init__()
        self.source = source
        self.spatial_df = spatial_df
        self.filtered_source = ColumnDataSource(data=self.source.data)
        self.mesh, self.surface_mesh_nodes, self.surface_mesh_facets = read_2dm_pyvista(mesh_file)
        self.current_datetime = None
        self.mat_palette_colors = Set2_8
        self.vtk_mat_palette_colors = ListedColormap(Set2_8[:self.highest_matid])
        self.vtk_vtk_palette_colors = ListedColormap('plasma')
        self.custom_color_maps = {
            'flir': flir_cmap,
            'white hot': white_hot_cmap,
            'black hot': black_hot_cmap,
            'rainbow1234': rainbow1234_cmap,
            'artic': artic_cmap,
            'yellow': yellow_cmap,
            'lava': lava_cmap,
        }
        self.default_color_maps = {
            'plasma': cm.plasma,
        }
        self.met_unit_dict = {
            'Air Temperature': 'Temperature (°C)',
            'Pressure': 'Millibar (mbar)',
            'Global Radiation': 'Flux (W/m^2)',
            'Direct Radiation': 'Flux (W/m^2)',
            'Diffuse Radiation': 'Flux (W/m^2)',
            'Precipitation': 'Millimeters (mm)',
            'Wind Speed': 'Meters per Second (m/s)',
            'Wind Direction': 'Direction (degrees)',
            'Zenith': 'Degrees',
            'Azimuth': 'Degrees',
            'Relative Humidity': 'Percent (%)',
            'Longwave Downwelling': 'Flux (W/m^2)',
        }

    # Properties
    @property
    def highest_matid(self) -> int:
        matid_cols = [col for col in self.source.data.keys() if
                      'Temperature_Ave_MatID_' in col]
        return max([int(col.split('_')[-1]) for col in matid_cols])


    def update_image_panel(self, image_data_or_path: Union[str, np.ndarray], palette='white hot') -> pn.pane.JPG:

        if isinstance(image_data_or_path, str):
            with open(image_data_or_path, 'rb') as f:
                image = Image.open(f)
        else:
            image = Image.fromarray(image_data_or_path)

        img_array = np.array(image)

        if palette in self.custom_color_maps:
            cmap = self.custom_color_maps[palette]
        elif palette in self.default_color_maps:
            cmap = cm.get_cmap(self.default_color_maps[palette])
        else:
            cmap = cm.get_cmap(palette)

        colored_image = cmap(img_array)

        buf = io.BytesIO()
        plt.imsave(buf, colored_image, format='jpg')
        buf.seek(0)

        return pn.pane.JPG(buf, width=500, height=500)

    def update_filtered_source(self, date_range: Tuple[datetime, datetime]):
        date_range_new = date_range.new
        start, end = np.datetime64(date_range_new[0]), np.datetime64(date_range_new[1])
        mask = (self.source.data['index'] >= start) & (self.source.data['index'] <= end)
        self.filtered_source.data = {k: v[mask] for k, v in self.source.data.items()}

    def update_mat_palette(self, mat_palette: param.Event):
        mat_palette_new = mat_palette.new
        log.debug(f'From inside Mat Palette: {mat_palette_new=}')
        if mat_palette_new == 'Viridis':
            # MatIDs are never negative or zero so subtract one for indexing
            self.mat_palette_colors = viridis(self.highest_matid)
        elif mat_palette_new == 'Set2_8':
            self.mat_palette_colors = Set2_8
        elif mat_palette_new == 'Set3_10':
            self.mat_palette_colors = Set3_10
        elif mat_palette_new == 'Category10':
            self.mat_palette_colors = Category10[self.highest_matid]
        elif mat_palette_new == 'Category20':
            self.mat_palette_colors = Category20[self.highest_matid]
        else:
            raise ValueError('Invalid Mat Palette')

    def update_vtk_mat_palette(self, mat_palette: str):
        mat_palette_new = mat_palette.new
        print(f"From inside VTK mat palette: {mat_palette_new=}")
        if mat_palette_new == 'Viridis':
            self.vtk_mat_palette_colors = ListedColormap(viridis(self.highest_matid))
        elif mat_palette_new == 'Set2_8':
            truncated_cmap = Set2_8[:self.highest_matid]
            self.vtk_mat_palette_colors = ListedColormap(truncated_cmap)
        elif mat_palette_new == 'Set3_10':
            truncated_cmap = Set3_10[:self.highest_matid]
            self.vtk_mat_palette_colors = ListedColormap(truncated_cmap)
        elif mat_palette_new == 'Category10':
            truncated_cmap = Category10[self.highest_matid]
            self.vtk_mat_palette_colors = ListedColormap(truncated_cmap)
        elif mat_palette_new == 'Category20':
            truncated_cmap = Category20[self.highest_matid]
            self.vtk_mat_palette_colors = ListedColormap(truncated_cmap)
        else:
            raise ValueError('Invalid Mat Palette')

    def update_vtk_temp_palette(self, cmap_str: str, event=None):
        log.debug(f"From inside VTK temp palette: {cmap_str}")
        if cmap_str in self.custom_color_maps:
            cmap = self.custom_color_maps[cmap_str]
        elif cmap_str in self.default_color_maps:
            cmap = cmap_str
        self.vtk_temp_palette_colors = cmap
        log.debug(f"After update vtk temp: {self.vtk_temp_palette_colors=}")

    def update_current_datetime(self, date_event: np.datetime64):
        date = date_event.new
        date_dt = numpy_datetime64_to_datetime(date)
        self.current_datetime = date_dt

    def update_temperatures(self):
        log.debug(f"Updating temperatures to date {self.current_datetime}")
        temp_data_or_path = self.spatial_df.loc[self.current_datetime]['Surface Temperature']

        if isinstance(temp_data_or_path, str):
            temps = read_fsd(temp_data_or_path)
        else:
            temps = temp_data_or_path

        self.mesh.point_data['Temperature'] = np.array(temps, dtype=float)


class DashboardView(param.Parameterized):
    model: DashboardModel = param.Parameter()

    def __init__(self, model: DashboardModel):
        super().__init__()
        self.model = model
        self.date_range_slider = self.init_date_range_slider()
        self.date_slider = self.init_date_slider()
        self.met_variable_select = self.init_met_variable_select()
        self.mat_palette_select = self.init_mat_palette_select()
        self.image_color_palette_select = self.init_image_color_palette_select()
        self.image_hist_bin_number_widget = self.init_image_hist_bin_number_widget()
        self.vtk_array_select = self.init_vtk_array_select()
        self.model.current_datetime = self.date_slider.value
        self.current_datetime_text = self.init_current_datetime_text()
        self.spatial_current_button = self.init_spatial_current_button()
        self.span_line = Span(location=self.date_slider.value, dimension='height',
                              line_color='red', line_dash='dashed', line_width=1)
        self.mat_palette_colors = viridis(self.model.highest_matid)
        self.plotter = plot_mesh(self.model.mesh, self.model.vtk_mat_palette_colors)
        self.spacer = self.init_makeshift_spacer(50)
        self.match_color_palette = self.init_checkbox_match_color_palette()
        self.mesh_show_edges = self.init_checkbox_mesh_show_edges()

    @property
    def current_datetime_string(self) -> str:
        print(f"From inside current_datetime_string: {self.model.current_datetime=}")
        date_dt = numpy_datetime64_to_datetime(self.model.current_datetime)
        return f'{date_dt:%Y-%m-%d %H:%M}'

    # Methods
    def update_span_line(self, event: param.Event):
        date_value = np.datetime64(event.new)
        self.span_line.location = date_value

    def create_image_pane(self):
        curr_image = self.model.spatial_df.loc[self.model.current_datetime]['Imagery']
        new_image_pane = self.model.update_image_panel(curr_image,
                                          palette=self.image_color_palette_select.value)
        datetime_text = self.init_current_datetime_text()
        datetime_text.name = ''
        datetime_text.style={'font-size': '20px'}
        return pn.Column(pn.Row(self.image_color_palette_select, pn.Column(self.init_makeshift_spacer(10),
                                                                           datetime_text)), new_image_pane,
                         sizing_mode='stretch_both', height=100, width=100)

    def create_image_hist_pane(self):
        curr_image = self.model.spatial_df.loc[self.model.current_datetime]['Imagery']
        new_image_hist_pane = image_histogram(curr_image,
                                          bins=self.image_hist_bin_number_widget.value)
        return pn.Column(self.image_hist_bin_number_widget, self.init_makeshift_spacer(1),
                         pn.Row(self.init_makeshift_width_spacer(20), new_image_hist_pane),
                         sizing_mode='stretch_both', height=100, width=100)

    def create_vtk_pane(self):
        return pn.Column(pn.Row(self.vtk_array_select, pn.Column(self.match_color_palette, self.mesh_show_edges)), pn.panel(
            self.plotter.ren_win, sizing_mode='stretch_both', enable_keybindings=True, orientation_widget=True),
                         sizing_mode='stretch_both', height=200, width=400)

    def met_plot(self, met_variable: str, date: np.datetime64) -> pn.layout:
        met_plot_future = concurrent.futures.Future()
        def met_thread():
            met_plot = create_met_plot(source=self.model.filtered_source, met_column=met_variable,
                                y_axis_label=self.model.met_unit_dict[met_variable], span_line=self.span_line,
                                legend_label=met_variable, width=550, height=400)
            met_plot_future.set_result(met_plot)
        threading.Thread(target=met_thread).start()
        return met_plot_future.result()

    def temperature_plot(self, mat_palette_select, date: np.datetime64) -> pn.layout:
        temp_plot_future = concurrent.futures.Future()

        def temp_thread():
            title = 'Temperature vs. Date'
            y_axis_label = 'Temperature (°C)'
            temp_cols = [col for col in self.model.filtered_source.data.keys() if
                                'Temperature_Ave_MatID_' in col]
            temp_plot = create_variable_plot(self.model.filtered_source, temp_cols, title,
                                    y_axis_label, self.span_line, self.model.mat_palette_colors, width=550, height=400)
            temp_plot_future.set_result(temp_plot)
        threading.Thread(target=temp_thread).start()
        return temp_plot_future.result()

    def flux_plot(self, mat_palette_select, date: np.datetime64) -> pn.layout:
        flux_plot_future = concurrent.futures.Future()

        def flux_thread():
            title = 'Flux vs. Date'
            y_axis_label = 'Flux (W/m^2)'
            flux_cols = [col for col in self.model.filtered_source.data.keys() if
                            'Flux_Ave_MatID_' in col]
            flux_plot = create_variable_plot(self.model.filtered_source, flux_cols, title,
                                        y_axis_label, self.span_line,
                                        self.model.mat_palette_colors, width=550,
                                        height=400)
            flux_plot_future.set_result(flux_plot)
        threading.Thread(target=flux_thread).start()
        return flux_plot_future.result()

    def update_vtk_pane_with_new_mesh(self):
        vtk_array = self.vtk_array_select.value
        log.debug(f"Vtk array from update_vtk_pane_with_new_mesh: {vtk_array}")
        log.debug(f"Image cmap {self.image_color_palette_select.value}")
        show_edges = self.mesh_show_edges.value
        if vtk_array == 'Temperature':
            title = 'Temperature (°C)'
            print(f"{self.match_color_palette.value=}")
            if self.match_color_palette.value:
                self.model.update_vtk_temp_palette( self.image_color_palette_select.value)
                cmap = self.model.vtk_temp_palette_colors
            else:
                cmap = 'plasma'
        elif vtk_array == 'MatID':
            title = 'Material ID'
            cmap = self.model.vtk_mat_palette_colors
        else:
            raise ValueError('Invalid VTK Array')
        log.debug(f"Updating VTK Pane with {vtk_array} and colormap {cmap}")
        self.plotter.remove_actor(self.plotter.renderer.GetActors().GetLastActor())
        self.plotter.add_mesh(self.model.mesh,
                              scalars=vtk_array, show_edges=show_edges, cmap=cmap,
                              show_scalar_bar=True, scalar_bar_args=dict(title=title),
                              name=vtk_array)
    def view(self):

        # Top Row Widgets
        self.met_plot_widget = pn.Column(self.met_variable_select, self.met_plot_bind,
                                            sizing_mode='fixed', width=550, height=465)
        self.temp_plot_widget = pn.Column(self.mat_palette_select, self.temp_plot_bind,
                                            sizing_mode='fixed', width=550, height=465)
        self.flux_plot_widget = pn.Column(self.spacer, self.flux_plot_bind, sizing_mode='fixed', width=550,
                                          height=465)

        # Bottom row Widgets
        self.image_pane = self.create_image_pane()
        self.image_hist_pane = self.create_image_hist_pane()
        self.vtk_pane = self.create_vtk_pane()

        self.bottom_row = pn.Row(self.image_pane, self.image_hist_pane, self.vtk_pane,
                                    sizing_mode='stretch_width')
        self.top_row = pn.Row(self.met_plot_widget, self.init_makeshift_width_spacer(10), self.temp_plot_widget,
                                self.init_makeshift_width_spacer(10), self.flux_plot_widget,
                                sizing_mode='stretch_both')

        logo_base64 = image_to_base64('./data/ERDC_Graphic_Breakdown_ERDC_Gear-symbols.png')
        logo_uri = f"data:image/jpeg;base64,{logo_base64}"
        footer_img = pn.pane.HTML(f'<img src="{logo_uri}" width="150" height="150" style="center-align: middle;">')
        header_base64 = image_to_base64('./data/ERDC_Graphic_Breakdown_Full_ERDC_Graphic-White_Text.png')
        header_uri = f"data:image/jpeg;base64,{header_base64}"
        header_img = pn.pane.HTML(f'<img src="{header_uri}" width="200", height="100", style="vertical-align: middle;">')

        header_row = pn.Row(self.init_makeshift_width_spacer(650), header_img)

        sidebar_params = [
            pn.pane.Markdown('# Date Range', style={'color': FairyLayout.label_color}),
            pn.pane.Markdown('### Choose the date range to graph the data', style={'color': FairyLayout.sublabel_color}),
            self.date_range_slider, pn.Spacer(height=35), pn.layout.Divider(),
            pn.pane.Markdown('# Date', style={'color': FairyLayout.label_color}),
            pn.pane.Markdown('### Choose the particular date to view', style={'color': FairyLayout.sublabel_color}),
            self.date_slider, pn.Spacer(height=35), pn.layout.Divider(),
            pn.Spacer(height=90),
            pn.pane.Markdown('# Current Spatial Data Date', style={'color': FairyLayout.label_color}),
            self.current_datetime_text,
            pn.Spacer(height=15), self.spatial_current_button,
            pn.Spacer(height=340), footer_img
        ]

        self.template = MaterialTemplate(title='VESPA Simulation Analysis',
                                         theme=DarkTheme,
                                         )
        self.template.header.append(header_row)

        # Add components as separate roots to the main area
        self.template.main.append(pn.pane.Markdown('# Plot Components', style={'color': FairyLayout.label_color}))
        self.template.main.append(self.top_row)
        self.template.main.append(pn.Spacer(background='#ffffff', height=2))
        self.template.main.append(pn.Row(pn.pane.Markdown('# Spatial Components', style={'color': FairyLayout.label_color}),
                                         self.init_makeshift_width_spacer(900),
                                         pn.pane.Markdown(f'## Mesh Nodes: {self.model.surface_mesh_nodes}', style={'color': FairyLayout.label_color}),
                                         self.init_makeshift_width_spacer(150),
                                         pn.pane.Markdown(f'## Mesh Facets: {self.model.surface_mesh_facets}', style={'color': FairyLayout.label_color})))
        self.template.main.append(self.bottom_row)

        for component in sidebar_params:
            self.template.sidebar.append(component)

        return self.template

    def init_date_range_slider(self) -> pn.widgets.DateRangeSlider:
        return pn.widgets.DateRangeSlider(name='',
                                          start=self.model.source.data['index'][0],
                                          end=self.model.source.data['index'][-1],
                                          value=(self.model.source.data['index'][0],
                                                 self.model.source.data['index'][-1]),
                                          step=60 * 60 * 1000,
                                          tooltips=True,
                                          format='%Y-%m-%d %H:%M',
                                          css_classes=['custom-date-range-slider']
                                          )

    def init_date_slider(self) -> pn.widgets.DateSlider:
        return pn.widgets.DateSlider(name='',
                                     as_datetime=True,
                                     start=self.model.source.data['index'][0],
                                     end=self.model.source.data['index'][-1],
                                     value=self.model.source.data['index'][0],
                                     step=60 * 60 * 1000,
                                     tooltips=True,
                                     format='%Y-%m-%d %H:%M',
                                     css_classes=['custom-date-slider']
                                     )

    def init_met_variable_select(self) -> pn.widgets.Select:
        return pn.widgets.Select(name='Met Variable',
                                 options=['Air Temperature', 'Relative Humidity', 'Global Radiation', 'Wind Direction',
                                          'Wind Speed', 'Direct Radiation', 'Diffuse Radiation', 'Longwave Downwelling',
                                          'Zenith', 'Azimuth', 'Precipitation', 'Pressure'],
                                 value='Air Temperature', sizing_mode='fixed')

    def init_mat_palette_select(self) -> pn.widgets.Select:
        options = ['Set2_8', 'Viridis', 'Set3_10', 'Category10', 'Category20']
        disabled_options=[]

        if self.model.highest_matid < 3:
            disabled_options = {'Category10', 'Category20'}

        return pn.widgets.Select(name='Material Palette',
                                 options=options,
                                 disabled_options=disabled_options,
                                 value='Set2_8', sizing_mode='fixed')

    def init_image_color_palette_select(self) -> pn.widgets.Select:
        return pn.widgets.Select(name='Image Color Palette',
                                 options=['flir',
                                          'rainbow1234',
                                          'yellow',
                                          'white hot',
                                          'black hot',
                                          'artic',
                                          'plasma',
                                          'lava',
                                          ],
                                 value='white hot',
                                 sizing_mode='fixed')

    def init_image_hist_bin_number_widget(self) -> pn.widgets.IntSlider:
        return pn.widgets.IntSlider(name='Image Histogram Bin Number',
                                    start=2, end=20, step=1, value=5)

    def init_vtk_array_select(self) -> pn.widgets.Select:
        return pn.widgets.Select(name='VTK Cell Array',
                                 options=['MatID',
                                          'Temperature'],
                                 value='MatID',
                                 sizing_mode='fixed')

    def init_current_datetime_text(self) -> pn.widgets.StaticText:
        return pn.widgets.StaticText(name='',
                                     value=self.current_datetime_string,
                                     css_classes=['custom-static-text'])

    def init_spatial_current_button(self) -> pn.widgets.Button:
        return pn.widgets.Button(name='Render Current Date Spatial Components',
                                 button_type='primary',
                                 css_classes=['custom-btn'])

    def init_makeshift_spacer(self, pixels: int) -> pn.pane.HTML:
        return pn.pane.HTML(f'<div style="height: {pixels}px;"></div>')

    def init_makeshift_width_spacer(self, pixels: int) -> pn.pane.HTML:
        return pn.pane.HTML(f'<div style="width: {pixels}px;"></div>')

    def init_checkbox_match_color_palette(self) -> pn.widgets.Checkbox:
        return pn.widgets.Checkbox(name="Match Image Temperature Color Palette",
                                   value=False)

    def init_checkbox_mesh_show_edges(self) -> pn.widgets.Checkbox:
        return pn.widgets.Checkbox(name="Show Edges",
                                   value=False)

class DashboardController(param.Parameterized):
    model: DashboardModel = param.Parameter()
    view: DashboardView = param.Parameter()

    def __init__(self, model: DashboardModel, view: DashboardView):
        super().__init__()
        self.model = model
        self.view = view

        # Watch widgets for changes in model
        self.view.date_range_slider.param.watch(self.model.update_filtered_source,
                                              'value')
        self.view.mat_palette_select.param.watch(self.model.update_mat_palette, 'value')
        self.view.mat_palette_select.param.watch(self.model.update_vtk_mat_palette, 'value')
        self.view.date_slider.param.watch(self.model.update_current_datetime, 'value')

        # Watch widgets for changes in view
        self.view.date_slider.param.watch(self.view.update_span_line, 'value')
        self.view.image_color_palette_select.param.watch(self.update_image_pane,
                                                         'value')
        self.view.image_color_palette_select.param.watch(self.update_vtk_pane, 'value')
        self.view.image_hist_bin_number_widget.param.watch( self.update_image_hist_pane,
                                                      'value_throttled')
        self.view.vtk_array_select.param.watch(self.update_vtk_pane, 'value')
        self.view.mat_palette_select.param.watch(self.update_vtk_pane, 'value')
        self.view.date_range_slider.param.watch(self.update_date_slider,
                                                'value_throttled')
        self.view.match_color_palette.param.watch(self.update_vtk_pane, 'value')
        self.view.mesh_show_edges.param.watch(self.update_vtk_pane, 'value')

        # Button
        self.view.spatial_current_button.on_click(
            self.on_spatial_current_datetime_change)

        self.view.met_plot_bind = pn.bind(self.view.met_plot,
                                          self.view.met_variable_select,
                                     self.view.date_slider.param.value)
        self.view.temp_plot_bind = pn.bind(self.view.temperature_plot,
                                     self.view.mat_palette_select,
                                     self.view.date_slider.param.value)
        self.view.flux_plot_bind = pn.bind(self.view.flux_plot,
                                     self.view.mat_palette_select,
                                     self.view.date_slider.param.value)

    # Button callback
    def on_spatial_current_datetime_change(self, event):
        self.view.current_datetime_text.value = self.view.current_datetime_string
        self.view.image_pane = self.view.create_image_pane()
        self.view.image_hist_pane = self.view.create_image_hist_pane()
        self.view.vtk_pane = self.update_vtk_pane()
        self.view.bottom_row[0] = self.view.image_pane
        self.view.bottom_row[1] = self.view.image_hist_pane

    def update_date_slider(self, event):
        date_range = event.new
        start, end = np.datetime64(date_range[0]), np.datetime64(date_range[1])
        self.view.date_slider.start = start
        self.view.date_slider.end = end
        if self.view.date_slider.value < start:
            self.view.date_slider.value = start
        if self.view.date_slider.value > end:
            self.view.date_slider.value = end

    def update_image_pane(self, event):
        self.view.image_pane = self.view.create_image_pane()
        self.view.bottom_row[0] = self.view.image_pane

    def update_image_hist_pane(self, event):
        self.view.image_hist_pane = self.view.create_image_hist_pane()
        self.view.bottom_row[1] = self.view.image_hist_pane

    def update_vtk_pane(self, event=None):
        vtk_array = self.view.vtk_array_select.value
        print(f'Updating VTK Pane with {vtk_array}')
        if vtk_array == 'Temperature':
            self.model.update_temperatures()
            self.view.update_vtk_pane_with_new_mesh()
        elif vtk_array == 'MatID':
            self.view.update_vtk_pane_with_new_mesh()
        else:
            raise ValueError('Invalid VTK Array')
        self.view.vtk_pane = self.view.create_vtk_pane()
        self.view.bottom_row[2] = self.view.vtk_pane

if __name__ == "__main__":

    scenario = 'VizPlot'

    if scenario == 'VizPlot':
        data_dir = '/Users/rdgslmdb/data_repo/VizTestPlot/VizPlot'
        met_file = f'{data_dir}/VizPlot.met'
        surf_mesh_file = f'{data_dir}/VizPlot.2dm'
        surf_temp_files = glob.glob(f'{data_dir}/file_sock300*.fsd')
        surf_flux_files = glob.glob(f'{data_dir}/file_sock200*.fsd')
        image_files = glob.glob(f'{data_dir}/VizPlot*.jpg')
        plot_comp_hdf = 'VizPlot_plot_comp.h5'
        spatial_comp_hdf = 'VizPlot_spatial_comp.h5'
    elif scenario == 'Scenario1':
        data_dir = '/Users/rdgslmdb/data_repo/VizTestPlot/Scenario1/new'
        met_file = f'{data_dir}/Scenario1.met'
        surf_mesh_file = f'{data_dir}/Scenario1_mats.2dm'
        surf_temp_files = glob.glob(f'{data_dir}/file_sock300*.fsd')
        surf_flux_files = glob.glob(f'{data_dir}/file_sock200*.fsd')
        image_files = glob.glob(f'{data_dir}/Scenario1*.jpg')
        plot_comp_hdf = 'Scenario1_plot_comp.h5'
        spatial_comp_hdf = 'Scenario1_spatial_comp.h5'

    first_run = False
    store_data = True

    if first_run:
        met = read_met(met_file)

        log.debug("Reading Temperatures")
        met_dt = set_met_index_to_datetime(met, 2022)
        surf_temp_plot_df = compute_fsd_averages_by_material_id(
            surf_mesh_file,
            surf_temp_files,
            2022,
            'Temperature',
            num_processors=6
        )

        if scenario == 'Scenario1':
            surf_flux_plot_df = compute_fsd_averages_by_material_id(
                surf_mesh_file,
                surf_flux_files,
                2022,
                'Flux',
                num_processors=6,
                is_nodal=True
            )

        image_df = build_image_dataframe(
            image_files,
            2022,
            'Imagery',
            store_data=store_data
        )

        surf_mesh_temp_df = build_vtk_temperature_dataframe(
            surf_temp_files,
            2022,
            'Surface Temperature',
            store_data=store_data
        )

        surf_mesh_flux_df = build_vtk_flux_dataframe(
            surf_flux_files,
            2022,
            'Surface Flux'
        )

        spatial_df = merge_dataframes_on_datetime(image_df, surf_mesh_temp_df, surf_mesh_flux_df)

        if scenario == 'VizPlot':
            dfs = merge_dataframes_on_datetime(met_dt, surf_temp_plot_df)  # , surf_flux_plot_df)
            dfs['Flux_Ave_MatID_1'] = dfs['Temperature_Ave_MatID_1'] * 35
            dfs['Flux_Ave_MatID_2'] = dfs['Temperature_Ave_MatID_2'] * 35
            dfs['Flux_Ave_MatID_3'] = dfs['Temperature_Ave_MatID_3'] * 35
        elif scenario == 'Scenario1':
            dfs = merge_dataframes_on_datetime(met_dt, surf_temp_plot_df, surf_flux_plot_df)

        dfs.to_hdf(plot_comp_hdf, key='data')
        spatial_df.to_hdf(spatial_comp_hdf, key='data')
    else:
        dfs = pd.read_hdf(plot_comp_hdf, key='data')
        spatial_df = pd.read_hdf(spatial_comp_hdf, key='data')

    print(f"{dfs.head()=}")
    print(f"{spatial_df.head()=}")
    source = ColumnDataSource(dfs)

    # dashboard = Dashboard(source, spatial_df)
    # dashboard.view().show()
    db_model = DashboardModel(source, spatial_df, surf_mesh_file)
    db_view = DashboardView(db_model)
    db_controller = DashboardController(db_model, db_view)
    db_view.view().show()
