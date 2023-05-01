#!/usr/bin/env python3

import concurrent.futures
from datetime import date, datetime, time
import logging
import glob
import io
import re
from typing import List, Dict, Tuple, Union

from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Span
from bokeh.palettes import viridis, Set2_8
from bokeh.plotting import figure
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
    mesh_file: str, fsd_file: str, is_nodal=True
) -> pd.Series:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(
            serial_read_and_average_fsd, mesh_file, fsd_file, is_nodal
        )
        return future.result()


def compute_fsd_averages_by_material_id(
    mesh_file: str,
    temperature_files: List[str],
    start_year: int,
    column_prefix: str = "Temperature",
    is_nodal: bool = True,
    num_processors: int = 1,
) -> pd.DataFrame:
    def process_file(temperature_file):
        if num_processors == 1:
            return serial_read_and_average_fsd(mesh_file, temperature_file, is_nodal)
        else:
            return parallel_read_and_average_fsd(mesh_file, temperature_file, is_nodal)

    file_averages = {
        temperature_file: process_file(temperature_file)
        for temperature_file in temperature_files
    }

    avg_temps_df = build_fsd_dataframe(
        file_averages, start_year=start_year, column_prefix=column_prefix
    )

    return avg_temps_df

def build_image_dataframe(images: List[str], start_year: int, column_prefix: str =
    "Imagery") -> pd.DataFrame:

    datetime_file_dict = { image_filename_to_datetime(filename, start_year): filename
                           for filename in images }
    image_series = pd.Series(datetime_file_dict)
    image_series.sort_index(inplace=True)

    image_series.name = column_prefix

    return image_series.to_frame()

def build_vtk_temperature_dataframe(surface_temp_files: List[str], start_year: int,
                                    column_prefix: str = "Surface Temperature") -> \
        pd.DataFrame:
    datetime_file_dict = { fsd_filename_to_datetime(filename, start_year): filename
                             for filename in surface_temp_files }
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

def image_histogram(image_path: str, bins: int = 5):

    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')
    image_data = np.array(image)
    df = pd.DataFrame({
        'Grayscale': image_data.flatten(),
    })

    histogram = df.hvplot.hist(y='Grayscale', bins=bins, color='gray',
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
    # print(f"{len(mesh.points)=} points")
    # print(f"{len(mesh.faces)=} faces")
    # print(f"{mesh.n_cells=}")
    mesh.cell_data['MatID'] = np.array(mats, dtype=np.int32)

    return mesh

def plot_mesh(mesh: pv.PolyData) -> pv.Plotter:
    plotter = pv.Plotter(window_size=(1000, 1000))
    plotter.add_mesh(mesh, scalars='MatID', show_edges=True, show_scalar_bar=True)
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
                    legend_label: str, y_axis_label: str, span_line: Span,
                    width: int, height: int) -> figure:
    title = f"{met_column.capitalize()} vs. Date"
    met_plot = figure(title=title, x_axis_type='datetime',
                      x_axis_label='Date', y_axis_label=y_axis_label, plot_width=width,
                      plot_height=height
                      )
    met_plot.line(x='index', y=met_column, source=source, legend_label=legend_label)
    # span_line = Span(location=span_location, dimension='width', line_color='red',
    #                     line_dash='dashed', line_width=3)
    met_plot.add_layout(span_line)
    met_plot.legend.location = 'top_left'
    met_plot.legend.click_policy = 'hide'

    return met_plot

def create_variable_plot(source: ColumnDataSource, columns: List[str], title: str,
                         y_axis_label: str, span_line: Span, colors, width: int,
                         height: int) -> figure:
    var_plot = figure(title=title, x_axis_type='datetime', x_axis_label='Date',
                      y_axis_label=y_axis_label, plot_width=width, plot_height=height)
    for column in columns:
        matid = int(column.split('_')[-1])
        legend_label = f'Material ID {matid}'
        var_plot.line(x='index', y=column, source=source, line_width=2,
                      color=colors[matid-1], legend_label=legend_label)

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
        self.mesh = read_2dm_pyvista(mesh_file)
        self.current_datetime = None
        self.mat_palette_colors = viridis(self.highest_matid)
        self.vtk_mat_palette_colors = ListedColormap(viridis(self.highest_matid))
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

    # Properties
    @property
    def highest_matid(self) -> int:
        matid_cols = [col for col in self.source.data.keys() if
                      'Temperature_Ave_MatID_' in col]
        return max([int(col.split('_')[-1]) for col in matid_cols])


    def update_image_panel(self, image_path: str, palette='white hot') -> pn.pane.JPG:
        image = Image.open(image_path)
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
        if mat_palette_new == 'Viridis':
            # MatIDs are never negative or zero so subtract one for indexing
            self.mat_palette_colors = viridis(self.highest_matid)
        elif mat_palette_new == 'Set2_8':
            self.mat_palette_colors = Set2_8

    def update_vtk_mat_palette(self, mat_palette: str):
        print(f"From inside VTK mat palette: {mat_palette=}")
        mat_palette_new = mat_palette.new
        if mat_palette_new == 'Viridis':
            self.vtk_mat_palette_colors = ListedColormap(viridis(self.highest_matid))
        elif mat_palette_new == 'Set2_8':
            truncated_cmap = Set2_8[:self.highest_matid]
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
        print(f"AFter update vtk temp: {self.vtk_temp_palette_colors=}")

    def update_current_datetime(self, date_event: np.datetime64):
        date = date_event.new
        date_dt = numpy_datetime64_to_datetime(date)
        self.current_datetime = date_dt

    def update_temperatures(self):
        print(f"Updating temperatures to date {self.current_datetime}")
        temps = read_fsd(self.spatial_df.loc[self.current_datetime]['Surface '
                                                                    'Temperature'])
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
        self.plotter = plot_mesh(self.model.mesh)
        self.spacer = self.init_makeshift_spacer()
        self.match_color_palette = self.init_checkbox_match_color_palette()

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
        return pn.Column(self.image_color_palette_select, new_image_pane,
                         sizing_mode='stretch_both', height=100, width=100)

    def create_image_hist_pane(self):
        curr_image = self.model.spatial_df.loc[self.model.current_datetime]['Imagery']
        new_image_hist_pane = image_histogram(curr_image,
                                          bins=self.image_hist_bin_number_widget.value)
        return pn.Column(self.image_hist_bin_number_widget, new_image_hist_pane,
                         sizing_mode='stretch_both', height=100, width=100)

    def create_vtk_pane(self):
        return pn.Column(self.vtk_array_select, self.match_color_palette, pn.panel(
            self.plotter.ren_win, sizing_mode='stretch_both', enable_keybindings=True,
                                                              orientation_widget=True),
                         sizing_mode='stretch_both', height=200, width=200)

    @pn.depends("met_variable_select.param.value", "date_slider.param.value")
    def met_plot(self, met_variable: str, date: np.datetime64) -> pn.layout:
        met_plot = create_met_plot(source=self.model.filtered_source, met_column=met_variable,
                                   y_axis_label=met_variable, span_line=self.span_line,
                                   legend_label=met_variable, width=500, height=400)
        return met_plot

    @pn.depends("mat_palette_select.param.value", "date_slider.param.value")
    def temperature_plot(self, mat_palette_select, date: np.datetime64) -> pn.layout:
        title = 'Temperature vs. Date'
        y_axis_label = 'Temperature (°C)'
        temp_cols = [col for col in self.model.filtered_source.data.keys() if
                        'Temperature_Ave_MatID_' in col]
        return create_variable_plot(self.model.filtered_source, temp_cols, title,
                                    y_axis_label, self.span_line,
                                    self.model.mat_palette_colors, width=500,
                                    height=400)

    @pn.depends("mat_palette_select.param.value", "date_slider.param.value")
    def flux_plot(self, mat_palette_select, date: np.datetime64) -> pn.layout:
        title = 'Flux vs. Date'
        y_axis_label = 'Flux (W/m^2)'
        flux_cols = [col for col in self.model.filtered_source.data.keys() if
                        'Flux_Ave_MatID_' in col]
        return create_variable_plot(self.model.filtered_source, flux_cols, title,
                                    y_axis_label, self.span_line,
                                    self.model.mat_palette_colors, width=500,
                                    height=400)

    def update_vtk_pane_with_new_mesh(self):
        vtk_array = self.vtk_array_select.value
        print(f"Vtk array from update_vtk_pane_with_new_mesh: {vtk_array}")
        print(f"Image cmap {self.image_color_palette_select.value}")
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
        print(f"Updating VTK Pane with {vtk_array} and colormap {cmap}")
        self.plotter.remove_actor(self.plotter.renderer.GetActors().GetLastActor())
        self.plotter.add_mesh(self.model.mesh,
                              scalars=vtk_array, cmap=cmap,
                              show_scalar_bar=True, scalar_bar_args=dict(title=title),
                              name=vtk_array)
    def view(self):
        # Top Row Widgets
        self.met_plot_widget = pn.Column(self.met_variable_select, self.met_plot_bind,
                                            sizing_mode='fixed', width=500, height=500)
        self.temp_plot_widget = pn.Column(self.mat_palette_select, self.temp_plot_bind,
                                            sizing_mode='fixed', width=500, height=500)
        self.flux_plot_widget = pn.Column(self.spacer, self.flux_plot_bind,
                                            sizing_mode='fixed', width=500, height=500)
        # Bottom row Widgets
        self.image_pane = self.create_image_pane()
        self.image_hist_pane = self.create_image_hist_pane()
        self.vtk_pane = self.create_vtk_pane()

        self.bottom_row = pn.Row(self.image_pane, self.image_hist_pane, self.vtk_pane,
                                    sizing_mode='stretch_both')
        self.top_row = pn.Row(self.met_plot_widget, self.temp_plot_widget, self.flux_plot_widget,
                                sizing_mode='stretch_both')
        sidebar_params = [
            pn.pane.Markdown('## Date Range'),
            pn.pane.Markdown('### Choose the date range to graph the data'),
            self.date_range_slider, pn.Spacer(height=35), pn.layout.Divider(),
            pn.pane.Markdown('## Date'),
            pn.pane.Markdown('### Choose the particular date to view'),
            self.date_slider, pn.Spacer(height=35), pn.layout.Divider(),
            pn.Spacer(height=15), self.current_datetime_text,
            pn.Spacer(height=15), self.spatial_current_button ]
        self.template = MaterialTemplate(title='VESPA Simulation Analysis',
                                         theme=DarkTheme,
                                         main=[pn.Column(
                                               pn.pane.Markdown('# Plot Components'),
                                               self.top_row,
                                               pn.Spacer(background='#ffffff',
                                                         height=2),
                                               pn.pane.Markdown('# Spatial Components'),
                                               self.bottom_row)],
                                         sidebar=sidebar_params
                                         )
        return self.template

    def init_date_range_slider(self) -> pn.widgets.DateRangeSlider:
        return pn.widgets.DateRangeSlider(name='Date Range',
                                          start=self.model.source.data['index'][0],
                                          end=self.model.source.data['index'][-1],
                                          value=(self.model.source.data['index'][0],
                                                 self.model.source.data['index'][-1]),
                                          step=60 * 60 * 1000,
                                          tooltips=True,
                                          format='%Y-%m-%d %H:%M')

    def init_date_slider(self) -> pn.widgets.DateSlider:
        return pn.widgets.DateSlider(name='Date',
                                     as_datetime=True,
                                     start=self.model.source.data['index'][0],
                                     end=self.model.source.data['index'][-1],
                                     value=self.model.source.data['index'][0],
                                     step=60 * 60 * 1000,
                                     tooltips=True,
                                     format='%Y-%m-%d %H:%M')

    def init_met_variable_select(self) -> pn.widgets.Select:
        return pn.widgets.Select(name='Met Variable',
                                 options=['Temp', 'RH', 'Global', 'WndDir', 'WndSpd',
                                          'Direct', 'Diffuse', 'LWdown', 'Zenith',
                                          'Azimuth', 'Precip'],
                                 value='Temp', sizing_mode='fixed')

    def init_mat_palette_select(self) -> pn.widgets.Select:
        return pn.widgets.Select(name='Material Palette',
                                 options=['Viridis', 'Set2_8'],
                                 value='Viridis', sizing_mode='fixed')

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
        return pn.widgets.StaticText(name='Displaying Spatial for Date ',
                                     value=self.current_datetime_string)

    def init_spatial_current_button(self) -> pn.widgets.Button:
        return pn.widgets.Button(name='Render Current Date Spatial Components',
                                 button_type='primary')

    def init_makeshift_spacer(self) -> pn.pane.HTML:
        return pn.pane.HTML('<div style="height: 50px;"></div>')

    def init_checkbox_match_color_palette(self) -> pn.widgets.Checkbox:
        return pn.widgets.Checkbox(name="Match Image TemperatureColor Palette",
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
        # Checkbox
        # self.view.match_color_palette = pn.bind(self.model.update_vtk_temp_palette,
        #                                         self.view.image_color_palette_select)

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


# class Dashboard(param.Parameterized):
#     source: ColumnDataSource  = param.Parameter()
#     spatial_df: pd.DataFrame = param.Parameter()
#     def __init__(self, source: ColumnDataSource, spatial_df: pd.DataFrame, **params):
#         super().__init__(**params)
#         self.source = source
#         self.filtered_source = ColumnDataSource(data=self.source.data)
#         self.spatial_df = spatial_df
#
#         self.date_range_slider = self.init_date_range_slider()
#         self.date_slider = self.init_date_slider()
#         self.met_variable_select = self.init_met_variable_select()
#         self.mat_palette_select = self.init_mat_palette_select()
#         self.image_color_palette_select = self.init_image_color_palette_select()
#         self.image_hist_bin_number_widget = self.init_image_hist_bin_number_widget()
#         self.vtk_cell_array_select = self.init_vtk_cell_array_select()
#         initial_date_value = self.date_slider.value
#
#         self.span_line = Span(location=initial_date_value, dimension='height',
#                               line_color='red', line_dash='dashed', line_width=1)
#
#         self.current_datetime = initial_date_value
#
#         self.current_datetime_text = self.init_current_datetime_text()
#         self.spatial_current_button = self.init_spatial_current_button()
#
#         self.bottom_row = None
#
#         # Bind date range slider to update filtered source
#         self.date_range_slider.param.watch(self.update_filtered_source, 'value')
#         self.date_slider.param.watch(self.update_span_line, 'value')
#         self.date_slider.param.watch(self.update_current_datetime, 'value')
#         self.image_color_palette_select.param.watch(self.update_image_pane, 'value')
#         self.image_hist_bin_number_widget.param.watch(self.update_image_hist_pane,
#                                                       'value_throttled')
#         self.vtk_cell_array_select.param.watch(self.update_vtk_pane, 'value')
#         self.mat_palette_colors = viridis(self.highest_matid)
#         self.mat_palette_select.param.watch(self.update_mat_palette, 'value')
#         self.mat_palette_select.param.watch(self.update_vtk_pane, 'value')
#
#     def init_date_range_slider(self):
#          return pn.widgets.DateRangeSlider(name='Date Range',
#                                            start=self.source.data['index'][0],
#                                            end=self.source.data['index'][-1],
#                                            value=(self.source.data['index'][0],
#                                            self.source.data['index'][-1]),
#                                            step=60 * 60 * 1000,
#                                            tooltips=True,
#                                            format='%Y-%m-%d %H:%M')
#
#     def init_date_slider(self):
#         return pn.widgets.DateSlider(name='Date',
#                                      as_datetime=True,
#                                      start=self.source.data['index'][0],
#                                      end=self.source.data['index'][-1],
#                                      value=self.source.data['index'][0],
#                                      step=60 * 60 * 1000,
#                                      tooltips=True,
#                                      format='%Y-%m-%d %H:%M')
#
#     def init_met_variable_select(self):
#         return pn.widgets.Select(name='Met Variable',
#                                  options=['Temp', 'RH', 'Global'],
#                                  value='Temp')
#
#     def init_mat_palette_select(self):
#         return pn.widgets.Select(name='Material Palette',
#                                  options=['Viridis', 'Set2_8'],
#                                  value='Viridis')
#
#     def init_image_color_palette_select(self):
#         return pn.widgets.Select(name='Image Color Palette',
#                                  options=['flir',
#                                           'rainbow1234',
#                                           'yellow',
#                                           'white hot',
#                                           'black hot',
#                                           'artic',
#                                           'plasma',
#                                           'lava',
#                                           ],
#                                  value='white hot',
#                                  sizing_mode='fixed')
#
#     def init_image_hist_bin_number_widget(self):
#         return pn.widgets.IntSlider(name='Image Histogram Bin Number',
#                                     start=2, end=20, step=1, value=5)
#
#     def init_vtk_cell_array_select(self):
#         return pn.widgets.Select(name='VTK Cell Array',
#                                  options=['MatID',
#                                           'Temperature'],
#                                  value='MatID',
#                                  sizing_mode='fixed')
#
#     def init_current_datetime_text(self):
#         return pn.widgets.StaticText(name='Displaying Spatial for Date ',
#                                      value=self.current_datetime_string)
#
#     def init_spatial_current_button(self):
#         return pn.widgets.Button(name='Render Current Date Spatial Components',
#                                  button_type='primary')
#
#     @property
#     def highest_matid(self):
#         # Get the highest {num} Temperature_Ave_MatID_{num} columns
#         matid_cols = [col for col in self.source.data.keys() if
#                       'Temperature_Ave_MatID_' in col]
#         matid_nums = [int(col.split('_')[-1]) for col in matid_cols]
#         return max(matid_nums)
#
#     @property
#     def current_datetime_string(self) -> str:
#         date_dt = numpy_datetime64_to_datetime(self.current_datetime)
#         return f'{date_dt:%Y-%m-%d %H:%M}'
#
#
#     @pn.depends("date_range_slider.param.value")
#     def update_date_slider(self, event):
#         date_range = event.new
#         start, end = np.datetime64(date_range[0]), np.datetime64(date_range[1])
#         self.date_slider.start = start
#         self.date_slider.end = end
#         if self.date_slider.value < start:
#             self.date_slider.value = start
#         elif self.date_slider.value > end:
#             self.date_slider.value = end
#
#     @pn.depends("image_color_palette_select.param.value")
#     def update_image_pane(self, event=None):
#         palette = self.image_color_palette_select.value
#         curr_image = self.spatial_df.loc[self.current_datetime]['Imagery']
#         new_image_pane = read_image_panel(curr_image, palette=palette)
#         self.image_pane = pn.Column(self.image_color_palette_select, new_image_pane,
#                                     sizing_mode='stretch_both', height=100, width=100)
#         self.bottom_row[0] = self.image_pane
#
#     @pn.depends("image_hist_bin_number_widget.param.value")
#     def update_image_hist_pane(self, event=None):
#         bin_number = self.image_hist_bin_number_widget.value
#         curr_image = self.spatial_df.loc[self.current_datetime]['Imagery']
#         new_image_hist_pane = image_histogram(curr_image, bins=bin_number)
#         # print(f"Updating image hist pane with {bin_number} bins")
#         self.image_hist = pn.Column(self.image_hist_bin_number_widget,
#                                     new_image_hist_pane, sizing_mode='stretch_both',
#                                     height=100, width=100)
#         self.bottom_row[1] = self.image_hist
#
#     @pn.depends("mat_palette_colors.param.value", "vtk_cell_array_select.param.value")
#     def update_vtk_pane(self, mat_palette_colors, event=None):
#         vtk_array = self.vtk_cell_array_select.value
#         if vtk_array == 'Temperature':
#             self.plotter.remove_actor(self.plotter.renderer.GetActors().GetLastActor())
#             temps = read_fsd(self.spatial_df.loc[self.current_datetime][
#                                  'Surface Temperature'])
#             self.mesh.point_data['Temperature'] = np.array(temps, dtype=float)
#             self.plotter.add_mesh(self.mesh, scalars='Temperature', cmap='plasma',
#                                     show_scalar_bar=True, scalar_bar_args=dict(
#                     title='Temperature (C)'), name='Temperature')
#         elif vtk_array == 'MatID':
#             # bokeh_viridis = Set2_8
#             # rgba_colors = [tuple(np.array(color, dtype=float) / 255) for color in
#             #                bokeh_viridis]
#             if self.mat_palette_select.value == 'Viridis':
#                 pyvista_cmap = ListedColormap(self.mat_palette_colors)
#             elif self.mat_palette_select.value == 'Set2_8':
#                 bokeh_set2_8 = Set2_8[:self.highest_matid]
#                 # rgba_colors = [to_rgba(color) for color in bokeh_set2_8]
#                 # print(f"{rgba_colors=}")
#                 pyvista_cmap = ListedColormap(bokeh_set2_8)
#             #print(f"Updating vtk color palette with {pyvista_cmap}")
#
#             self.plotter.remove_actor(self.plotter.renderer.GetActors().GetLastActor())
#             self.plotter.add_mesh(self.mesh, scalars='MatID',
#                                     show_scalar_bar=True,
#                                   cmap=pyvista_cmap, scalar_bar_args=dict(
#                     title='Material ID'), name='MatID')
#
#         self.vtk_pane = pn.Column(self.vtk_cell_array_select,
#                                   pn.panel(self.plotter.ren_win,
#                                            sizing_mode='stretch_both',
#                                            enable_keybindings=True, orientation_widget=True),
#                                   sizing_mode='stretch_both',
#                                   width=200, height=200)
#         self.bottom_row[2] = self.vtk_pane
#
#
#     @pn.depends("date_slider.param.value")
#     def update_span_line(self, date_event):
#         date_value = np.datetime64(self.date_slider.value)
#         self.span_line = Span(location=date_value, dimension='height',
#                               line_color='red', line_dash='dashed', line_width=1)
#
#     @pn.depends("date_range_slider.param.value")
#     def update_filtered_source(self, event):
#         date_range = event.new
#         start, end = np.datetime64(date_range[0]), np.datetime64(date_range[1])
#         mask = (self.source.data['index'] >= start) & (self.source.data['index'] <= end)
#         self.filtered_source.data = {key: value[mask] for key, value in self.source.data.items()}
#         self.update_date_slider(event)
#
#     @pn.depends("met_variable_select.param.value", "date_slider.param.value")
#     def update_met_plot(self, met_variable: str,  date: np.datetime64, event=None) -> \
#             pn.layout:
#         met_column = met_variable
#         y_axis_label = met_variable
#         legend_label = met_variable
#         return create_met_plot(self.filtered_source, met_column, legend_label,
#                                y_axis_label, self.span_line)
#
#     @pn.depends("mat_palette_select.param.value")
#     def update_mat_palette(self, mat_palette):
#         mat_palette_new = mat_palette.new
#         # print(f"Updating mat palette to {mat_palette_new}")
#         if mat_palette_new == 'Viridis':
#             # MatID are never negative or zero so subract 1 for indexing
#             self.mat_palette_colors = viridis(self.highest_matid)
#         elif mat_palette_new == 'Set2_8':
#             self.mat_palette_colors = Set2_8
#
#     @pn.depends("mat_palette_colors.param.value", "date_slider.param.value")
#     def temperature_plot(self, mat_palette_colors, date: np.datetime64) -> pn.layout:
#         title = 'Temperature vs. Date'
#         y_axis_label = 'Temperature (°C)'
#         print(f"Updating temperature plot with {mat_palette_colors}")
#         print(f"Checking out if self is the same value {self.mat_palette_colors}")
#         return create_variable_plot(self.filtered_source,
#                                     [x for x in self.filtered_source.data.keys()
#                                         if x.startswith('Temperature')],
#                                     title, y_axis_label, self.span_line,
#                                     self.mat_palette_colors)
#
#     @pn.depends("mat_palette_colors.param.value", "date_slider.param.value")
#     def flux_plot(self, mat_palette_colors, date: np.datetime64) -> pn.layout:
#         title = 'Flux vs. Date'
#         y_axis_label = 'Flux (W/m^2)'
#         return create_variable_plot(self.filtered_source,
#                                     [x for x in self.filtered_source.data.keys()
#                                                   if x.startswith('Flux')],
#                                     title, y_axis_label, self.span_line,
#                                     self.mat_palette_colors)
#
#     @pn.depends("date_slider.param.value")
#     def update_current_datetime(self, date_event: np.datetime64):
#         date = date_event.new
#         date_dt = numpy_datetime64_to_datetime(date)
#         self.current_datetime = date_dt
#
#     def on_spatial_current_datetime_change(self, event):
#         self.current_datetime_text.value = self.current_datetime_string
#         curr_image = self.spatial_df.loc[self.current_datetime]['Imagery']
#         new_image_pane = read_image_panel(curr_image, palette=self.image_color_palette_select.value)
#         self.image_pane = pn.Column(self.image_color_palette_select, new_image_pane,
#                                     sizing_mode='stretch_both', height=100, width=100)
#         new_hist_pane = image_histogram(curr_image,
#                                         bins=self.image_hist_bin_number_widget.value)
#         self.image_hist = pn.Column(self.image_hist_bin_number_widget, new_hist_pane,
#                                     sizing_mode='stretch_both')
#         self.bottom_row[0] = self.image_pane
#         self.bottom_row[1] = self.image_hist
#
#
#     def view(self):
#
#         # Button
#         self.spatial_current_button.on_click(self.on_spatial_current_datetime_change)
#
#         # Plots
#         self.met_plot = pn.bind(self.update_met_plot, self.met_variable_select,
#                            self.date_slider.param.value, self.date_slider.param.value)
#         self.temp_plot = pn.bind(self.temperature_plot, self.mat_palette_select,
#                                      self.date_slider.param.value)
#         self.flux_plot = pn.bind(self.flux_plot, self.mat_palette_select,
#                                            self.date_slider.param.value)
#
#         # Widgets
#         self.met_plot_widget = pn.Column(self.met_variable_select, self.met_plot,
#                                     sizing_mode='stretch_both', width=500, height=400)
#
#         self.temp_plot_widget = pn.Column(self.mat_palette_select,
#                                           self.temp_plot,
#                                      sizing_mode='stretch_both',
#                                      width=500,
#                                      height=400)
#
#         self.flux_plot_widget = pn.Column(pn.Spacer(height=50), self.flux_plot,
#                                      sizing_mode='stretch_both',
#                                      width=500,
#                                      height=400)
#
#         # Spatial VTK Plot
#         self.mesh = read_2dm_pyvista("data/Scenario1_mats.2dm")
#         self.plotter = plot_mesh(self.mesh)
#         self.vtk_plot = pn.panel(self.plotter.ren_win,
#                                  sizing_mode='stretch_both',
#                                  enable_keybindings=True, orientation_widget=True,
#                                  width=200, height=200)
#         self.vtk_pane = pn.Column(self.vtk_cell_array_select,
#                                     self.vtk_plot,
#                                     sizing_mode='stretch_both',
#                                     width=500, height=500)
#
#         # Image Plot
#         curr_image = self.spatial_df.loc[self.current_datetime]['Imagery']
#         self.image_pane = pn.Column(self.image_color_palette_select, read_image_panel(
#             curr_image, palette=self.image_color_palette_select.value),
#                                     sizing_mode='stretch_both', height=100, width=100)
#         self.image_pane.sizing_mode = 'stretch_both'
#         self.image_pane.width = 100
#         self.image_pane.height = 100
#
#         # Image Histogram Plot
#         self.image_hist = pn.Column(self.image_hist_bin_number_widget,
#                                     image_histogram(curr_image, bins=5),
#                                     sizing_mode='stretch_both')
#         self.image_hist.sizing_mode = 'stretch_both'
#         self.image_hist.width = 100
#         self.image_hist.height = 100
#
#         self.bottom_row = pn.Row(self.image_pane, self.image_hist, self.vtk_pane,
#                          sizing_mode='stretch_both')
#         top_row = pn.Row(self.met_plot_widget, self.temp_plot_widget,
#                          self.flux_plot_widget, sizing_mode='stretch_both')
#
#         self.template = MaterialTemplate(title="VESPA Simulation Analysis",
#                                          theme=DarkTheme,
#                                     main=[pn.Column(pn.pane.Markdown('## Plot Components'),
#                                                     top_row,
#                                                     pn.Spacer( background='#ffffff',
#                                                                height=5),
#                                         pn.pane.Markdown('## Spatial Components'),
#                                                             self.bottom_row),
#                                           ],
#                                     sidebar=[pn.pane.Markdown('## Date Range'),
#                                              pn.pane.Markdown('### Choose the date '
#                                                               'range to graph the '
#                                                               'data'),
#                                              self.date_range_slider,
#                                              pn.Spacer(height=35),
#                                              pn.layout.Divider(),
#                                              pn.pane.Markdown('## Date'),
#                                              pn.pane.Markdown('### Choose the '
#                                                               'particular date to '
#                                                               'view'),
#                                              self.date_slider,
#                                              pn.Spacer(height=35),
#                                              pn.layout.Divider(),
#                                              pn.Spacer(height=15),
#                                              self.current_datetime_text,
#                                              pn.Spacer(height=15),
#                                              self.spatial_current_button],
#                                     )
#         return self.template

if __name__ == "__main__":
    met = read_met('data/Scenario1.met')
    met_dt = set_met_index_to_datetime(met, 2022)
    temp_df = compute_fsd_averages_by_material_id(
        'data/Scenario1_mats.2dm',
        glob.glob("data/file_sock300*.fsd"),
        2022,
        'Temperature',
    )


    flux_df = compute_fsd_averages_by_material_id(
        'data/Scenario1_mats.2dm',
        glob.glob("data/file_sock200*.fsd"),
        2022,
        'Flux',
        False,
    )

    image_df = build_image_dataframe(
        glob.glob("data/Scenario1*.jpg"),
        2022,
        'Imagery'
    )

    mesh_temp_df = build_vtk_temperature_dataframe(
        glob.glob("data/file_sock300*.fsd"),
        2022,
        'Surface Temperature'
    )

    mesh_flux_df = build_vtk_flux_dataframe(
        glob.glob("data/file_sock200*.fsd"),
        2022,
        'Surface Flux'
    )



    spatial_df = merge_dataframes_on_datetime(image_df, mesh_temp_df, mesh_flux_df)

    dfs = merge_dataframes_on_datetime(met_dt, temp_df, flux_df)
    source = ColumnDataSource(dfs)

    # dashboard = Dashboard(source, spatial_df)
    # dashboard.view().show()
    db_model = DashboardModel(source, spatial_df, 'data/Scenario1_mats.2dm')
    db_view = DashboardView(db_model)
    db_controller = DashboardController(db_model, db_view)
    db_view.view().show()
