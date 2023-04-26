#!/usr/bin/env python3

import concurrent.futures
from datetime import date, datetime, time
import logging
import glob
import re
from typing import List, Dict, Tuple, Union

from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Span
from bokeh.models import CustomJS
from bokeh.plotting import figure, show
import hvplot.pandas as hvplot
import numpy as np
import pandas as pd
import panel as pn
from panel.template import FastListTemplate, MaterialTemplate, FastGridTemplate
from panel.template.theme import DarkTheme
from panel.layout.gridstack import GridStack
import param
from PIL import Image
import pyvista as pv
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

    return image_series

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
def read_image_panel(image_path: str) -> pn.pane.JPG:
    return pn.pane.JPG(image_path, width=500, height=500)

def display_image(image_path: str) -> pn.Column:
    # image_panes = [read_image_panel(image_path) for image_path in image_paths]
    image_pane = read_image_panel(image_path)
    return image_pane

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
    print(f"{len(mesh.points)=} points")
    print(f"{len(mesh.faces)=} faces")
    print(f"{mesh.n_cells=}")
    mesh.cell_data['MatID'] = np.array(mats, dtype=np.int32)

    return mesh

def plot_mesh(mesh_file: str) -> pv.Plotter:
    mesh = read_2dm_pyvista(mesh_file)
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
                    legend_label: str, y_axis_label: str, span_line: Span) -> figure:
    title = f"{met_column.capitalize()} vs. Date"
    met_plot = figure(title=title, x_axis_type='datetime',
                      x_axis_label='Date', y_axis_label=y_axis_label,
                      )
    met_plot.line(x='index', y=met_column, source=source, legend_label=legend_label)
    # span_line = Span(location=span_location, dimension='width', line_color='red',
    #                     line_dash='dashed', line_width=3)
    met_plot.add_layout(span_line)
    met_plot.legend.location = 'top_left'
    met_plot.legend.click_policy = 'hide'

    return met_plot

def create_variable_plot(source: ColumnDataSource, columns: List[str], title: str,
                         y_axis_label: str, span_line: Span) -> figure:
    var_plot = figure(title=title, x_axis_type='datetime', x_axis_label='Date',
                      y_axis_label=y_axis_label)
    colors = ["blue", "green", "orange", "cyan", "magenta", "purple", "brown", "pink"]
    for i, column in enumerate(columns):
        var_plot.line(x='index', y=column, source=source, line_width=2,
                      color=colors[i], legend_label=column)

    # span_line = Span(location=span_location, dimension='width', line_color='red',
    #                     line_dash='dashed', line_width=3)
    var_plot.add_layout(span_line)
    var_plot.legend.location = 'top_left'
    var_plot.legend.click_policy = 'hide'
    return var_plot


class Dashboard(param.Parameterized):
    def __init__(self, source: ColumnDataSource, spatial_df: pd.DataFrame, **params):
        super().__init__(**params)
        self.source = source
        self.filtered_source = ColumnDataSource(data=self.source.data)
        self.spatial_df = spatial_df

        self.date_range_slider = pn.widgets.DateRangeSlider(
            name='Range', start=source.data['index'][0],
            end=source.data['index'][-1], value=(source.data['index'][0],
                                                source.data['index'][-1]),
            step=60 * 60 * 1000, tooltips=True, format='%Y-%m-%d %H:%M')

        self.date_slider = pn.widgets.DateSlider(name='Date',
                                                 as_datetime=True,
                                                 start=source.data['index'][0],
                                                 end=source.data['index'][-1],
                                                 value=source.data['index'][0],
                                                 step=60 * 60 * 1000,
                                                 tooltips=True,
                                                 format='%Y-%m-%d %H:%M')

        self.met_variable_select = pn.widgets.Select(name='Met Variable',
                                                 options=['Temp', 'RH', 'Global'],
                                                 value='Temp')

        initial_date_value = self.date_slider.value
        if not self.filtered_source.data['index'].min() <= initial_date_value <= \
                self.filtered_source.data['index'].max():
            print(f"Span location is NOT within range of filtered source")
            print(f"{self.filtered_source.data['index'].min()=}")
            print(f"{self.filtered_source.data['index'].max()=}")
            print(f"{initial_date_value=}")

        self.span_line = Span(location=initial_date_value, dimension='height',
                              line_color='red', line_dash='dashed', line_width=1)

        self.current_datetime = initial_date_value

        self.current_datetime_text =  pn.widgets.StaticText(name='Displaying Spatial '
                                                                 'for Date ',
                                                            value=self.current_datetime_string())
        self.spatial_current_button = pn.widgets.Button(name='Render Current '
                                                             'Date Spatial Components',
                                                        button_type='primary')
        self.bottom_row = None

        # Bind date range slider to update filtered source
        self.date_range_slider.param.watch(self.update_filtered_source, 'value')
        self.date_slider.param.watch(self.update_span_line, 'value')
        self.date_slider.param.watch(self.update_current_datetime, 'value')

    @pn.depends("date_range_slider.param.value")
    def update_date_slider(self, event):
        date_range = event.new
        start, end = np.datetime64(date_range[0]), np.datetime64(date_range[1])
        self.date_slider.start = start
        self.date_slider.end = end
        if self.date_slider.value < start:
            self.date_slider.value = start
        elif self.date_slider.value > end:
            self.date_slider.value = end

    @pn.depends("date_slider.param.value")
    def update_span_line(self, date_event):
        date_value = np.datetime64(self.date_slider.value)
        self.span_line = Span(location=date_value, dimension='height',
                              line_color='red', line_dash='dashed', line_width=1)

    @pn.depends("date_range_slider.param.value")
    def update_filtered_source(self, event):
        date_range = event.new
        start, end = np.datetime64(date_range[0]), np.datetime64(date_range[1])
        mask = (self.source.data['index'] >= start) & (self.source.data['index'] <= end)
        self.filtered_source.data = {key: value[mask] for key, value in self.source.data.items()}
        self.update_date_slider(event)

    @pn.depends("met_variable_select.param.value", "date_slider.param.value")
    def update_met_plot(self, met_variable: str,  date: np.datetime64, event=None) -> \
            pn.layout:
        met_column = met_variable
        y_axis_label = met_variable
        legend_label = met_variable
        return create_met_plot(self.filtered_source, met_column, legend_label,
                               y_axis_label, self.span_line)

    @pn.depends("date_slider.param.value")
    def temperature_plot(self, date: np.datetime64) -> pn.layout:
        title = 'Temperature vs. Date'
        y_axis_label = 'Temperature (°C)'
        return create_variable_plot(self.filtered_source,
                                    [x for x in self.filtered_source.data.keys()
                                        if x.startswith('Temperature')],
                                    title, y_axis_label, self.span_line)

    @pn.depends("date_slider.param.value")
    def flux_plot(self, date: np.datetime64) -> pn.layout:
        title = 'Flux vs. Date'
        y_axis_label = 'Flux (W/m^2)'
        return create_variable_plot(self.filtered_source,
                                    [x for x in self.filtered_source.data.keys()
                                                  if x.startswith('Flux')],
                                    title, y_axis_label, self.span_line)

    @pn.depends("date_slider.param.value")
    def update_current_datetime(self, date_event: np.datetime64):
        date = date_event.new
        date_dt = numpy_datetime64_to_datetime(date)
        self.current_datetime = date_dt

    def current_datetime_string(self):
        print(f"From current_datetime_string: {self.current_datetime=}")
        date_dt = numpy_datetime64_to_datetime(self.current_datetime)
        return f'{date_dt:%Y-%m-%d %H:%M}'

    def on_spatial_current_datetime_change(self, event):
        self.current_datetime_text.value = self.current_datetime_string()
        print(f"Clicked on spatial current datetime button at {self.current_datetime_string()}")
        print(f"On click: {self.spatial_df.loc[self.current_datetime]['Imagery']=}")
        curr_image = self.spatial_df.loc[self.current_datetime]['Imagery']
        new_image_pane = display_image(curr_image)
        self.image_pane = pn.Column(new_image_pane)
        new_hist_pane = image_histogram(curr_image)
        self.image_hist = pn.Column(new_hist_pane, sizing_mode='stretch_both')
        self.bottom_row[0] = self.image_pane
        self.bottom_row[1] = self.image_hist
        print(f"On spatial after click: {self.image_pane=}")
        print(f"On spatial after click type: {type(self.image_pane)=}")


    def view(self):

        # Button
        self.spatial_current_button.on_click(self.on_spatial_current_datetime_change)

        # Plots
        met_plot = pn.panel(pn.bind(self.update_met_plot, self.met_variable_select,
                           self.date_slider.param.value, self.date_slider.param.value))
        temp_plot = pn.panel(pn.bind(self.temperature_plot,
                                     self.date_slider.param.value))
        flux_plot = pn.panel(pn.bind(self.flux_plot, self.date_slider.param.value))

        # Widgets
        met_plot_widget = pn.Column(self.met_variable_select, met_plot,
                                    sizing_mode='stretch_both', width=100, height=75)

        temp_plot_widget = pn.Column(pn.Spacer(height=50), temp_plot,
                                     sizing_mode='stretch_both',
                                     width=100,
                                     height=75)

        flux_plot_widget = pn.Column(pn.Spacer(height=50), flux_plot,
                                     sizing_mode='stretch_both',
                                     width=100,
                                     height=75)

        # Spatial VTK Plot
        plotter = plot_mesh("data/Scenario1_mats.2dm")
        vtk_pane = pn.panel(plotter.ren_win, sizing_mode='stretch_both', width=200,
                            height=200)

        # Image Plot
        curr_image = self.spatial_df.loc[self.current_datetime]['Imagery']
        self.image_pane = pn.Column(display_image(curr_image))
        self.image_pane.sizing_mode = 'stretch_both'
        self.image_pane.width = 100
        self.image_pane.height = 100

        # Image Histogram Plot
        self.image_hist = pn.Column(image_histogram(curr_image),
                                    sizing_mode='stretch_both')
        self.image_hist.sizing_mode = 'stretch_both'
        self.image_hist.width = 100
        self.image_hist.height = 100

        self.bottom_row = pn.Row(self.image_pane, self.image_hist, vtk_pane,
                         sizing_mode='stretch_both')
        top_row = pn.Row(met_plot_widget, temp_plot_widget, flux_plot_widget,
                            sizing_mode='stretch_both')

        self.template = MaterialTemplate(title="VESPA Simulation Analysis",
                                         theme=DarkTheme,
                                    main=[pn.Column(pn.pane.Markdown('## Plot Components'),
                                                    top_row,
                                                    pn.Spacer( background='#ffffff',
                                                               height=5),
                                        pn.pane.Markdown('## Spatial Components'),
                                                            self.bottom_row)],
                                    sidebar=[pn.pane.Markdown('## Date Range'),
                                             pn.pane.Markdown('### Choose the date '
                                                              'range to graph the '
                                                              'data'),
                                             self.date_range_slider,
                                             pn.Spacer(height=35),
                                             pn.layout.Divider(),
                                             pn.pane.Markdown('## Date'),
                                             pn.pane.Markdown('### Choose the '
                                                              'particular date to '
                                                              'view'),
                                             self.date_slider,
                                             pn.Spacer(height=35),
                                             pn.layout.Divider(),
                                             pn.Spacer(height=15),
                                             self.current_datetime_text,
                                             pn.Spacer(height=15),
                                             self.spatial_current_button]
                                    )
        print(f"In View: {self.image_pane=}")
        print(f"In View type: {type(self.image_pane)=}")
        return self.template

if __name__ == "__main__":
    met = read_met("data/Scenario1.met")
    met_dt = set_met_index_to_datetime(met, 2022)
    temp_df = compute_fsd_averages_by_material_id(
        "data/Scenario1_mats.2dm",
        glob.glob("data/file_sock300*.fsd"),
        2022,
        "Temperature",
    )


    flux_df = compute_fsd_averages_by_material_id(
        "data/Scenario1_mats.2dm",
        glob.glob("data/file_sock200*.fsd"),
        2022,
        "Flux",
        False,
    )

    image_df = build_image_dataframe(
        glob.glob("data/Scenario1*.jpg"),
        2022,
        "Imagery"
    )

    spatial_df = image_df.to_frame()

    dfs = merge_dataframes_on_datetime(met_dt, temp_df, flux_df)

    source = ColumnDataSource(dfs)

    dashboard = Dashboard(source, spatial_df)
    dashboard.view().show()