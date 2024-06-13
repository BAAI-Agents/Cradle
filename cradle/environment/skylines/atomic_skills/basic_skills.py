from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment.skylines.skill_registry import register_skill
import time

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("open_roads_menu")
def open_roads_menu():
    """
    The function to open the roads options in the lower menu bar for further determination of which types of roads to build.
    """
    io_env.mouse_move(625, 1015)
    io_env.mouse_click_button("left", clicks=1)


@register_skill("open_electricity_menu")
def open_electricity_menu():
    """
    The function to open the electricity options in the lower menu bar for further determination of which types of power facility to build.
    """
    io_env.mouse_move(790, 1015)
    io_env.mouse_click_button("left", clicks=1)
    # time.sleep(1)
    # io_env.mouse_drag(200, 80, 1920, 0)


@register_skill("open_water_sewage_menu")
def open_water_sewage_menu():
    """
    The function to open the water and sewage options in the lower menu bar for further determination of which types of water and sewage to build.
    """
    io_env.mouse_move(835, 1015)
    io_env.mouse_click_button("left", clicks=1)
    # time.sleep(1)
    # io_env.mouse_drag(200, 80, 1920, 0)
    

@register_skill("open_zoning_menu")
def open_zoning_menu():
    """
    The function to open the zoning options in the lower menu bar for further determination of which types of zonings to build.
    """
    io_env.mouse_move(670, 1015)
    io_env.mouse_click_button("left", clicks=1)


@register_skill("try_place_two_lane_road")
def try_place_two_lane_road(x1=0, y1=0, x2=1920, y2=840):
    """
    Previews the placement of a road between two specified points, (x1, y1) and (x2, y2). This function does not actually construct the road, but rather displays a visual representation of where the road would be placed if confirmed.

    Parameters:
     - x1: The x-coordinate of start point of the road. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of start point of the road. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of end point of the road. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of end point of the road. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)

    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select the type of roads as two-lane road in the panel
    io_env.mouse_move(700, 930, duration=0.1)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the road
    io_env.mouse_move(x1, y1, duration=0.1)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)
    io_env.mouse_move(x2, y2, duration=0.1)


@register_skill("try_place_wind_turbine")
def try_place_wind_turbine(x=0, y=0):
    """
    Previews the placement of a wind turbine on point, (x, y). This function does not actually construct the wind turbine, but rather displays a visual representation of where the wind turbine would be placed if confirmed.

    Parameters:
     - x: The x-coordinate of the wind turbine. 0 <= x <= 1920. The default value is 0.
     - y: The y-coordinate of the wind turbine. 0 <= y <= 840. The default value is 0.
    """
    x, y = _check_valid_coordinates(x, y)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select the type of the electricity as wind turbine in the panel
    io_env.mouse_move(825, 930)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the wind turbine
    io_env.mouse_move(x, y, duration=0.1)


@register_skill("try_place_power_line")
def try_place_power_line(x1=0, y1=0, x2=1920, y2=840):
    """
    Previews the placement of a power line between two specified points, (x1, y1) and (x2, y2). This function does not actually construct the power line, but rather displays a visual representation of where the power line would be placed if confirmed.

    Parameters:
     - x1: The x-coordinate of start point of the power line. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of start point of the power line. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of end point of the power line. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of end point of the power line. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select the type of the electricity as power line in the panel
    io_env.mouse_move(700, 930)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the power line
    io_env.mouse_move(x1, y1, duration=0.1)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)
    io_env.mouse_move(x2, y2, duration=0.1)


@register_skill("try_place_water_pumping_station")
def try_place_water_pumping_station(x=0, y=0):
    """
    Previews the placement of a water pumping station on point, (x, y). This function does not actually construct the water pumping station, but rather displays a visual representation of where the water pumping station would be placed if confirmed.

    Parameters:
     - x: The x-coordinate of the water pumping station. 0 <= x <= 1920. The default value is 0.
     - y: The y-coordinate of the water pumping station. 0 <= y <= 840. The default value is 0.
    """
    x, y = _check_valid_coordinates(x, y)

    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select the type of the water & sewage as water pumping station in the panel
    io_env.mouse_move(825, 930)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the water pumping station
    io_env.mouse_move(x, y, duration=0.1)


@register_skill("try_place_water_pipe")
def try_place_water_pipe(x1=0, y1=0, x2=1920, y2=840):
    """
    Previews the placement of a water pipe between two specified points, (x1, y1) and (x2, y2). This function does not actually construct the water pipe, but rather displays a visual representation of where the water pipe would be placed if confirmed.

    Parameters:
     - x1: The x-coordinate of start point of the water pipe. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of start point of the water pipe. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of end point of the water pipe. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of end point of the water pipe. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select the type of the water & sewage as water pipe in the panel
    io_env.mouse_move(700, 930)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the water pipe
    io_env.mouse_move(x1, y1, duration=0.1)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)
    io_env.mouse_move(x2, y2, duration=0.1)


@register_skill("try_place_water_drain_pipe")
def try_place_water_drain_pipe(x=0, y=0):
    """
    Previews the placement of a water drain pipe on point, (x, y). This function does not actually construct the water drain pipe, but rather displays a visual representation of where the water drain pipe would be placed if confirmed.

    Parameters:
     - x: The x-coordinate of the water drain pipe. 0 <= x <= 1920. The default value is 0.
     - y: The y-coordinate of the water drain pipe. 0 <= y <= 840. The default value is 0.
    """
    x, y = _check_valid_coordinates(x, y)

    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select the type of the water & sewage as water drain pipe in the panel
    io_env.mouse_move(1025, 930)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the water drain pipe
    io_env.mouse_move(x, y, duration=0.1)


@register_skill("try_place_residential_zone")
def try_place_residential_zone(x1=0, y1=0, x2=1920, y2=840):
    """
    Previews the placement of a residential zone within a rectangular region with diagonal corners at (x1, y1) and (x2, y2). This function does not actually construct the residential zone, but rather displays a visual representation of where the residential zone would be placed if confirmed.

    Parameters:
     - x1: The x-coordinate of start point of the residential zone. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of start point of the residential zone. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of end point of the residential zone. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of end point of the residential zone. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)
    
    # Cover the left side of the road
    x1 = max(x1 - 90, 100)
    y1 = max(y1 - 90, 100)
    x2 = min(x2 + 90, 1920)
    y2 = min(y2 + 90, 875)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)

    # Select Marquee
    io_env.mouse_move(384, 960)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Select the type of zonings as residential zone in the panel
    io_env.mouse_move(700, 950)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the residential zone
    io_env.mouse_move(x1, y1)
    time.sleep(1)
    io_env.mouse_hold('left')
    time.sleep(1)
    io_env.mouse_move(x2, y2)


@register_skill("try_place_commercial_zone")
def try_place_commercial_zone(x1=0, y1=0, x2=1920, y2=840):
    """
    Previews the placement of a commercial zone within a rectangular region with diagonal corners at (x1, y1) and (x2, y2). This function does not actually construct the commercial zone, but rather displays a visual representation of where the commercial zone would be placed if confirmed.

    Parameters:
     - x1: The x-coordinate of start point of the commercial zone. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of start point of the commercial zone. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of end point of the commercial zone. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of end point of the commercial zone. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)

    # Cover the left side of the road
    x1 = max(x1 - 90, 100)
    y1 = max(y1 - 90, 100)
    x2 = min(x2 + 90, 1920)
    y2 = min(y2 + 90, 875)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)

    # Select Marquee
    io_env.mouse_move(384, 960)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Select the type of zonings as commercial zone in the panel
    io_env.mouse_move(910, 950)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)
    
    # Try to place the commercial zone
    io_env.mouse_move(x1, y1)
    time.sleep(1)
    io_env.mouse_hold('left')
    time.sleep(1)
    io_env.mouse_move(x2, y2)
    

@register_skill("try_place_industrial_zone")
def try_place_industrial_zone(x1=0, y1=0, x2=1920, y2=840):
    """
    Previews the placement of a industrial zone within a rectangular region with diagonal corners at (x1, y1) and (x2, y2). This function does not actually construct the industrial zone, but rather displays a visual representation of where the industrial zone would be placed if confirmed.

    Parameters:
     - x1: The x-coordinate of start point of the industrial zone. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of start point of the industrial zone. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of end point of the industrial zone. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of end point of the industrial zone. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)

    # Cover the left side of the road
    x1 = max(x1 - 90, 100)
    y1 = max(y1 - 90, 100)
    x2 = min(x2 + 90, 1920)
    y2 = min(y2 + 90, 875)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)

    # Select Marquee
    io_env.mouse_move(384, 960)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Select the type of zonings as residential zone in the panel
    io_env.mouse_move(1140, 950)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to place the zone
    io_env.mouse_move(x1, y1)
    time.sleep(1)
    io_env.mouse_hold('left')
    time.sleep(1)
    io_env.mouse_move(x2, y2)


@register_skill("try_de_zone")
def try_de_zone(x1=0, y1=0, x2=1920, y2=840):
    """
    The function to remove the zone in the game. The zone must cover the road.

    Parameters:
     - x1: The x-coordinate of top left corner point of the zone. 0 <= x1 <= 1920. The default value is 0.
     - y1: The y-coordinate of top left corner of the zone. 0 <= y1 <= 840. The default value is 0.
     - x2: The x-coordinate of bottom right point of the zone. 0 <= x2 <= 1920. The default value is 1920.
     - y2: The y-coordinate of bottom right point of the zone. 0 <= y2 <= 840. The default value is 840.
    """
    x1, y1 = _check_valid_coordinates(x1, y1)
    x2, y2 = _check_valid_coordinates(x2, y2)
    
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    # Select de-zone in the panel
    io_env.mouse_move(1350, 950)
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)

    # Try to de-zone the area
    io_env.mouse_move(x1, y1)
    time.sleep(1)
    io_env.mouse_hold('left')
    time.sleep(1)
    io_env.mouse_move(x2, y2)


@register_skill("confirm_placement")
def confirm_placement():
    """
    The function to confirm the placement and build the object after the try_place_[object] function.
    """
    io_env.mouse_release("left")
    time.sleep(1)
    io_env.mouse_click_button("left", clicks=1)
    time.sleep(1)
    io_env.mouse_click_button("right", clicks=1)
    time.sleep(1)
    io_env.mouse_move(1920, 1000)

@register_skill("cancel_placement")
def cancel_placement():
    """
    The function to cancel the placement of the object after the try_place_[object] function.
    """
    io_env.mouse_click_button("right", clicks=1)


def _check_valid_coordinates(x, y):
    if x < 100: x = 100
    if y < 100: y = 100
    if y > 840: y = 840
    return x, y


__all__ = [
    # Roads
    "open_roads_menu",
    "try_place_two_lane_road",

    # Electricity
    "open_electricity_menu",
    "try_place_wind_turbine",
    "try_place_power_line",

    # Water & Sewage
    "open_water_sewage_menu",
    "try_place_water_pumping_station",
    "try_place_water_drain_pipe",
    "try_place_water_pipe",

    # Zoning
    "open_zoning_menu",
    "try_place_residential_zone",
    "try_place_commercial_zone",
    "try_place_industrial_zone",
    "try_de_zone",

    # General
    "confirm_placement",
    "cancel_placement",
]