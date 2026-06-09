import math
import os
import queue
import random

import carla
import numpy as np


IMAGE_W = 3840
IMAGE_H = 2160
DEFAULT_FOV = 90.0
OUTPUT_DIR = os.environ.get("CARLA_OUTPUT_DIR", "dataset_hard_cctv")


CLASS_ID = 0


CAMERA_MIN_DISTANCE_M = 14.0
CAMERA_DISTANCE_LENGTH_MULT = 3.0
CAMERA_DISTANCE_RANDOM_EXTRA_M = 16.0


CAMERA_MIN_HEIGHT_M = 6.0
CAMERA_MAX_HEIGHT_M = 14.0
CAMERA_MONITORING_VIEW_PROB = 0.05


CAMERA_WALL_CCTV_VIEW_PROB = 0.0

CAMERA_ROOF_CCTV_VIEW_PROB = 0.20

CAMERA_HARD_CCTV_VIEW_PROB = 0.45

CAMERA_TOPDOWN_VIEW_PROB = 0.25
CAMERA_TOPDOWN_MIN_DISTANCE_M = 5.0
CAMERA_TOPDOWN_DISTANCE_LENGTH_MULT_MIN = 0.8
CAMERA_TOPDOWN_DISTANCE_LENGTH_MULT_MAX = 2.2
CAMERA_TOPDOWN_MIN_HEIGHT_M = 16.0
CAMERA_TOPDOWN_MAX_HEIGHT_M = 30.0
CAMERA_TOPDOWN_MIN_PITCH_DEG = -55.0
CAMERA_TOPDOWN_LOOK_XY_JITTER_M = 1.0
CAMERA_TOPDOWN_LOOK_Z_MIN_M = 0.1
CAMERA_TOPDOWN_LOOK_Z_MAX_M = 0.8
CAMERA_TOPDOWN_FOV_MIN = 65.0
CAMERA_TOPDOWN_FOV_MAX = 88.0

CAMERA_WALL_CCTV_MIN_DISTANCE_M = 2.5
CAMERA_WALL_CCTV_DISTANCE_LENGTH_MULT_MIN = 0.55
CAMERA_WALL_CCTV_DISTANCE_LENGTH_MULT_MAX = 1.60
CAMERA_WALL_CCTV_MIN_HEIGHT_M = 7.0
CAMERA_WALL_CCTV_MAX_HEIGHT_M = 16.0
CAMERA_WALL_CCTV_MIN_PITCH_DEG = -38.0
CAMERA_WALL_CCTV_LOOK_XY_JITTER_M = 1.0
CAMERA_WALL_CCTV_LOOK_Z_MIN_M = 0.0
CAMERA_WALL_CCTV_LOOK_Z_MAX_M = 0.9
CAMERA_WALL_CCTV_FOV_MIN = 65.0
CAMERA_WALL_CCTV_FOV_MAX = 95.0
CAMERA_WALL_CCTV_POSE_ATTEMPTS = 25

CAMERA_ROOF_CCTV_MIN_DISTANCE_M = 2.0
CAMERA_ROOF_CCTV_DISTANCE_LENGTH_MULT_MIN = 0.45
CAMERA_ROOF_CCTV_DISTANCE_LENGTH_MULT_MAX = 1.35
CAMERA_ROOF_CCTV_MIN_HEIGHT_M = 10.0
CAMERA_ROOF_CCTV_MAX_HEIGHT_M = 18.0
CAMERA_ROOF_CCTV_MIN_PITCH_DEG = -58.0
CAMERA_ROOF_CCTV_LOOK_XY_JITTER_M = 0.7
CAMERA_ROOF_CCTV_LOOK_Z_MIN_M = 0.0
CAMERA_ROOF_CCTV_LOOK_Z_MAX_M = 0.7
CAMERA_ROOF_CCTV_FOV_MIN = 65.0
CAMERA_ROOF_CCTV_FOV_MAX = 92.0
CAMERA_ROOF_CCTV_POSE_ATTEMPTS = 25

CAMERA_HARD_CCTV_MIN_DISTANCE_M = 3.0
CAMERA_HARD_CCTV_DISTANCE_LENGTH_MULT_MIN = 0.55
CAMERA_HARD_CCTV_DISTANCE_LENGTH_MULT_MAX = 1.75
CAMERA_HARD_CCTV_MIN_HEIGHT_M = 9.0
CAMERA_HARD_CCTV_MAX_HEIGHT_M = 22.0
CAMERA_HARD_CCTV_MIN_PITCH_DEG = -48.0
CAMERA_HARD_CCTV_LOOK_XY_JITTER_M = 2.0
CAMERA_HARD_CCTV_EDGE_AIM_PROB = 0.80
CAMERA_HARD_CCTV_EDGE_AIM_MIN_M = 3.0
CAMERA_HARD_CCTV_EDGE_AIM_MAX_M = 10.0
CAMERA_HARD_CCTV_LOOK_Z_MIN_M = 0.0
CAMERA_HARD_CCTV_LOOK_Z_MAX_M = 0.7
CAMERA_HARD_CCTV_FOV_MIN = 78.0
CAMERA_HARD_CCTV_FOV_MAX = 108.0
CAMERA_HARD_CCTV_ALLOW_OCCLUDED_RAY_PROB = 0.10
CAMERA_HARD_CCTV_POSE_ATTEMPTS = 25

HARD_CCTV_EDGE_MARGIN_RATIO = 0.12
HARD_CCTV_MIN_TRUNCATION_RATIO = 0.03
HARD_CCTV_OCCLUDED_VISIBLE_POINTS_MAX = 4
MIN_HARD_LABELS_PER_IMAGE = 1

UNLOAD_PARKED_VEHICLES = True
UNLOAD_FOLIAGE = os.environ.get("CARLA_UNLOAD_FOLIAGE", "1") != "0"
UNLOAD_STREET_LIGHTS = os.environ.get("CARLA_UNLOAD_STREET_LIGHTS", "0") == "1"

FORCE_LIGHTS_FOR_CURRENT_WEATHER = False

NUM_BACKGROUND_CARS = 30
TARGET_IMAGES = int(os.environ.get("CARLA_TARGET_IMAGES", "2000"))
SPAWN_CLEARANCE_M = 8.0

WALL_NEAR_TARGET_PROB = 0.0
WALL_NEAR_MIN_BUILDING_DISTANCE_M = 0.5
WALL_NEAR_MAX_BUILDING_DISTANCE_M = 14.0
MAX_WALL_NEAR_WAYPOINTS = 1200

ENABLE_RANDOM_VEHICLE_LIGHTS = True
VEHICLE_LIGHTS_PROBABILITY = 0.75


DETECTION_MIN_BOX_WIDTH_PX = 20
DETECTION_MIN_BOX_HEIGHT_PX = 20
DETECTION_MIN_BOX_AREA_PX = 20 * 20
DETECTION_MAX_TRUNCATION_RATIO = 0.70
DETECTION_MAX_LABEL_DISTANCE_M = 85.0
MIN_DETECTION_VISIBLE_POINTS = 3


POSE_MIN_BOX_WIDTH_PX = 180
POSE_MIN_BOX_HEIGHT_PX = 120
POSE_MIN_BOX_AREA_PX = 180 * 120
POSE_MAX_TRUNCATION_RATIO = 0.25
MIN_POSE_VISIBLE_POINTS = 3
MIN_KEYPOINTS_IN_FRAME = 4
MIN_VISIBLE_KEYPOINTS = 1
POSE_MAX_BOX_WIDTH_RATIO = 0.65
POSE_MAX_BOX_HEIGHT_RATIO = 0.75

OBJECT_SELF_HIT_TOLERANCE_M = 1.0
KEYPOINT_RAY_Z_LIFT_M = 0.08
KEYPOINT_SELF_HIT_TOLERANCE_M = 0.5


MIN_LABELS_PER_IMAGE = 1
MIN_POSE_LABELS_PER_IMAGE = 1


SAVE_ONLY_POSE_LABELS = False


LABEL_TARGET_ONLY = False


USE_WHEEL_CONTACT_POINTS = False
WHEEL_CM_TO_M = 0.01

WEATHER_PRESETS = (
    {
        "name": "clear_day",
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 5.0,
        "wind_intensity": 8.0,
        "sun_altitude_angle": 65.0,
        "scattering_intensity": 0.2,
        "mie_scattering_scale": 0.03,
    },
    {
        "name": "bright_cloudy",
        "cloudiness": 45.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 10.0,
        "wind_intensity": 12.0,
        "sun_altitude_angle": 52.0,
        "scattering_intensity": 0.35,
        "mie_scattering_scale": 0.08,
    },
    {
        "name": "overcast",
        "cloudiness": 82.0,
        "precipitation": 0.0,
        "precipitation_deposits": 8.0,
        "wetness": 18.0,
        "wind_intensity": 18.0,
        "sun_altitude_angle": 42.0,
        "scattering_intensity": 0.55,
        "mie_scattering_scale": 0.12,
    },
    {
        "name": "light_rain",
        "cloudiness": 92.0,
        "precipitation": 28.0,
        "precipitation_deposits": 35.0,
        "wetness": 55.0,
        "wind_intensity": 24.0,
        "sun_altitude_angle": 34.0,
        "scattering_intensity": 0.65,
        "mie_scattering_scale": 0.18,
    },
    {
        "name": "heavy_rain",
        "cloudiness": 100.0,
        "precipitation": 68.0,
        "precipitation_deposits": 72.0,
        "wetness": 92.0,
        "wind_intensity": 34.0,
        "sun_altitude_angle": 26.0,
        "scattering_intensity": 0.8,
        "mie_scattering_scale": 0.28,
    },
    {
        "name": "late_afternoon",
        "cloudiness": 30.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 8.0,
        "wind_intensity": 10.0,
        "sun_altitude_angle": 20.0,
        "scattering_intensity": 0.3,
        "mie_scattering_scale": 0.07,
    },
        {
        "name": "dark_evening_lights",
        "cloudiness": 65.0,
        "precipitation": 0.0,
        "precipitation_deposits": 10.0,
        "wetness": 25.0,
        "wind_intensity": 10.0,
        "sun_altitude_angle": 3.0,
        "scattering_intensity": 0.75,
        "mie_scattering_scale": 0.25,
        "force_vehicle_lights": True,
    },
        {
        "name": "very_dark_evening_lights",
        "cloudiness": 88.0,
        "precipitation": 0.0,
        "precipitation_deposits": 15.0,
        "wetness": 35.0,
        "wind_intensity": 12.0,
        "sun_altitude_angle": -4.0,
        "scattering_intensity": 0.9,
        "mie_scattering_scale": 0.35,
        "force_vehicle_lights": True,
    },
)


def get_camera_matrix(fov):
    focal = IMAGE_W / (2.0 * np.tan(fov * np.pi / 360.0))
    matrix = np.identity(3)
    matrix[0, 0] = matrix[1, 1] = focal
    matrix[0, 2] = IMAGE_W / 2.0
    matrix[1, 2] = IMAGE_H / 2.0
    return matrix


def next_image_index(output_dir):
    counter = 1
    if not os.path.exists(output_dir):
        return counter

    existing = [
        name for name in os.listdir(output_dir)
        if name.startswith("auto_") and name.endswith(".png")
    ]
    if not existing:
        return counter

    indices = [int(name.replace("auto_", "").replace(".png", "")) for name in existing]
    return max(indices) + 1


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def projected_box_edge_margin_ratio(projected_box):
    return min(
        projected_box["min_x"] / IMAGE_W,
        projected_box["min_y"] / IMAGE_H,
        (IMAGE_W - projected_box["max_x"]) / IMAGE_W,
        (IMAGE_H - projected_box["max_y"]) / IMAGE_H,
    )


def is_hard_cctv_label(projected_box, visible_object_points):
    return (
        projected_box["truncation_ratio"] >= HARD_CCTV_MIN_TRUNCATION_RATIO
        or projected_box_edge_margin_ratio(projected_box) <= HARD_CCTV_EDGE_MARGIN_RATIO
        or visible_object_points <= HARD_CCTV_OCCLUDED_VISIBLE_POINTS_MAX
    )


def sample_weather():
    preset = random.choice(WEATHER_PRESETS)

    preset_name = preset["name"]
    is_dark_scene = preset.get("force_vehicle_lights", False)

    if preset_name == "very_dark_evening_lights":
        sun_altitude_min = -10.0
        sun_altitude_max = 1.0
    elif is_dark_scene:
        sun_altitude_min = -5.0
        sun_altitude_max = 10.0
    else:
        sun_altitude_min = 15.0
        sun_altitude_max = 75.0

    weather = carla.WeatherParameters(
        cloudiness=clamp(preset["cloudiness"] + random.uniform(-8.0, 8.0), 0.0, 100.0),
        precipitation=clamp(
            preset["precipitation"] + random.uniform(-10.0, 10.0), 0.0, 100.0
        ),
        sun_altitude_angle=clamp(
            preset["sun_altitude_angle"] + random.uniform(-3.0, 3.0),
            sun_altitude_min,
            sun_altitude_max,
        ),
        sun_azimuth_angle=random.uniform(0.0, 360.0),
        precipitation_deposits=clamp(
            preset["precipitation_deposits"] + random.uniform(-10.0, 10.0),
            0.0,
            100.0,
        ),
        wind_intensity=clamp(
            preset["wind_intensity"] + random.uniform(-8.0, 8.0), 0.0, 100.0
        ),
        scattering_intensity=clamp(
            preset["scattering_intensity"] + random.uniform(-0.08, 0.08), 0.0, 1.0
        ),
        mie_scattering_scale=clamp(
            preset["mie_scattering_scale"] + random.uniform(-0.04, 0.04), 0.0, 1.0
        ),
        wetness=clamp(preset["wetness"] + random.uniform(-10.0, 10.0), 0.0, 100.0),
    )

    return preset["name"], weather


def ray_hits_target(world, camera_location, target_location, tolerance_m):
    ray_trace = world.cast_ray(camera_location, target_location)
    if len(ray_trace) == 0:
        return True
    return target_location.distance(ray_trace[0].location) <= tolerance_m


def project_world_point(world_location, world_to_camera, camera_matrix):
    point_3d = np.array([world_location.x, world_location.y, world_location.z, 1.0])
    point_camera = np.dot(world_to_camera, point_3d)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]], dtype=float)

    if point_camera[2] <= 0.1:
        return None

    point_2d = np.dot(camera_matrix, point_camera)
    point_2d /= point_2d[2]
    return point_2d[:2]


def build_local_bbox_vertices(bbox):
    ext = bbox.extent
    loc = bbox.location
    return [
        carla.Location(x=loc.x + ext.x, y=loc.y + ext.y, z=loc.z + ext.z),
        carla.Location(x=loc.x + ext.x, y=loc.y - ext.y, z=loc.z + ext.z),
        carla.Location(x=loc.x - ext.x, y=loc.y + ext.y, z=loc.z + ext.z),
        carla.Location(x=loc.x - ext.x, y=loc.y - ext.y, z=loc.z + ext.z),
        carla.Location(x=loc.x + ext.x, y=loc.y + ext.y, z=loc.z - ext.z),
        carla.Location(x=loc.x + ext.x, y=loc.y - ext.y, z=loc.z - ext.z),
        carla.Location(x=loc.x - ext.x, y=loc.y + ext.y, z=loc.z - ext.z),
        carla.Location(x=loc.x - ext.x, y=loc.y - ext.y, z=loc.z - ext.z),
    ]


def project_vehicle_box(vehicle, world_to_camera, camera_matrix):
    vehicle_transform = vehicle.get_transform()
    projected_points = []

    for local_vertex in build_local_bbox_vertices(vehicle.bounding_box):
        world_vertex = carla.Location(local_vertex.x, local_vertex.y, local_vertex.z)
        vehicle_transform.transform(world_vertex)
        point_2d = project_world_point(world_vertex, world_to_camera, camera_matrix)
        if point_2d is not None:
            projected_points.append(point_2d)

    if not projected_points:
        return None

    projected = np.array(projected_points)
    raw_min_x = float(np.min(projected[:, 0]))
    raw_max_x = float(np.max(projected[:, 0]))
    raw_min_y = float(np.min(projected[:, 1]))
    raw_max_y = float(np.max(projected[:, 1]))

    if raw_max_x < 0 or raw_min_x > IMAGE_W or raw_max_y < 0 or raw_min_y > IMAGE_H:
        return None

    raw_width = raw_max_x - raw_min_x
    raw_height = raw_max_y - raw_min_y
    if raw_width <= 0 or raw_height <= 0:
        return None

    min_x = max(0.0, raw_min_x)
    max_x = min(float(IMAGE_W), raw_max_x)
    min_y = max(0.0, raw_min_y)
    max_y = min(float(IMAGE_H), raw_max_y)

    clipped_width = max_x - min_x
    clipped_height = max_y - min_y
    clipped_area = clipped_width * clipped_height
    raw_area = raw_width * raw_height

    if clipped_width <= 0 or clipped_height <= 0 or raw_area <= 0:
        return None

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": clipped_width,
        "height": clipped_height,
        "area": clipped_area,
        "truncation_ratio": 1.0 - (clipped_area / raw_area),
    }


def get_object_visibility_checkpoints(vehicle):
    bbox = vehicle.bounding_box
    ext = bbox.extent
    loc = bbox.location
    local_points = [
        carla.Location(x=loc.x, y=loc.y, z=loc.z + ext.z),
        carla.Location(x=loc.x + ext.x, y=loc.y, z=loc.z),
        carla.Location(x=loc.x - ext.x, y=loc.y, z=loc.z),
        carla.Location(x=loc.x, y=loc.y + ext.y, z=loc.z),
        carla.Location(x=loc.x, y=loc.y - ext.y, z=loc.z),
    ]

    vehicle_transform = vehicle.get_transform()
    world_points = []
    for local_point in local_points:
        world_point = carla.Location(local_point.x, local_point.y, local_point.z)
        vehicle_transform.transform(world_point)
        world_points.append(world_point)
    return world_points


def get_wheel_contact_local_points(vehicle):
    if not USE_WHEEL_CONTACT_POINTS:
        return None

    try:
        wheels = list(vehicle.get_physics_control().wheels)
    except RuntimeError:
        return None

    if len(wheels) < 4:
        return None

    wheel_points = []
    for wheel in wheels[:4]:
        wheel_points.append(
            carla.Location(
                x=wheel.position.x * WHEEL_CM_TO_M,
                y=wheel.position.y * WHEEL_CM_TO_M,
                z=(wheel.position.z - wheel.radius) * WHEEL_CM_TO_M,
            )
        )

    wheel_points.sort(key=lambda point: point.x, reverse=True)
    front_pair = sorted(wheel_points[:2], key=lambda point: point.y)
    rear_pair = sorted(wheel_points[2:4], key=lambda point: point.y)


    return [front_pair[0], front_pair[1], rear_pair[0], rear_pair[1]]


def get_bbox_contact_local_points(vehicle):
    bbox = vehicle.bounding_box
    ext = bbox.extent
    loc = bbox.location
    z_bottom = loc.z - ext.z


    return [
        carla.Location(x=loc.x + ext.x, y=loc.y - ext.y, z=z_bottom),
        carla.Location(x=loc.x + ext.x, y=loc.y + ext.y, z=z_bottom),
        carla.Location(x=loc.x - ext.x, y=loc.y + ext.y, z=z_bottom),
        carla.Location(x=loc.x - ext.x, y=loc.y - ext.y, z=z_bottom),
    ]


def get_vehicle_keypoint_world_locations(vehicle):
    local_points = get_wheel_contact_local_points(vehicle)
    if local_points is None:
        local_points = get_bbox_contact_local_points(vehicle)

    vehicle_transform = vehicle.get_transform()
    world_points = []
    for local_point in local_points:
        world_point = carla.Location(local_point.x, local_point.y, local_point.z)
        vehicle_transform.transform(world_point)
        world_points.append(world_point)
    return world_points


def project_keypoint(world, camera_location, world_location, world_to_camera, camera_matrix):

    point_2d = project_world_point(world_location, world_to_camera, camera_matrix)
    if point_2d is None:
        return 0.0, 0.0, "0"

    x_norm = float(point_2d[0]) / IMAGE_W
    y_norm = float(point_2d[1]) / IMAGE_H


    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return 0.0, 0.0, "0"


    lifted_world_location = carla.Location(
        x=world_location.x,
        y=world_location.y,
        z=world_location.z + KEYPOINT_RAY_Z_LIFT_M,
    )

    visible = ray_hits_target(
        world,
        camera_location,
        lifted_world_location,
        KEYPOINT_SELF_HIT_TOLERANCE_M,
    )


    visibility = "2" if visible else "1"

    return x_norm, y_norm, visibility


def empty_keypoint_values(count=4):
    return [(0.0, 0.0, "0") for _ in range(count)]

def set_random_vehicle_lights(vehicle):
    if not ENABLE_RANDOM_VEHICLE_LIGHTS:
        vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
        return

    force_lights = FORCE_LIGHTS_FOR_CURRENT_WEATHER

    if not force_lights and random.random() > VEHICLE_LIGHTS_PROBABILITY:
        vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
        return

    light_state = int(carla.VehicleLightState.Position) | int(carla.VehicleLightState.LowBeam)

    if force_lights:


        if random.random() < 0.25:
            light_state |= int(carla.VehicleLightState.Brake)
    else:
        if random.random() < 0.20:
            light_state |= int(carla.VehicleLightState.Brake)

    blinker_random = random.random()
    if blinker_random < 0.08:
        light_state |= int(carla.VehicleLightState.LeftBlinker)
    elif blinker_random < 0.16:
        light_state |= int(carla.VehicleLightState.RightBlinker)

    if force_lights and random.random() < 0.10:
        light_state |= int(carla.VehicleLightState.HighBeam)
    elif random.random() < 0.05:
        light_state |= int(carla.VehicleLightState.HighBeam)

    vehicle.set_light_state(carla.VehicleLightState(light_state))


def xy_distance_to_building_bbox(location, building_bbox):
    dx = max(0.0, abs(location.x - building_bbox.location.x) - building_bbox.extent.x)
    dy = max(0.0, abs(location.y - building_bbox.location.y) - building_bbox.extent.y)
    return math.sqrt(dx * dx + dy * dy)


def collect_wall_near_waypoints(world, waypoints):
    try:
        building_bboxes = list(world.get_level_bbs(carla.CityObjectLabel.Buildings))
    except RuntimeError:
        return []

    if not building_bboxes:
        return []

    candidates = []
    shuffled_waypoints = list(waypoints)
    random.shuffle(shuffled_waypoints)

    for waypoint in shuffled_waypoints:
        location = waypoint.transform.location
        nearest_distance = min(
            xy_distance_to_building_bbox(location, building_bbox)
            for building_bbox in building_bboxes
        )

        if (
            WALL_NEAR_MIN_BUILDING_DISTANCE_M
            <= nearest_distance
            <= WALL_NEAR_MAX_BUILDING_DISTANCE_M
        ):
            candidates.append(waypoint)

        if len(candidates) >= MAX_WALL_NEAR_WAYPOINTS:
            break

    return candidates


def spawn_random_car(world, blueprints, waypoints):
    new_vehicle = None
    while new_vehicle is None:
        blueprint = random.choice(blueprints)


        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)

        start_waypoint = random.choice(waypoints)

        too_close = False
        for existing_vehicle in world.get_actors().filter("vehicle.*"):
            distance = existing_vehicle.get_transform().location.distance(
                start_waypoint.transform.location
            )
            if distance < SPAWN_CLEARANCE_M:
                too_close = True
                break

        if too_close:
            continue

        spawn_transform = start_waypoint.transform
        spawn_transform.location.z += 1.0
        new_vehicle = world.try_spawn_actor(blueprint, spawn_transform)

    new_vehicle.set_autopilot(True)
    set_random_vehicle_lights(new_vehicle)
    return new_vehicle


def spawn_target_car(world, blueprints, waypoints, wall_near_waypoints):
    use_wall_near = bool(wall_near_waypoints) and random.random() < WALL_NEAR_TARGET_PROB
    target_waypoints = wall_near_waypoints if use_wall_near else waypoints
    return spawn_random_car(world, blueprints, target_waypoints), use_wall_near


def destroy_actors(actors):
    for actor in actors:
        if actor is None:
            continue
        try:
            actor.destroy()
        except RuntimeError:
            pass


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        world.get_map().generate_waypoints(5.0)
    except RuntimeError:
        print("Broken map detected. Loading Town01.")
        client.load_world("Town01")
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    destroy_actors(world.get_actors().filter("vehicle.*"))

    try:
        if UNLOAD_PARKED_VEHICLES:
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        else:
            world.load_map_layer(carla.MapLayer.ParkedVehicles)

        if UNLOAD_FOLIAGE:
            world.unload_map_layer(carla.MapLayer.Foliage)
        else:
            world.load_map_layer(carla.MapLayer.Foliage)

        if UNLOAD_STREET_LIGHTS:
            world.unload_map_layer(carla.MapLayer.StreetLights)
        else:
            world.load_map_layer(carla.MapLayer.StreetLights)

        print(
            "Map layers:",
            f"parked_unloaded={UNLOAD_PARKED_VEHICLES}",
            f"foliage_unloaded={UNLOAD_FOLIAGE}",
            f"street_lights_unloaded={UNLOAD_STREET_LIGHTS}",
        )
    except RuntimeError:
        pass

    blueprint_library = world.get_blueprint_library()
    car_blueprints = [
        blueprint
        for blueprint in blueprint_library.filter("vehicle.*")
        if int(blueprint.get_attribute("number_of_wheels").as_int()) == 4
    ]
    waypoints = world.get_map().generate_waypoints(5.0)
    wall_near_waypoints = collect_wall_near_waypoints(world, waypoints)
    print(f"Wall-near waypoint candidates: {len(wall_near_waypoints)}")

    print(f"Spawning {NUM_BACKGROUND_CARS} background cars...")
    background_vehicles = [
        spawn_random_car(world, car_blueprints, waypoints)
        for _ in range(NUM_BACKGROUND_CARS)
    ]

    target_vehicle, target_wall_near = spawn_target_car(
        world,
        car_blueprints,
        waypoints,
        wall_near_waypoints,
    )
    start_transform = target_vehicle.get_transform()
    print(f"Target vehicle: {target_vehicle.type_id}, wall_near={target_wall_near}")

    spectator = world.get_spectator()
    spectator_location = start_transform.location + carla.Location(z=8.0, x=-8.0)
    spectator.set_transform(
        carla.Transform(
            spectator_location,
            carla.Rotation(pitch=-30.0, yaw=start_transform.rotation.yaw),
        )
    )

    camera_blueprint = blueprint_library.find("sensor.camera.rgb")
    camera_blueprint.set_attribute("image_size_x", str(IMAGE_W))
    camera_blueprint.set_attribute("image_size_y", str(IMAGE_H))
    camera_blueprint.set_attribute("fov", str(DEFAULT_FOV))
    camera_blueprint.set_attribute("motion_blur_intensity", "0")
    camera_blueprint.set_attribute("motion_blur_max_distortion", "0")

    counter = next_image_index(OUTPUT_DIR)
    target_limit = counter + TARGET_IMAGES - 1

    print("\n" + "=" * 50)
    print(
        f"AUTO DATASET GENERATION ({TARGET_IMAGES} new images, starting from {counter})"
    )
    print("=" * 50 + "\n")

    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    for _ in range(10):
        world.tick()

    camera = None
    try:
        while counter <= target_limit:
            if (counter - 1) % 50 == 0:
                weather_name, weather = sample_weather()
                world.set_weather(weather)

                global FORCE_LIGHTS_FOR_CURRENT_WEATHER
                FORCE_LIGHTS_FOR_CURRENT_WEATHER = weather_name in {
                    "dark_evening_lights",
                    "very_dark_evening_lights",
                }

                for vehicle in world.get_actors().filter("vehicle.*"):
                    set_random_vehicle_lights(vehicle)

                print(f"Weather preset changed to: {weather_name}, force_lights={FORCE_LIGHTS_FOR_CURRENT_WEATHER}")

            if (counter - 1) % 10 == 0 and counter > 1:
                destroy_actors([target_vehicle])

                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)

                target_vehicle, target_wall_near = spawn_target_car(
                    world,
                    car_blueprints,
                    waypoints,
                    wall_near_waypoints,
                )
                print(f"Target vehicle changed to: {target_vehicle.type_id}, wall_near={target_wall_near}")

                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)

                for _ in range(10):
                    world.tick()

            for _ in range(random.randint(5, 15)):
                world.tick()

            target_transform = target_vehicle.get_transform()
            target_bbox = target_vehicle.bounding_box
            target_length = target_bbox.extent.x * 2.0

            camera_mode_rand = random.random()

            if camera_mode_rand < CAMERA_WALL_CCTV_VIEW_PROB:
                camera_mode = "wall_cctv"
                min_distance = max(
                    CAMERA_WALL_CCTV_MIN_DISTANCE_M,
                    target_length * CAMERA_WALL_CCTV_DISTANCE_LENGTH_MULT_MIN,
                )
                max_distance = max(
                    min_distance + 2.0,
                    target_length * CAMERA_WALL_CCTV_DISTANCE_LENGTH_MULT_MAX,
                )
                sampled_camera_height = random.uniform(
                    CAMERA_WALL_CCTV_MIN_HEIGHT_M,
                    CAMERA_WALL_CCTV_MAX_HEIGHT_M,
                )
                look_xy_jitter = CAMERA_WALL_CCTV_LOOK_XY_JITTER_M
                look_z_min = CAMERA_WALL_CCTV_LOOK_Z_MIN_M
                look_z_max = CAMERA_WALL_CCTV_LOOK_Z_MAX_M
                fov_min = CAMERA_WALL_CCTV_FOV_MIN
                fov_max = CAMERA_WALL_CCTV_FOV_MAX
            elif camera_mode_rand < CAMERA_WALL_CCTV_VIEW_PROB + CAMERA_ROOF_CCTV_VIEW_PROB:
                camera_mode = "roof_cctv"
                min_distance = max(
                    CAMERA_ROOF_CCTV_MIN_DISTANCE_M,
                    target_length * CAMERA_ROOF_CCTV_DISTANCE_LENGTH_MULT_MIN,
                )
                max_distance = max(
                    min_distance + 2.0,
                    target_length * CAMERA_ROOF_CCTV_DISTANCE_LENGTH_MULT_MAX,
                )
                sampled_camera_height = random.uniform(
                    CAMERA_ROOF_CCTV_MIN_HEIGHT_M,
                    CAMERA_ROOF_CCTV_MAX_HEIGHT_M,
                )
                look_xy_jitter = CAMERA_ROOF_CCTV_LOOK_XY_JITTER_M
                look_z_min = CAMERA_ROOF_CCTV_LOOK_Z_MIN_M
                look_z_max = CAMERA_ROOF_CCTV_LOOK_Z_MAX_M
                fov_min = CAMERA_ROOF_CCTV_FOV_MIN
                fov_max = CAMERA_ROOF_CCTV_FOV_MAX
            elif camera_mode_rand < (
                CAMERA_WALL_CCTV_VIEW_PROB
                + CAMERA_ROOF_CCTV_VIEW_PROB
                + CAMERA_HARD_CCTV_VIEW_PROB
            ):
                camera_mode = "hard_cctv"
                min_distance = max(
                    CAMERA_HARD_CCTV_MIN_DISTANCE_M,
                    target_length * CAMERA_HARD_CCTV_DISTANCE_LENGTH_MULT_MIN,
                )
                max_distance = max(
                    min_distance + 2.0,
                    target_length * CAMERA_HARD_CCTV_DISTANCE_LENGTH_MULT_MAX,
                )
                sampled_camera_height = random.uniform(
                    CAMERA_HARD_CCTV_MIN_HEIGHT_M,
                    CAMERA_HARD_CCTV_MAX_HEIGHT_M,
                )
                look_xy_jitter = CAMERA_HARD_CCTV_LOOK_XY_JITTER_M
                look_z_min = CAMERA_HARD_CCTV_LOOK_Z_MIN_M
                look_z_max = CAMERA_HARD_CCTV_LOOK_Z_MAX_M
                fov_min = CAMERA_HARD_CCTV_FOV_MIN
                fov_max = CAMERA_HARD_CCTV_FOV_MAX
            elif camera_mode_rand < (
                CAMERA_WALL_CCTV_VIEW_PROB
                +
                CAMERA_ROOF_CCTV_VIEW_PROB
                + CAMERA_HARD_CCTV_VIEW_PROB
                + CAMERA_TOPDOWN_VIEW_PROB
            ):
                camera_mode = "topdown"
                min_distance = max(
                    CAMERA_TOPDOWN_MIN_DISTANCE_M,
                    target_length * CAMERA_TOPDOWN_DISTANCE_LENGTH_MULT_MIN,
                )
                max_distance = max(
                    min_distance + 2.0,
                    target_length * CAMERA_TOPDOWN_DISTANCE_LENGTH_MULT_MAX,
                )
                sampled_camera_height = random.uniform(
                    CAMERA_TOPDOWN_MIN_HEIGHT_M,
                    CAMERA_TOPDOWN_MAX_HEIGHT_M,
                )
                look_xy_jitter = CAMERA_TOPDOWN_LOOK_XY_JITTER_M
                look_z_min = CAMERA_TOPDOWN_LOOK_Z_MIN_M
                look_z_max = CAMERA_TOPDOWN_LOOK_Z_MAX_M
                fov_min = CAMERA_TOPDOWN_FOV_MIN
                fov_max = CAMERA_TOPDOWN_FOV_MAX
            elif camera_mode_rand < (
                CAMERA_WALL_CCTV_VIEW_PROB
                +
                CAMERA_ROOF_CCTV_VIEW_PROB
                +
                CAMERA_HARD_CCTV_VIEW_PROB
                + CAMERA_TOPDOWN_VIEW_PROB
                + CAMERA_MONITORING_VIEW_PROB
            ):
                camera_mode = "monitoring"
                min_distance = max(18.0, target_length * 4.0)
                max_distance = min_distance + 22.0
                sampled_camera_height = random.uniform(8.0, 16.0)
                look_xy_jitter = 4.0
                look_z_min = 0.5
                look_z_max = 1.5
                fov_min = 75.0
                fov_max = 100.0
            else:
                camera_mode = "standard"
                min_distance = max(CAMERA_MIN_DISTANCE_M, target_length * CAMERA_DISTANCE_LENGTH_MULT)
                max_distance = min_distance + CAMERA_DISTANCE_RANDOM_EXTRA_M
                sampled_camera_height = random.uniform(CAMERA_MIN_HEIGHT_M, CAMERA_MAX_HEIGHT_M)
                look_xy_jitter = 4.0
                look_z_min = 0.5
                look_z_max = 1.5
                fov_min = 75.0
                fov_max = 100.0

            if camera_mode == "wall_cctv" and not target_wall_near:
                print("Skipping wall_cctv frame because target is not near a building.")
                continue

            valid_camera_found = False
            camera_transform = None
            allow_occluded_camera_ray = (
                camera_mode == "hard_cctv"
                and random.random() < CAMERA_HARD_CCTV_ALLOW_OCCLUDED_RAY_PROB
            )
            camera_pose_attempts = (
                CAMERA_HARD_CCTV_POSE_ATTEMPTS
                if camera_mode == "hard_cctv"
                else CAMERA_WALL_CCTV_POSE_ATTEMPTS
                if camera_mode == "wall_cctv"
                else CAMERA_ROOF_CCTV_POSE_ATTEMPTS
                if camera_mode == "roof_cctv"
                else 10
            )

            for _ in range(camera_pose_attempts):
                if camera_mode == "wall_cctv":
                    vehicle_yaw_rad = math.radians(target_transform.rotation.yaw)
                    side = random.choice([-1.0, 1.0])
                    angle = vehicle_yaw_rad + side * (math.pi / 2.0) + random.uniform(-0.45, 0.45)
                else:
                    angle = random.uniform(0.0, 2.0 * math.pi)
                distance = random.uniform(min_distance, max_distance)
                height = sampled_camera_height

                raw_x = target_transform.location.x + distance * math.cos(angle)
                raw_y = target_transform.location.y + distance * math.sin(angle)

                if camera_mode in {"roof_cctv", "wall_cctv"}:
                    safe_waypoint = None
                else:
                    safe_waypoint = world.get_map().get_waypoint(
                        carla.Location(
                            x=raw_x,
                            y=raw_y,
                            z=target_transform.location.z,
                        ),
                        project_to_road=True,
                    )

                if safe_waypoint is not None:
                    camera_location = safe_waypoint.transform.location
                    camera_location.z += height
                else:
                    camera_location = carla.Location(
                        x=raw_x,
                        y=raw_y,
                        z=target_transform.location.z + height,
                    )

                target_ray_location = carla.Location(
                    x=target_transform.location.x,
                    y=target_transform.location.y,
                    z=target_transform.location.z + 1.0,
                )
                ray_trace = world.cast_ray(camera_location, target_ray_location)
                if len(ray_trace) > 0:
                    hit_label = ray_trace[0].label
                    valid_vehicle_labels = [
                        carla.CityObjectLabel.Car,
                        carla.CityObjectLabel.Truck,
                        carla.CityObjectLabel.Bus,
                        carla.CityObjectLabel.Dynamic,
                        carla.CityObjectLabel.Any,
                    ]
                    if hit_label not in valid_vehicle_labels and not allow_occluded_camera_ray:
                        continue

                if camera_mode == "hard_cctv" and random.random() < CAMERA_HARD_CCTV_EDGE_AIM_PROB:
                    edge_angle = random.uniform(0.0, 2.0 * math.pi)
                    edge_offset = random.uniform(
                        CAMERA_HARD_CCTV_EDGE_AIM_MIN_M,
                        CAMERA_HARD_CCTV_EDGE_AIM_MAX_M,
                    )
                    look_offset_x = edge_offset * math.cos(edge_angle)
                    look_offset_y = edge_offset * math.sin(edge_angle)
                else:
                    look_offset_x = random.uniform(-look_xy_jitter, look_xy_jitter)
                    look_offset_y = random.uniform(-look_xy_jitter, look_xy_jitter)

                target_look_location = carla.Location(
                    x=target_transform.location.x + look_offset_x,
                    y=target_transform.location.y + look_offset_y,
                    z=target_transform.location.z + random.uniform(look_z_min, look_z_max),
                )

                direction_x = target_look_location.x - camera_location.x
                direction_y = target_look_location.y - camera_location.y
                direction_z = target_look_location.z - camera_location.z
                distance_2d = math.sqrt(direction_x ** 2 + direction_y ** 2)

                pitch = math.degrees(math.atan2(direction_z, distance_2d))
                yaw = math.degrees(math.atan2(direction_y, direction_x))


                if camera_mode == "topdown" and pitch > CAMERA_TOPDOWN_MIN_PITCH_DEG:
                    continue
                if camera_mode == "wall_cctv" and pitch > CAMERA_WALL_CCTV_MIN_PITCH_DEG:
                    continue
                if camera_mode == "roof_cctv" and pitch > CAMERA_ROOF_CCTV_MIN_PITCH_DEG:
                    continue
                if camera_mode == "hard_cctv" and pitch > CAMERA_HARD_CCTV_MIN_PITCH_DEG:
                    continue

                camera_transform = carla.Transform(
                    camera_location,
                    carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
                )
                valid_camera_found = True
                break

            if not valid_camera_found:
                print("Skipping frame because no clean camera pose was found.")
                continue

            random_fov = random.uniform(fov_min, fov_max)
            camera_blueprint.set_attribute("fov", str(random_fov))
            camera_matrix = get_camera_matrix(random_fov)
            spectator.set_transform(camera_transform)

            camera = world.spawn_actor(camera_blueprint, camera_transform)
            image_queue = queue.Queue()
            camera.listen(image_queue.put)

            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
                world_to_camera = np.array(camera.get_transform().get_inverse_matrix())
                camera_location = camera.get_transform().location

                candidate_vehicles = (
                    [target_vehicle]
                    if LABEL_TARGET_ONLY
                    else list(world.get_actors().filter("vehicle.*"))
                )

                label_lines = []
                pose_label_count = 0
                hard_label_count = 0
                for vehicle in candidate_vehicles:
                    vehicle_location = vehicle.get_transform().location
                    if vehicle_location.distance(camera_location) > DETECTION_MAX_LABEL_DISTANCE_M:
                        continue

                    projected_box = project_vehicle_box(
                        vehicle,
                        world_to_camera,
                        camera_matrix,
                    )
                    if projected_box is None:
                        continue

                    if projected_box["width"] < DETECTION_MIN_BOX_WIDTH_PX:
                        continue
                    if projected_box["height"] < DETECTION_MIN_BOX_HEIGHT_PX:
                        continue
                    if projected_box["area"] < DETECTION_MIN_BOX_AREA_PX:
                        continue
                    if projected_box["truncation_ratio"] > DETECTION_MAX_TRUNCATION_RATIO:
                        continue

                    visible_object_points = 0
                    for world_point in get_object_visibility_checkpoints(vehicle):
                        if ray_hits_target(
                            world,
                            camera_location,
                            world_point,
                            OBJECT_SELF_HIT_TOLERANCE_M,
                        ):
                            visible_object_points += 1

                    if visible_object_points < MIN_DETECTION_VISIBLE_POINTS:
                        continue


                    keypoint_values = empty_keypoint_values()
                    has_pose_points = False

                    pose_worthy = (
                        projected_box["width"] >= POSE_MIN_BOX_WIDTH_PX
                        and projected_box["height"] >= POSE_MIN_BOX_HEIGHT_PX
                        and projected_box["area"] >= POSE_MIN_BOX_AREA_PX
                        and projected_box["width"] <= IMAGE_W * POSE_MAX_BOX_WIDTH_RATIO
                        and projected_box["height"] <= IMAGE_H * POSE_MAX_BOX_HEIGHT_RATIO
                        and projected_box["truncation_ratio"] <= POSE_MAX_TRUNCATION_RATIO
                        and visible_object_points >= MIN_POSE_VISIBLE_POINTS
                    )

                    if pose_worthy:
                        candidate_keypoint_values = []
                        visible_keypoints = 0
                        keypoints_in_frame = 0

                        for world_keypoint in get_vehicle_keypoint_world_locations(vehicle):
                            x_norm, y_norm, visibility = project_keypoint(
                                world,
                                camera_location,
                                world_keypoint,
                                world_to_camera,
                                camera_matrix,
                            )

                            if visibility != "0":
                                keypoints_in_frame += 1

                            if visibility == "2":
                                visible_keypoints += 1

                            candidate_keypoint_values.append((x_norm, y_norm, visibility))

                        if (
                            keypoints_in_frame >= MIN_KEYPOINTS_IN_FRAME
                            and visible_keypoints >= MIN_VISIBLE_KEYPOINTS
                        ):
                            keypoint_values = candidate_keypoint_values
                            has_pose_points = True


                    if SAVE_ONLY_POSE_LABELS and not has_pose_points:
                        continue

                    x_center = (
                        (projected_box["min_x"] + projected_box["max_x"]) / 2.0
                    ) / IMAGE_W
                    y_center = (
                        (projected_box["min_y"] + projected_box["max_y"]) / 2.0
                    ) / IMAGE_H
                    box_w = projected_box["width"] / IMAGE_W
                    box_h = projected_box["height"] / IMAGE_H

                    line = (
                        f"{CLASS_ID} "
                        f"{x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                    )
                    for x_norm, y_norm, visibility in keypoint_values:
                        line += f" {x_norm:.6f} {y_norm:.6f} {visibility}"
                    label_lines.append(line)

                    if has_pose_points:
                        pose_label_count += 1

                    if camera_mode == "hard_cctv" and is_hard_cctv_label(
                        projected_box,
                        visible_object_points,
                    ):
                        hard_label_count += 1

                if not label_lines:
                    print("Skipping frame because no labels passed the detection filters.")
                    continue

                if len(label_lines) < MIN_LABELS_PER_IMAGE:
                    print(
                        f"Skipping frame because only {len(label_lines)} vehicles were labeled "
                        f"but {MIN_LABELS_PER_IMAGE} are required."
                        )
                    continue

                if pose_label_count < MIN_POSE_LABELS_PER_IMAGE:
                    print(
                        f"Skipping frame because only {pose_label_count} vehicles received pose keypoints "
                        f"but {MIN_POSE_LABELS_PER_IMAGE} are required."
                    )
                    continue

                if camera_mode == "hard_cctv" and hard_label_count < MIN_HARD_LABELS_PER_IMAGE:
                    print(
                        f"Skipping hard_cctv frame because only {hard_label_count} hard labels "
                        f"were found but {MIN_HARD_LABELS_PER_IMAGE} are required."
                    )
                    continue

                image_name = f"auto_{counter:03d}"
                image_path = os.path.join(OUTPUT_DIR, f"{image_name}.png")
                label_path = os.path.join(OUTPUT_DIR, f"{image_name}.txt")

                image.save_to_disk(image_path)
                with open(label_path, "w", encoding="utf-8") as label_file:
                    label_file.write("\n".join(label_lines) + "\n")

                print(f"Saved image {counter}/{target_limit} [{camera_mode}] with {len(label_lines)} labels.")
                counter += 1

            except queue.Empty:
                print("Image retrieval failed, retrying current frame.")
            finally:
                camera.stop()
                destroy_actors([camera])
                camera = None

    finally:
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
            destroy_actors([camera])

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        destroy_actors([target_vehicle])
        destroy_actors(background_vehicles)
        print("Finished. Dataset saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
