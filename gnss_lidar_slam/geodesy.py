"""Small WGS84 geodetic-to-local-map transform with no implicit alignment."""
import math
import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

def enu_to_map_rotation(config):
    """Compose ENU yaw with optional fixed KISS-map tilt.

    ``lidar_roll_deg`` and ``lidar_pitch_deg`` describe the KISS-map -> ENU
    leveling rotation.  GNSS positions require the inverse direction.
    """
    yaw = float(config["yaw_alignment_rad"])
    yaw_rotation = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    if not config.get("set_lidar_roll_pitch", False):
        return yaw_rotation
    roll = math.radians(-float(config["lidar_roll_deg"]))
    pitch = math.radians(-float(config["lidar_pitch_deg"]))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rotation_x = np.array([
        [1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr],
    ])
    rotation_y = np.array([
        [cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp],
    ])
    return rotation_x @ rotation_y @ yaw_rotation

def ecef(lla):
    lat, lon, alt = map(float, lla); lat = math.radians(lat); lon = math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(n + alt) * math.cos(lat) * math.cos(lon), (n + alt) * math.cos(lat) * math.sin(lon), (n * (1.0 - WGS84_E2) + alt) * math.sin(lat)])

def lla_to_enu(lla, config):
    origin = config["enu_origin_lla"]; lat0 = math.radians(float(origin[0])); lon0 = math.radians(float(origin[1]))
    delta = ecef(lla) - ecef(origin)
    ecef_to_enu = np.array([[-math.sin(lon0), math.cos(lon0), 0], [-math.sin(lat0)*math.cos(lon0), -math.sin(lat0)*math.sin(lon0), math.cos(lat0)], [math.cos(lat0)*math.cos(lon0), math.cos(lat0)*math.sin(lon0), math.sin(lat0)]])
    return ecef_to_enu @ delta

def lla_to_map(lla, config):
    enu = lla_to_enu(lla, config)
    rotation = enu_to_map_rotation(config)
    return rotation @ enu + np.asarray(config["map_translation_m"], dtype=float)

def gnss_fix(record, config):
    from kiss_slam.gnss import GnssFix
    covariance = record.get("covariance")
    hdop = record.get("hdop") if config["quality_policy"] == "literal_hdop_or_covariance" else None
    lla = (record["latitude"], record["longitude"], record["altitude"])
    raw_position = lla_to_enu(lla, config)
    if covariance is not None:
        rotation = enu_to_map_rotation(config)
        covariance = rotation @ np.asarray(covariance, dtype=float) @ rotation.T
    return GnssFix(timestamp=record["timestamp"] + float(config.get("timestamp_offset_s", 0.0)), position=lla_to_map(lla, config), hdop=float(hdop) if hdop is not None else None, covariance=covariance, raw_position=raw_position)
