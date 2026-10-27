"""
Saarthi — Location Manager (Ola Maps Reverse Geocoding)
======================================================
Provides real-time spatial and street-level location awareness for
visually impaired individuals in India using the Ola Maps API
(via Krutrim Cloud) and Wi-Fi / IP fallback coordinates.

Usage:
    from backend.navigation.locator import LocationManager

    locator = LocationManager()
    lat, lng = locator.get_coordinates()
    address = locator.get_street_address(lat, lng)
"""

import os
import time
from typing import Optional, Tuple

import geocoder
import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


class LocationManager:
    """
    Manages location tracking and reverse geocoding via Ola Maps.
    Supports phone GPS sensor, browser/client GPS synchronization,
    configurable environment coordinates, and IP fallback.
    """

    OLA_REVERSE_GEOCODE_URL = "https://api.olamaps.io/places/v1/reverse-geocode"

    def __init__(self, api_key: Optional[str] = None, sensor_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OLA_MAPS_API_KEY", "")
        self.sensor_url = sensor_url or os.getenv("IP_WEBCAM_SENSOR_URL", "")

        # High-precision synchronized coordinates from browser/dashboard navigator.geolocation
        self.synced_coords: Optional[Tuple[float, float]] = None
        self.synced_source: str = ""
        self.synced_timestamp: float = 0.0

        # Optional environment overrides (e.g., DEFAULT_LOCATION_LAT=28.6139, DEFAULT_LOCATION_LNG=77.2090)
        env_lat = os.getenv("DEFAULT_LOCATION_LAT")
        env_lng = os.getenv("DEFAULT_LOCATION_LNG")
        if env_lat and env_lng:
            try:
                self.default_coords = (float(env_lat), float(env_lng))
                print(f"[LocationManager] Configured default location override: {self.default_coords}")
            except ValueError:
                self.default_coords = None
        else:
            self.default_coords = None

        if not self.api_key:
            print("[LocationManager] WARNING: OLA_MAPS_API_KEY not found in environment or .env file.")
        else:
            print("[LocationManager] Initialized with Ola Maps API Key.")

    def update_coordinates(self, lat: float, lng: float, source: str = "Browser GPS"):
        """
        Dynamically update real GPS coordinates from browser navigator.geolocation,
        mobile client, or caregiver dashboard override.
        """
        if lat is not None and lng is not None and (lat != 0.0 or lng != 0.0):
            self.synced_coords = (float(lat), float(lng))
            self.synced_source = source
            self.synced_timestamp = time.time()
            print(f"[LocationManager] Location synchronized from {source}: {lat:.6f}, {lng:.6f}", flush=True)

    def set_sensor_source(self, source_url: str):
        """
        Dynamically configures the IP Webcam sensor endpoint from the video stream URL.
        Example: http://192.168.1.15:8080/video -> http://192.168.1.15:8080/sensors.json
        """
        if isinstance(source_url, str) and "://" in source_url:
            clean_base = source_url.rstrip("/").split("/video")[0].split("/shot.jpg")[0]
            self.sensor_url = f"{clean_base}/sensors.json"
            print(f"[LocationManager] Sensor endpoint linked to phone: {self.sensor_url}", flush=True)

    def get_coordinates(self, sensor_url: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetches true coordinates following strict priority:
          1. Phone IP Webcam live GPS sensor (if active and lock acquired).
          2. Synchronized client / browser GPS from dashboard navigator.geolocation.
          3. Default environment location override (DEFAULT_LOCATION_LAT/LNG).
          4. IP-based geolocation fallback (last resort).

        Returns:
            Tuple of (lat, lng) as floats, or (None, None) if unavailable.
        """
        target_sensor_url = sensor_url or self.sensor_url

        # 1. Attempt high-accuracy GPS readout from phone IP Webcam sensor API
        if target_sensor_url:
            if not target_sensor_url.endswith("/sensors.json"):
                target_sensor_url = target_sensor_url.rstrip("/") + "/sensors.json"

            try:
                resp = requests.get(target_sensor_url, timeout=2.0)
                if resp.status_code == 200:
                    sensor_data = resp.json()
                    gps_entry = sensor_data.get("gps", {})
                    data_points = gps_entry.get("data", [])
                    if data_points and len(data_points) > 0:
                        latest_entry = data_points[-1]
                        if isinstance(latest_entry, (list, tuple)) and len(latest_entry) >= 2:
                            coords = latest_entry[1]
                            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                                lat = float(coords[0])
                                lng = float(coords[1])
                                if lat != 0.0 or lng != 0.0:
                                    print(f"[LocationManager] Phone GPS Sensor lock: {lat:.6f}, {lng:.6f}", flush=True)
                                    return lat, lng
            except Exception as e:
                pass

        # 2. Use synchronized client / browser GPS coordinates
        if self.synced_coords is not None:
            lat, lng = self.synced_coords
            print(f"[LocationManager] Using synchronized {self.synced_source} coordinates: {lat:.6f}, {lng:.6f}", flush=True)
            return lat, lng

        # 3. Use configured default location override if present
        if self.default_coords is not None:
            lat, lng = self.default_coords
            print(f"[LocationManager] Using configured default coordinates: {lat:.6f}, {lng:.6f}", flush=True)
            return lat, lng

        # 4. Fallback to IP/Wi-Fi geolocation (warn that ISP gateway may be in a different city)
        try:
            g = geocoder.ip("me")
            if g and g.latlng and len(g.latlng) == 2:
                lat, lng = float(g.latlng[0]), float(g.latlng[1])
                print(f"[LocationManager Warning] Using IP fallback coordinates ({g.city}, {g.state}): {lat:.4f}, {lng:.4f}. Sync browser GPS in Dashboard for street-level accuracy.", flush=True)
                return lat, lng
            else:
                print("[LocationManager] Failed to resolve coordinates via IP fallback.")
                return None, None
        except Exception as e:
            print(f"[LocationManager] Error fetching coordinates: {e}")
            return None, None

    def get_street_address(self, lat: Optional[float], lng: Optional[float]) -> str:
        """
        Performs reverse geocoding via Ola Maps API to obtain a formatted street address.

        Args:
            lat: Latitude float.
            lng: Longitude float.

        Returns:
            Human-readable address string.
        """
        if lat is None or lng is None:
            return "Unable to determine your current location."

        if not self.api_key:
            return "Ola Maps API key is not configured."

        try:
            params = {
                "latlng": f"{lat},{lng}",
                "api_key": self.api_key,
            }
            response = requests.get(
                self.OLA_REVERSE_GEOCODE_URL,
                params=params,
                timeout=5,
            )

            if response.status_code != 200:
                print(f"[LocationManager] Ola Maps API error {response.status_code}: {response.text}")
                return "Location service is currently unavailable."

            data = response.json()
            results = data.get("results", [])

            if results and len(results) > 0:
                first_result = results[0]
                formatted_address = first_result.get("formatted_address")
                if formatted_address:
                    return formatted_address.strip()

                name = first_result.get("name")
                if name:
                    return name.strip()

            error_msg = data.get("error_message")
            if error_msg:
                print(f"[LocationManager] Ola Maps error message: {error_msg}")

            return "Unknown location."

        except requests.exceptions.Timeout:
            print("[LocationManager] Request to Ola Maps API timed out.")
            return "Location request timed out."
        except Exception as e:
            print(f"[LocationManager] Exception during reverse geocoding: {e}")
            return "Could not retrieve address."

    def get_current_address(self) -> str:
        """
        Convenience method to resolve coordinates and return the reverse-geocoded address.
        """
        lat, lng = self.get_coordinates()
        return self.get_street_address(lat, lng)


if __name__ == "__main__":
    locator = LocationManager()
    print("Fetching coordinates...")
    lat, lng = locator.get_coordinates()
    print(f"Coordinates: {lat}, {lng}")
    if lat and lng:
        print("Fetching address from Ola Maps...")
        addr = locator.get_street_address(lat, lng)
        print(f"Address: {addr}")
