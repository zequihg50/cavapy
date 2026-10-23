import numpy as np

from cavapy.resources.loader import load_json

def ray_casting(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

class Country:
    def __init__(self, name):
        self._name = name
        countries = load_json("ne_110m_admin_0_countries.geojson")
        country = next((
            feature["geometry"]
            for feature in countries["features"]
            if feature["properties"]["ADMIN"] == name),
            None)
    
        if country is None:
            raise ValueError(f'Country "{name}" not found. Please provide a valid country name.')
    
        self._country = country

    @property
    def name(self):
        return self._name

    @property
    def country(self):
        return self._country

    @property
    def bbox(self):
        longitudes, latitudes = [p[0] for p in self.coords], [p[1] for p in self.coords]
        
        return min(longitudes), min(latitudes), max(longitudes), max(latitudes)

    @property
    def coords(self):
        if self.country["type"] == "Polygon":
            polygons = [self.country["coordinates"]]
        elif self.country["type"] == "MultiPolygon":
            polygons = self.country["coordinates"]
        else:
            raise ValueError(f"Unsupported geometry type: {self.country['type']}")

        coords = [coord for polygon in polygons for coord in polygon[0]]

        return coords

    def mask(self, lats, lons):
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        
        for i, lat_val in enumerate(lats):
            for j, lon_val in enumerate(lons):
                mask[i, j] = ray_casting(lon_val, lat_val, self.coords)

        return mask

    def __str__(self):
        return f"{self._name}: {self.bbox}"
