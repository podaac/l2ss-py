import unittest
from shapely.geometry import Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from podaac.subsetter.subset import translate_longitude

class TestTranslateLongitude(unittest.TestCase):

    def test_translate_point(self):
        point = Point(-170, 10)
        translated_point = translate_longitude(point)
        expected_point = Point(190, 10)  # -170 + 360 = 190
        self.assertEqual(translated_point, expected_point)

    def test_translate_polygon_2d(self):
        polygon_2d = Polygon([(-170, 10), (-180, 20), (-190, 10)])
        translated_polygon_2d = translate_longitude(polygon_2d)
        expected_polygon_2d = Polygon([(190, 10), (180, 20), (170, 10)])  # Translate by adding 360
        self.assertEqual(translated_polygon_2d, expected_polygon_2d)

    def test_translate_polygon_3d(self):
        polygon_3d = Polygon([(-170, 10, 1), (-180, 20, 2), (-190, 10, 3)])
        translated_polygon_3d = translate_longitude(polygon_3d)
        expected_polygon_3d = Polygon([(190, 10, 1), (180, 20, 2), (170, 10, 3)])  # Translate by adding 360
        self.assertEqual(translated_polygon_3d, expected_polygon_3d)

    def test_translate_multipolygon(self):
        polygon_1 = Polygon([(-170, 10), (-180, 20), (-190, 10)])
        polygon_2 = Polygon([(10, 20), (20, 30), (30, 20)])
        multi_polygon = MultiPolygon([polygon_1, polygon_2])
        translated_multi_polygon = translate_longitude(multi_polygon)
        expected_polygon_1 = Polygon([(190, 10), (180, 20), (170, 10)])
        expected_polygon_2 = Polygon([(10, 20), (20, 30), (30, 20)])
        expected_multi_polygon = MultiPolygon([expected_polygon_1, expected_polygon_2])
        self.assertEqual(translated_multi_polygon, expected_multi_polygon)

    def test_identity_translation(self):
        # Test for identity translation (no change)
        point = Point(30, 40)
        translated_point = translate_longitude(point)
        self.assertEqual(translated_point, point)

        polygon = Polygon([(30, 40), (40, 50), (50, 40)])
        translated_polygon = translate_longitude(polygon)
        self.assertEqual(translated_polygon, polygon)

    def test_unknown_geometry(self):
        # Test for handling unknown geometry types
        geometry = "This is not a valid geometry"
        translated_geometry = translate_longitude(geometry)
        self.assertEqual(translated_geometry, geometry)

if __name__ == '__main__':
    unittest.main()


