# NOVA - Color Detection Module
# Detects the dominant color(s) of an object and builds a natural English sentence.
#
# How it works (short version):
#   1. Crop the object's region from the frame using its bounding box.
#   2. Convert that region to HSV and group similar pixels using K-means clustering.
#   3. Match each group's center color to the closest named color in COLOR_DATABASE.
#   4. Turn the result into a sentence like "I see a red car".

import cv2
import numpy as np
from collections import Counter
import colorsys

# ------------------------------------------------------------------
# Color database: name -> HSV range(s)
# Each range is (lower_bound, upper_bound). Red needs two ranges
# because it wraps around both ends of the Hue scale (0 and 180).
# ------------------------------------------------------------------
COLOR_DATABASE = {
    "red":    [(0, 100, 50), (10, 255, 255), (170, 100, 50), (180, 255, 255)],
    "orange": [(10, 100, 100), (25, 255, 255)],
    "yellow": [(25, 100, 100), (35, 255, 255)],
    "green":  [(35, 50, 50), (85, 255, 255)],
    "cyan":   [(85, 100, 100), (100, 255, 255)],
    "blue":   [(100, 80, 50), (140, 255, 255)],
    "purple": [(140, 50, 50), (160, 255, 255)],
    "pink":   [(160, 50, 100), (180, 255, 255)],
    "brown":  [(10, 50, 20), (25, 150, 130)],
}

GRAYSCALE_NAMES = ["black", "white", "gray"]


class ColorOrganizer:
    """Detects dominant colors in an image region and describes them in English."""

    def __init__(self):
        self.min_saturation_for_color = 25

    def get_dominant_colors(self, frame, bbox=None, top_n=3):
        roi = self._crop_region(frame, bbox)
        if roi is None or roi.size == 0:
            return []

        roi = self._resize_for_speed(roi)
        hsv_pixels = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

        k = min(top_n + 2, 6)
        centers, counts = self._kmeans_cluster(hsv_pixels, k)

        return self._centers_to_color_list(centers, counts, top_n)

    def _crop_region(self, frame, bbox):
        if bbox is None:
            return frame
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _resize_for_speed(self, roi):
        h, w = roi.shape[:2]
        if max(h, w) > 200:
            scale = 200 / max(h, w)
            roi = cv2.resize(roi, None, fx=scale, fy=scale)
        return roi

    def _kmeans_cluster(self, pixels, k):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = Counter(labels.flatten())
        return centers, counts

    def _centers_to_color_list(self, centers, counts, top_n):
        total_pixels = sum(counts.values())
        results = []

        for i, (h, s, v) in enumerate(centers):
            name = self._hsv_to_name(h, s, v)
            percentage = (counts[i] / total_pixels) * 100

            if name in ("black", "white") and percentage < 25:
                continue
            if percentage <= 5:
                continue

            results.append({
                "color": name,
                "percentage": round(percentage, 1),
                "hex": self._hsv_to_hex(h, s, v),
            })

        results.sort(key=lambda c: c["percentage"], reverse=True)
        return results[:top_n]

    def _hsv_to_name(self, h, s, v):
        if s < self.min_saturation_for_color:
            if v < 40:
                return "black"
            if v > 220:
                return "white"
            return "gray"

        best_match, best_distance = "unknown", float("inf")

        for name, ranges in COLOR_DATABASE.items():
            for j in range(0, len(ranges), 2):
                lower, upper = ranges[j], ranges[j + 1]
                if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                    return name

                center_h = (lower[0] + upper[0]) / 2
                center_s = (lower[1] + upper[1]) / 2
                center_v = (lower[2] + upper[2]) / 2
                h_diff = min(abs(h - center_h), 180 - abs(h - center_h))
                distance = np.sqrt(h_diff**2 + (s - center_s)**2 + (v - center_v)**2)

                if distance < best_distance:
                    best_distance, best_match = distance, name

        return best_match

    def _hsv_to_hex(self, h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h / 179, s / 255, v / 255)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def describe_object(self, object_name, colors, distance_text=None):
        if not colors:
            sentence = f"I see a {object_name}"
        elif len(colors) == 1:
            sentence = f"I see a {colors[0]['color']} {object_name}"
        else:
            main_color = colors[0]["color"]
            second_color = colors[1]["color"]
            if abs(colors[0]["percentage"] - colors[1]["percentage"]) < 15:
                sentence = f"I see a {object_name}, a mix of {main_color} and {second_color}"
            else:
                sentence = f"I see a {main_color} {object_name} with some {second_color}"

        if distance_text:
            sentence += f", about {distance_text} away"

        return sentence

    def describe_scene(self, objects_data):
        if not objects_data:
            return "I don't see anything clearly."
        return ". ".join(self.describe_object(name, colors, dist) for name, colors, dist in objects_data)

    def draw_color_info(self, frame, bbox, colors):
        x1, y1, x2, y2 = map(int, bbox)
        swatch_size = 18
        swatch_x = min(x2 + 5, frame.shape[1] - 100)

        for i, c in enumerate(colors[:3]):
            bgr = self._hex_to_bgr(c["hex"])
            y = y1 + i * (swatch_size + 4)

            cv2.rectangle(frame, (swatch_x, y), (swatch_x + swatch_size, y + swatch_size), bgr, -1)
            cv2.rectangle(frame, (swatch_x, y), (swatch_x + swatch_size, y + swatch_size), (255, 255, 255), 1)

            label = f"{c['color']} {c['percentage']:.0f}%"
            cv2.putText(frame, label, (swatch_x + swatch_size + 4, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _hex_to_bgr(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return (b, g, r)


if __name__ == "__main__":
    organizer = ColorOrganizer()

    test_cases = [
        ("shirt", [{"color": "blue", "percentage": 85, "hex": "#0000ff"}]),
        ("car", [{"color": "red", "percentage": 70, "hex": "#ff0000"},
                 {"color": "black", "percentage": 25, "hex": "#000000"}]),
    ]

    for obj, colors in test_cases:
        print(organizer.describe_object(obj, colors, "2 meters"))
