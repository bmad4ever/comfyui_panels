from PIL import Image, ImageDraw
import math
from shapely import Polygon


# region Subroutines and DRY

def _setup_img_draw(canvas, upscale):
    """
    Mind checking None in draw, which targets the edge-case of width or height being less or equal to zero!
    """
    xmin, ymin, xmax, ymax = canvas.bounds
    width = int(math.ceil(xmax - xmin))
    height = int(math.ceil(ymax - ymin))
    if width <= 0 or height <= 0:
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = None
    else:
        W, H = round(width * upscale), round(height * upscale)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
    return draw, height, img, width, xmin, ymax


def _compute_cumulative_lengths(points, closed=True):
    """Compute cumulative arc lengths of a polyline/polygon."""
    lengths = [0.0]
    total = 0.0
    n = len(points)
    end = n if not closed else n + 1
    for i in range(1, end):
        x1, y1 = points[i - 1]
        x2, y2 = points[i % n]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        total += seg_len
        lengths.append(total)
    return lengths, total


def _interpolate_point(points, lengths, s, closed=True):
    """Interpolate a point along the polyline at arc length s."""
    n = len(points)
    end = n if not closed else n + 1
    for i in range(1, end):
        if s <= lengths[i]:
            t = (s - lengths[i - 1]) / (lengths[i] - lengths[i - 1] + 1e-12)
            x1, y1 = points[i - 1]
            x2, y2 = points[i % n]
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return points[-1]

# endregion Subroutines and DRY


def _draw_dashed(draw: ImageDraw, points, width=2.0, dash_length=10.0, gap_length=5.0, fill=(0, 0, 0, 255)):
    """Draw a polygon with dashed outline, evenly spaced along the perimeter."""
    lengths, total_length = _compute_cumulative_lengths(points, closed=True)

    # adjust spacing to nearest value that can close the loop without messing the spacing
    adj_spacing  = total_length / round(total_length / (dash_length + gap_length))
    adjust_scale = adj_spacing  / (dash_length + gap_length)
    dash_length *= adjust_scale
    gap_length  *= adjust_scale


    pos = 0.0
    while pos < total_length:
        start_pos = pos
        end_pos = min(pos + dash_length, total_length)

        sx, sy = _interpolate_point(points, lengths, start_pos)
        ex, ey = _interpolate_point(points, lengths, end_pos)

        print(f"NEW -> {fill}")
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

        pos += dash_length + gap_length


def _draw_dotted(draw: ImageDraw, points, radius=2.0, spacing=10.0, fill=(0, 0, 0, 255)):
    """Draw a polygon with dotted outline, evenly spaced along the perimeter."""
    lengths, total_length = _compute_cumulative_lengths(points, closed=True)

    # adjust spacing to nearest value that can close the loop without messing the spacing
    spacing = total_length / round(total_length / spacing)

    pos = 0.0
    while pos <= total_length:
        cx, cy = _interpolate_point(points, lengths, pos)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
        pos += spacing


def _draw_polygons_contours(
    draw_function,
    polys: list[Polygon],
    canvas: Polygon,
    upscale: int = 4,
) -> Image.Image:
    """
    Draw polygon contours on a transparent RGBA image and return the image.
    """
    draw, height, img, width, xmin, ymax = _setup_img_draw(canvas, upscale)
    if draw is None:
        return img

    for poly in polys:
        if poly is None or not isinstance(poly, Polygon):
            continue

        # exterior
        ext = [(x*upscale, y*upscale) for x, y in poly.exterior.coords]
        if len(ext) >= 2:
            draw_function(draw, ext)

        # holes
        for interior in poly.interiors:
            coords = [(x*upscale, y*upscale) for x, y in interior.coords]
            if len(coords) >= 2:
                draw_function(draw, ext)

    return img.resize((width, height), Image.Resampling.LANCZOS)


def draw_polygons_contours_line(
    polys: list[Polygon],
    canvas: Polygon,
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    stroke_width: float = 1,
    upscale: int = 4,
) -> Image.Image:
    scaled_width = round(stroke_width * upscale)

    def draw_func(draw: ImageDraw, points: list):
        extend_end = max(2, math.ceil(len(points)/4))   # ensure the last corner is properly drawn
        draw.line(points + points[0:extend_end], fill=stroke_color, width=scaled_width, joint="curve")

    return _draw_polygons_contours(draw_func, polys, canvas, upscale)


def draw_polygons_contours_dashed(
    polys: list[Polygon],
    canvas: Polygon,
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    stroke_width: float= 1,
    dash_length: float = 1,
    gap_length: float = 1,
    upscale: int = 4,
) -> Image.Image:
    scaled_width = round(stroke_width * upscale)
    scaled_dash = dash_length * upscale
    scaled_gap = gap_length * upscale

    def draw_func(draw: ImageDraw, points: list):
        _draw_dashed(draw, points + [points[0]], scaled_width, scaled_dash, scaled_gap, stroke_color)

    return _draw_polygons_contours(draw_func, polys, canvas, upscale)


def draw_polygons_contours_dotted(
    polys: list[Polygon],
    canvas: Polygon,
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    dot_radius: float  = 1,
    dot_spacing: float = 1,
    upscale: int = 4,
) -> Image.Image:
    scaled_radius = dot_radius * upscale
    scaled_spacing = dot_spacing * upscale

    def draw_func(draw: ImageDraw, points: list):
        _draw_dotted(draw, points + [points[0]], scaled_radius, scaled_spacing, stroke_color)

    return _draw_polygons_contours(draw_func, polys, canvas, upscale)


