from __future__ import annotations
from typing import Optional

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split, unary_union
from shapely.geometry.polygon import orient
from shapely import affinity
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import random
import math
import copy


# -----------------------
# Split presets & constants
# -----------------------
phi = (1.0 + math.sqrt(5.0)) / 2.0
SPLIT_MODES = {
    0: 0.5,  # midpoint (default)
    1: 1.0 / 3.0,  # 1/3
    2: 2.0 / 3.0,  # 2/3
    3: 1.0 - 1.0 / phi,
    4: 1.0 / phi,  # 1/phi
}


class CutNode:
    def __init__(self, vertical: bool, angle: int = 0, split_mode: int = 0):
        """
        vertical: True => vertical cuts; False => horizontal
        angle: integer degrees (slant)
        split_mode: index into SPLIT_MODES (0 allowed many cuts, nonzero -> only 1 cut)
        """
        self.vertical: bool = vertical
        self.angle: int = int(angle)
        self.split_mode: int = split_mode
        self.children: list[Optional[CutNode]] = []

    def __deepcopy__(self, memo):
        node = CutNode(self.vertical, self.angle, self.split_mode)
        node.children = [
            copy.deepcopy(c, memo) if c is not None else None
            for c in self.children
        ]
        return node

    @property
    def cuts(self) -> int:
        return max(0, len(self.children) - 1)

    def add_child(self, child: Optional[CutNode] = None) -> None:
        # enforce non-midpoint single-cut rule at add time
        if self.split_mode != 0 and len(self.children) >= 2:
            raise ValueError("Non-midpoint split modes only support one cut (two children).")
        self.children.append(child)

    def to_compact(self) -> str:
        """Serialize compactly: V15_3(...). Always include split_mode for simplicity."""
        prefix = "V" if self.vertical else "H"
        s = f"{prefix}{self.angle}_{self.split_mode}"
        if self.children:
            s += "(" + ",".join(c.to_compact() if c else "" for c in self.children) + ")"
        return s

    @staticmethod
    def from_compact(data: str) -> Optional[CutNode]:
        """Parser. Builds the embedded cut tree."""
        if not data or not data.strip():
            return None

        def _split_top_level_commas(s: str) -> list[str]:
            tokens: list[str] = []
            buf: list[str] = []
            depth = 0
            for ch in s:
                if ch == "(":
                    depth += 1
                    buf.append(ch)
                elif ch == ")":
                    depth -= 1
                    buf.append(ch)
                elif ch == "," and depth == 0:
                    tokens.append("".join(buf))
                    buf = []
                else:
                    buf.append(ch)
            tokens.append("".join(buf))
            return tokens

        def _parse_node_str(s: str) -> Optional[CutNode]:
            s = s.strip()
            if not s:
                return None
            i = 0
            n = len(s)
            if s[i] not in ("V", "H"):
                raise ValueError(f"Expected 'V' or 'H' at start of node: {s!r}")
            vertical = s[i] == "V"
            i += 1

            # integer angle (supports sign)
            angle_str = ""
            while i < n and (s[i].isdigit() or s[i] in "-+"):
                angle_str += s[i]
                i += 1
            angle = int(angle_str) if angle_str else 0

            # optional split_mode after '_'
            split_mode = 0
            if i < n and s[i] == "_":
                i += 1
                mode_str = ""
                while i < n and s[i].isdigit():
                    mode_str += s[i]
                    i += 1
                split_mode = int(mode_str) if mode_str else 0

            node = CutNode(vertical, angle, split_mode)

            # children (optional)
            if i < n and s[i] == "(":
                # find matching closing paren
                j = i + 1
                depth = 0
                while j < n:
                    if s[j] == "(":
                        depth += 1
                    elif s[j] == ")":
                        if depth == 0:
                            break
                        depth -= 1
                    j += 1
                if j >= n or s[j] != ")":
                    raise ValueError(f"Unmatched '(' in node string: {s!r}")
                inner = s[i + 1: j]
                child_tokens = _split_top_level_commas(inner)
                for tok in child_tokens:
                    tok = tok.strip()
                    if tok == "":
                        node.add_child(None)
                    else:
                        child = _parse_node_str(tok)
                        node.add_child(child)
                # i = j + 1
            return node

        return _parse_node_str(data.strip())

    @staticmethod
    def _weighted_midpoint_of_lines(intersection_geom) -> Optional[tuple[float, float]]:
        """
        Given the result of polygon.intersection(line) (maybe LineString, MultiLineString, Point, GeometryCollection),
        compute a robust midpoint: weighted average of segment midpoints by their length.
        """
        if intersection_geom.is_empty:
            return None

        # single line
        if intersection_geom.geom_type == "LineString":
            coords = list(intersection_geom.coords)
            x0, y0 = coords[0]
            x1, y1 = coords[-1]
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        # point
        if intersection_geom.geom_type == "Point":
            return (intersection_geom.x, intersection_geom.y)

        # multi or collection: gather lines and points
        total_len = 0.0
        sum_x = 0.0
        sum_y = 0.0
        geoms = getattr(intersection_geom, "geoms", [])  # Shapely 1.x and 2.x compat
        for g in geoms:
            if g.geom_type == "LineString":
                coords = list(g.coords)
                x0, y0 = coords[0]
                x1, y1 = coords[-1]
                midx, midy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                L = g.length
                sum_x += midx * L
                sum_y += midy * L
                total_len += L
            elif g.geom_type == "Point":
                sum_x += g.x
                sum_y += g.y
                total_len += 1.0
        if total_len <= 0:
            return None
        return (sum_x / total_len, sum_y / total_len)

    @staticmethod
    def _build_seam_lines_for_panel(
            panel: Polygon,
            vertical: bool,
            angle_deg: int,
            cuts: int,
            split_mode: int,
    ) -> list[LineString]:
        """
        Produce a list of seam LineString objects for the given panel.
        - cuts: suggested number of cuts (children-1). If split_mode != 0 then we only allow 1 cut.
        - split_mode chooses ratio presets (if 0 we use evenly spaced j/(cuts+1)).
        Each seam is centered so its midpoint (intersection with panel) lies on the reference point.
        """
        if cuts <= 0 or panel.is_empty:
            return []

        if split_mode != 0:
            # enforce single cut for non-midpoint modes
            cuts = 1

        xmin, ymin, xmax, ymax = panel.bounds
        width = xmax - xmin
        height = ymax - ymin
        pad = max(width, height) * 10.0 + 1.0  # long enough line to cross panel
        cx, cy = panel.centroid.x, panel.centroid.y

        seam_lines: list[LineString] = []

        for j in range(1, cuts + 1):
            # compute ratio
            if split_mode == 0:
                ratio = j / (cuts + 1)
            else:
                ratio = SPLIT_MODES.get(split_mode, 0.5)

            if vertical:
                c = xmin + (xmax - xmin) * ratio
                base = LineString([(c, ymin - pad), (c, ymax + pad)])
                ref_point = Point(c, (ymin + ymax) / 2.0)
            else:
                c = ymin + (ymax - ymin) * ratio
                base = LineString([(xmin - pad, c), (xmax + pad, c)])
                ref_point = Point((xmin + xmax) / 2.0, c)

            # rotate around centroid
            line_rot = affinity.rotate(base, angle_deg, origin=(cx, cy))

            # compute intersection with panel and midpoint
            inter = panel.intersection(line_rot)
            mid = CutNode._weighted_midpoint_of_lines(inter)
            if mid is not None:
                dx = ref_point.x - mid[0]
                dy = ref_point.y - mid[1]
                line_centered = affinity.translate(line_rot, xoff=dx, yoff=dy)
            else:
                # no intersection found, keep rotated line (it will probably not split)
                line_centered = line_rot

            # Option: sanity - only keep the line if it intersects the panel
            if not panel.intersects(line_centered):
                # skip seam that doesn't intersect (degenerate)
                continue

            seam_lines.append(LineString(line_centered))  # make a copy

        return seam_lines

    @staticmethod
    def partition_panel(panel: Polygon, node: CutNode, margin_here: float) -> list[Polygon]:
        """
        Given a panel polygon and a CutNode (for that panel), compute the list of sub-panels
        after applying node's cuts and carving the margin gaps only on seams.

        Returns list of polygons in ascending order along the cut axis (left->right for vertical,
        bottom->top for horizontal).
        """
        if panel.is_empty or node is None or len(node.children) == 0:
            return [panel]

        # number of cuts (children - 1); enforce split_mode rule inside seam builder
        cuts = node.cuts

        # 1) build seam lines (centered)
        seam_lines = CutNode._build_seam_lines_for_panel(panel, node.vertical, node.angle, cuts, node.split_mode)

        if not seam_lines:
            return [panel]

        # 2) unify seams into a single multilinestring for splitting
        seam_union = unary_union(seam_lines)

        # 3) split panel by seam union
        split_result = split(panel, seam_union)
        raw_pieces = [g for g in getattr(split_result, "geoms", [split_result]) if g.geom_type == "Polygon"]

        if not raw_pieces:
            # fallback
            return [panel]

        # 4) order raw pieces along the primary axis (so child mapping is deterministic)
        if node.vertical:
            raw_pieces.sort(key=lambda p: p.centroid.x)
        else:
            raw_pieces.sort(key=lambda p: p.centroid.y)

        # 5) build margin gaps as buffer of seam lines and subtract them from each raw piece
        if margin_here is not None and margin_here > 0:
            offsets = margin_here / 2.0
            # rectangular-ish gaps using square caps (cap_style=2) to keep straight edges
            gap_polys = [s.buffer(offsets, cap_style=2) for s in seam_lines]
            gap_union = unary_union(gap_polys)
            adjusted = []
            for rp in raw_pieces:
                diff = rp.difference(gap_union)
                # difference may produce Polygon or MultiPolygon; keep as-is
                if diff.is_empty:
                    # If fully removed, keep an empty placeholder (to preserve counts)
                    adjusted.append(Polygon())
                else:
                    adjusted.append(diff)
        else:
            adjusted = raw_pieces

        # 6) result should have len == node.cuts+1 (or close); if not, pad with empties
        expected = max(1, (node.cuts + 1))
        while len(adjusted) < expected:
            adjusted.append(Polygon())

        return adjusted

    @staticmethod
    def process_tree(node: Optional[CutNode], panel: Polygon, margin: float = 0.0, rtl=False, depth: int = 0) -> list[Polygon]:
        """
        Process the cut tree, starting at `panel`. Uses exact seam-based margin carving.
        margin is the top-level margin; it is decayed by phi per depth:
            margin_at_depth = margin / (phi ** depth)

        rtl = True, then panels are sorted from right to left, instead of from left to right
        """
        if panel.is_empty:
            return []
        if node is None or len(node.children) == 0:
            return [panel]

        # compute margin for this depth
        current_margin = margin / (phi ** depth) if margin > 0 else 0.0

        # partition this panel into pieces and carve gaps (exact)
        pieces = CutNode.partition_panel(panel, node, current_margin)

        out: list[Polygon] = []
        # Now map children to pieces in order
        for child, piece in zip(node.children, pieces):
            if child is None:
                # leaf: piece is final (could be Polygon or MultiPolygon)
                out.append(piece)
            else:
                # recursive
                out.extend(CutNode.process_tree(child, piece, margin=margin, rtl=rtl, depth=depth + 1))

        if depth == 0 and rtl:
            # flip polygons
            xmin, ymin, xmax, ymax = panel.bounds
            cx = (xmin + xmax) / 2  # center x of canvas
            return [orient(affinity.scale(p, xfact=-1, yfact=1, origin=(cx, 0)), -1.0) for p in out]  # bruh

        return out

    @staticmethod
    def gen_rand_node(rng: random.Random, max_cuts: int = 2, min_angle: int = -15, max_angle: int = 15):
        vertical = rng.choice([True, False])
        angle = rng.randint(min_angle, max_angle)
        split_mode = rng.choice(list(SPLIT_MODES.keys()))
        cuts = rng.randint(1, max_cuts) if split_mode == 0 else 1
        node = CutNode(vertical, angle, split_mode)
        for _ in range(cuts + 1):
            node.add_child(None)
        return node


def layout_to_image(
    cut_tree: CutNode,
    rtl: bool = False,  # stored image in right-to-left format (x mirrored)
    canvas_width: int = 210,
    canvas_height: int = 297,
    font_size: int = 9,
    margin: int = 4,
    index_font_size: int = 10,
) -> tuple[Image, str]:
    """
    Save panels as a PNG image, showing both left-to-right and right-to-left indices.
    The cut tree's compact code is also drawn below and stored in PNG metadata.
    """
    from shapely.geometry import box
    from PIL import Image, ImageDraw, ImageFont

    root_panel = box(0, 0, canvas_width, canvas_height)
    polygons = CutNode.process_tree(cut_tree, root_panel, margin=0, rtl=rtl)
    base_im = panels_to_image(polygons, index_font_size, "pink" if rtl else "lightblue")
    compact_code = cut_tree.to_compact()

    # Extend image at bottom
    width, height = base_im.size
    extra_height = font_size + 2 * margin
    new_im = Image.new("RGBA", (width, height + extra_height), (255, 255, 255, 255))
    new_im.paste(base_im, (0, 0))

    # Draw compact code text
    draw = ImageDraw.Draw(new_im)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), compact_code, font=font)
    text_w = bbox[2] - bbox[0]
    # text_h = bbox[3] - bbox[1]
    x_pos = (width - text_w) // 2
    y_pos = height + margin
    draw.text((x_pos, y_pos), compact_code, font=font, fill=(0, 0, 0, 255))

    return new_im, compact_code


def panels_to_image(panels: list[Polygon], index_font_size: int = 10,
                    annotate_color: Optional[str] = "lightblue",
                    canvas: Optional[Polygon] = None) -> Image:
    """
    :param annotate_rtl: None->No annotations; False->left to right; True->right to left
    """
    import io
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.yaxis.set_inverted(True)

    if canvas is not None:
        x, y = canvas.exterior.xy
        ax.fill(x, y, alpha=1, color="white", edgecolor="black", linewidth=2)

    for poly in panels:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.6, edgecolor="black", linewidth=1)

    if annotate_color is not None:
        for idx, poly_ltr in enumerate(panels):
            cx, cy = poly_ltr.centroid.coords[0]

            bg_color = annotate_color

            ax.text(
                cx, cy, str(idx),
                ha="center", va="center",
                fontsize=index_font_size,
                color="black",
                bbox=dict(facecolor=bg_color, edgecolor="none", boxstyle="circle,pad=0.2", alpha=0.7)
            )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")
