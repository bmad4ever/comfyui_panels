from shapely.affinity import rotate, scale, translate
from shapely.geometry import box  # , Polygon
from PIL.PngImagePlugin import PngInfo
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import random
import torch
import json
import os
import re

from comfy.cli_args import args
from nodes import LoadImage
import folder_paths
import node_helpers

from .CutNode import *
from .aux_data import *

CATEGORY_PATH = "Bmad/Panels"
META_DATA_KEY = "cut_tree"


class IO_Types:
    PANEL_LAYOUT = "PANEL_LAYOUT"  # Cuts Tree ( technically a node of the tree )
    PANEL = "POLYGON"  # The layout panels. using original type to potentially re-use in or interface w/ other packages
    BBOX = "BBOX"
    BBOX_SNAP = "BBOX_SNAP"


def unwrap_bbox_as_ints(bbox, container_tensor=None) -> tuple[int, int, int, int]:
    """
    :param bbox: tuple w/ 4 floats ( as returned by polygon.bounds )
    :param container_tensor: image or mask comfy tensor. Constrains the output to be within this container bounds.
    """
    x0, y0, x1, y1 = bbox
    x0, y0 = round(x0), round(y0)   # TODO consider changing to floor and ceil on the next line
    x1, y1 = round(x1), round(y1)
    if container_tensor is not None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(container_tensor.shape[2], x1), min(container_tensor.shape[1], y1)
    return x0, y0, x1, y1


# region Core Nodes

class LoadPanelLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return LoadImage.INPUT_TYPES()

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL_LAYOUT,)
    OUTPUT_TOOLTIPS = ("'Abstract' Panel layout, represented as a tree of cuts.",)
    FUNCTION = "func"
    DESCRIPTION = "Load the panel layout embedded in an image."

    def func(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        cut_tree_code: str | None = img.info.get("cut_tree", None)  # should ret False on VALIDATE_INPUT I think... TBT
        if cut_tree_code is None:
            raise Exception("cut_tree metadata not found in provided image.")
        cut_tree = CutNode.from_compact(cut_tree_code)
        return (cut_tree,)

    @classmethod
    def IS_CHANGED(cls, image):
        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        is_image = LoadImage.VALIDATE_INPUTS(image)
        if not is_image:
            return False
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        return META_DATA_KEY in img.info


class SavePanelLayout:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (IO_Types.PANEL_LAYOUT,),
                "draw_as": ("BOOLEAN", {"default": False,
                                        "label_on": "right to left", "label_off": "left to right",
                                        "tooltip": "Flips the layout drawn on the stored image but the stored data is exactly the same."}),
                "filename_prefix": ("STRING", {"default": "PanelLayout",
                                               "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "func"

    OUTPUT_NODE = True

    CATEGORY = CATEGORY_PATH
    DESCRIPTION = "Saves an image of the layout with it embedded to your ComfyUI output directory."

    def func(self, layout: CutNode, draw_as, filename_prefix: str = "PanelLayout"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir))

        image, compact_code = layout_to_image(layout, draw_as)
        metadata = PngInfo()
        metadata.add_text("cut_tree", compact_code)

        filename_with_batch_num_removed = filename.replace("%batch_num%", "")
        file = f"{filename_with_batch_num_removed}_{counter:05}_.png"
        image.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)

        results = list()
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        return {"ui": {"images": results}}


class StringDecodePanelLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_code": ("STRING", {"default": "Paste the layout code here."}),
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL_LAYOUT,)
    FUNCTION = "func"

    def func(self, layout_code):
        layout_root_node = CutNode.from_compact(layout_code)
        return (layout_root_node,)


class StringEncodePanelLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (IO_Types.PANEL_LAYOUT,),
            }
        }

    OUTPUT_NODE = True
    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"

    def func(self, layout):
        str_code = CutNode.to_compact(layout)
        print(f"Encoded panel layout -> {str_code}")
        return (str_code,)


class BuildLayoutPanels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (IO_Types.PANEL_LAYOUT, {"tooltip":
                                                    "Root node of the 'abstract' cut's tree."}),
                "canvas": (IO_Types.PANEL, {"tooltip":
                                            "A box shaped polygon representing the area to be cut into the panels."}),
                "margin": ("INT", {"default": 32, "min": 0, "max": 1000, "tooltip":
                    "The distance (in pixels) between the panels formed by the 1st cut."
                    "The distance for nested cuts decreases the higher the depth in the layout hierarchy."}),
                "reading_dir": ("BOOLEAN", {"default": False,
                                            "label_on": "right to left", "label_off": "left to right",
                                            "tooltip":
                                            "Invert the panel layout horizontally to be read from right to left."}),
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("Panels (Shapely Polygons)",)
    FUNCTION = "func"
    DESCRIPTION = ("Obtains a list of panels from the provided layout."
                   "The panels are sorted with respect to hierarchy and defined reading order."
                   "For example: A vertical cut in left-to-right reading order will place, on the list, the panels"
                   " from the left side of the cut before to the panels on the right side of the cut.")

    def func(self, layout, canvas, margin, reading_dir):
        panels = CutNode.process_tree(layout, canvas, margin, reading_dir)
        return (panels,)


class CanvasPanel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 2480, "min": 0, "max": 5000}),
                "height": ("INT", {"default": 3508, "min": 0, "max": 5000}),
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL,)
    OUTPUT_TOOLTIPS = ("Panel (Shapely Polygon)",)
    FUNCTION = "func"
    DESCRIPTION = "Canvas bounds for panel related operations."

    def func(self, width, height):
        canvas = box(0, 0, width, height)
        return (canvas,)


class Panel2Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panel": (IO_Types.PANEL,),
                "canvas": (IO_Types.PANEL,),
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = ("MASK",)
    FUNCTION = "func"
    DESCRIPTION = "A mask representing the panel area on canvas."

    def func(self, panel, canvas):
        """Assumes no holes & no multipolygons."""

        xmin, ymin, xmax, ymax = canvas.bounds  # min should be zero, but better safe than sorry later
        w, h = int(xmax - xmin), int(ymax - ymin)

        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        coords = [(x - xmin, y - ymin) for x, y in panel.exterior.coords]
        draw.polygon(coords, fill=1, outline=1)

        # Convert to torch tensor (1, H, W)
        arr = np.array(img, dtype=np.uint8)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        print(f"tensor dim for mask {tensor.shape}")
        return torch.from_numpy(arr).unsqueeze(0)


class PreviewPanelLayout(SavePanelLayout):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    DESCRIPTION = ("Preview the Panel Layout.\n"
                   "Without any margins or any panel adjustments.")

    @classmethod
    def INPUT_TYPES(cls):
        types = SavePanelLayout.INPUT_TYPES()
        del types["required"]["filename_prefix"]
        return types


class PreviewPanels:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panels": (IO_Types.PANEL,),
            },
            "optional": {
                "canvas": (IO_Types.PANEL,)
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "func"
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_PATH

    def func(self, panels, canvas: Optional[list[Polygon]] = None, prompt=None, extra_pnginfo=None):
        canvas = None if canvas is None else canvas[0]
        prompt = None if prompt is None else prompt[0]
        extra_pnginfo = None if extra_pnginfo is None else extra_pnginfo[0]

        image = panels_to_image(panels, annotate_color="white", canvas=canvas, )

        #filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path("_", self.output_dir))

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        filename_with_batch_num_removed = filename.replace("%batch_num%", "")
        file = f"{filename_with_batch_num_removed}_{counter:05}_.png"
        image.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)

        results = list()
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        return {"ui": {"images": results}}

# endregion Core Nodes

# region Layout Generators


class GridPanelLayoutGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 4, "min": 1, "max": 32}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 32}),
                "vcut_first": ("BOOLEAN", {"default": False, "tooltip":
                    "Whether the cut orientation in the first node is vertical or horizontal."
                    "Cuts' width decreases the higher the depth on the layout hierarchy."
                    "The first cut(s) will be the widest."}),
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL_LAYOUT,)
    FUNCTION = "func"
    DESCRIPTION = "Generates a grid like layout (not its panels, use BuildLayoutPanels node to get the panels)."

    @staticmethod
    def grid_cut_tree(rows: int, cols: int, vertical_first: bool = False) -> Optional[CutNode]:
        """
        Generate a cut tree that produces an even grid of panels.

        Args:
            rows: number of rows
            cols: number of columns
            vertical_first: whether to slice vertically first (default True)

        Returns:
            CutNode root representing the grid
        """
        if rows <= 0 or cols <= 0:
            return None
        if rows == 1 and cols == 1:
            return None  # just one panel, no cuts

        def build_grid(r: int, c: int, cut_vertical: bool) -> Optional[CutNode]:
            if r == 1 and c == 1:
                return None

            if cut_vertical and c > 1:
                # vertical cut into c parts
                node = CutNode(vertical=True, angle=0, split_mode=0)
                for _ in range(c):
                    child = build_grid(r, 1, not cut_vertical) if r > 1 else None
                    node.add_child(child)
                return node
            elif not cut_vertical and r > 1:
                # horizontal cut into r parts
                node = CutNode(vertical=False, angle=0, split_mode=0)
                for _ in range(r):
                    child = build_grid(1, c, not cut_vertical) if c > 1 else None
                    node.add_child(child)
                return node
            else:
                # no further subdivision
                return None

        return build_grid(rows, cols, vertical_first)

    def func(self, rows, columns, vcut_first):
        layout = self.grid_cut_tree(rows, columns, vertical_first=vcut_first)
        return (layout,)


class RandomPanelLayoutGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_panels": ("INT", {"default": 5, "min": 2, "max": 32}),
                "max_cuts": ("INT", {"default": 2, "min": 1, "max": 9}),
                "min_angle": ("INT", {"default": -25, "min": -45, "max": 45}),
                "max_angle": ("INT", {"default": 25, "min": -45, "max": 45}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL_LAYOUT,)
    FUNCTION = "func"
    DESCRIPTION = "Generates random panel layout within the provided parameters."

    @staticmethod
    def random_cut_tree(
            num_panels: int,
            max_cuts: int = 2,
            min_angle: int = -15,
            max_angle: int = 15,
            seed: Optional[int] = None
    ) -> Optional[CutNode]:
        """
        Generate a random cut tree that produces approximately `num_panels` panels.
        """
        if num_panels <= 0:
            return None
        if num_panels == 1:
            return None  # single leaf

        rng = random.Random(seed)

        # 1st Generate nodes until we reach desired panel count
        panels = 1
        unused_nodes: list[CutNode] = []
        while panels < num_panels:
            remaining = num_panels - panels
            max_possible_cuts = min(max_cuts, remaining)  # not -1, cause the cuts are nested within a prior panel
            node = CutNode.gen_rand_node(rng, max_possible_cuts, min_angle, max_angle)
            panels += node.cuts
            unused_nodes.append(node)

        if not unused_nodes:
            return None

        # Build hierarchy after all nodes have been generated
        root = unused_nodes.pop(rng.randrange(len(unused_nodes)))
        used_nodes_available = [root]
        used_nodes_unavailable: list[CutNode] = []  # without empty children slots

        while unused_nodes:
            parent = rng.choice(used_nodes_available)
            child_idx = rng.choice([i for i, c in enumerate(parent.children) if c is None])
            node = unused_nodes.pop(rng.randrange(len(unused_nodes)))
            parent.children[child_idx] = node

            # update availability
            if all(c is not None for c in parent.children):
                used_nodes_available.remove(parent)
                used_nodes_unavailable.append(parent)

            used_nodes_available.append(node)

        return root

    def func(self, num_panels, max_cuts, min_angle, max_angle, seed):
        cut_tree = self.random_cut_tree(num_panels, max_cuts, min_angle, max_angle, seed)
        return (cut_tree,)


class MutatePanelLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "layout": (IO_Types.PANEL_LAYOUT,),
                "add_cut_prob": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": .005}),
                "rem_cut_prob": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": .005}),
                "num_cut_prob": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": .005}),
                "ang_adj_prob": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": .005}),
                "typ_cut_prob": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": .005}),
                "max_ang_delt": ("INT", {"default": 15, "min": 0, "max": 45}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.PANEL_LAYOUT,)
    OUTPUT_TOOLTIPS = ("Panel Layout",)
    FUNCTION = "func"
    DESCRIPTION = "Modifies an existing Panel Layout."

    @staticmethod
    def mutate_tree(
            node: CutNode,
            prob_add: float = 0.1,
            prob_remove: float = 0.1,
            prob_change_cuts: float = 0.1,
            prob_change_angle: float = 0.1,
            prob_change_split_mode: float = 0.05,  # NEW
            max_angle_delta: int = 15,
            seed: Optional[int] = None,
    ):
        """
        Recursively mutate a CutNode tree in place according to given probabilities.
        Deterministic if `seed` is provided.
        """
        if node is None:
            raise ValueError("node is can not be None")

        node = deepcopy(node)
        rng = random.Random(seed)

        def _mutate_node(n: CutNode):
            # 0) Recurse ( run from leafs to top to prevent endless operations )
            for child in n.children:
                if child is not None:
                    _mutate_node(child)

            removable_indices = [i for i, c in enumerate(n.children) if c is not None
                                 and rng.random() < prob_remove]  # roll for each individually
            addable_indices = [i for i, c in enumerate(n.children) if c is None
                               and rng.random() < prob_add]  # roll for each individually

            # 1) Add a cut
            for idx in addable_indices:
                n.children[idx] = CutNode.gen_rand_node(rng, 2, -max_angle_delta, max_angle_delta)

            # 2) Remove a cut (non None child)
            for idx in removable_indices:
                n.children[idx] = None

            # 3) Change number of cuts
            if n.split_mode == 0 and rng.random() < prob_change_cuts:
                target_cuts = max(1, rng.randint(1, len(n.children)))
                current_cuts = len(n.children) - 1
                if target_cuts > current_cuts:
                    for _ in range(target_cuts - current_cuts):
                        insert_idx = rng.randint(0, len(n.children))
                        n.children.insert(insert_idx, None)
                elif target_cuts < current_cuts:
                    removable_indices = [i for i, c in enumerate(n.children) if c is None]
                    rng.shuffle(removable_indices)
                    for idx in removable_indices[:current_cuts - target_cuts]:
                        n.children.pop(idx)

            # 4) Change angle
            if rng.random() < prob_change_angle:
                delta = rng.randint(-max_angle_delta, max_angle_delta)
                n.angle += delta

            # 5) Change split mode
            if rng.random() < prob_change_split_mode:
                available_modes = [k for k in SPLIT_MODES.keys() if k != n.split_mode]
                if available_modes:
                    n.split_mode = rng.choice(available_modes)
                    # ensure children are compatible: non-midpoint → only 1 cut
                    if n.split_mode != 0 and len(n.children) > 2:
                        n.children = n.children[:2]

        _mutate_node(node)
        return node

    def func(self, layout, add_cut_prob, rem_cut_prob, num_cut_prob, ang_adj_prob, typ_cut_prob, max_ang_delt, seed):
        tree = self.mutate_tree(layout, add_cut_prob, rem_cut_prob, num_cut_prob,
                                ang_adj_prob, typ_cut_prob, max_ang_delt, seed)
        return (tree,)

# endregion Layout Generators

# region  Polygon Operations


class OffsetPolygonBounds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": (IO_Types.PANEL,),
                "offset": ("FLOAT", {"default": 32, "min": -1000, "max": 1000, "step": 0.5, "tooltip":
                    "The distance (in pixels) to offset the polygons' edges"})
            },
            "optional": {
                "bbox_snap": (IO_Types.BBOX_SNAP, {"tooltip":
                                                       "Constrain the adjustment operation with respect to a bounding box"})
            }
        }

    RETURN_TYPES = (IO_Types.PANEL,)
    FUNCTION = "func"
    DESCRIPTION = "'Expand' the polygon, when using positive values; or 'erode' it using negative values."

    def func(self, polygon, offset: float, bbox_snap: Optional[BBoxSnap] = None):
        new_panel = self.offset_panel(polygon, offset, bbox_snap)
        return (new_panel,)

    @staticmethod
    def offset_panel(poly: Polygon,
                     distance: float,
                     bbox_snap: Optional[BBoxSnap] = None,
                     tol: float = 1e-6) -> Polygon:
        """
        Buffer polygon inward/outward while optionally snapping vertices along the bounding box edges.

        :param poly: Input polygon
        :param distance: Buffer distance (positive = dilation, negative = erosion)
        :param bbox_snap: Optional (xmin, ymin, xmax, ymax, snap_on_box) bounding box
            snap_on_box:
            - True: snap coordinates that lie on bbox edges
            - False: snap coordinates that do NOT lie on bbox edges
        :param tol: Tolerance to consider a vertex on bbox edge
        :return: Buffered polygon with snapped vertices
        """
        if poly.is_empty:
            return poly

        # Identify which coordinates (X or Y) are on bbox edges
        snap_info = {}
        if bbox_snap is not None:
            xmin, ymin, xmax, ymax = bbox_snap.as_tuple()
            for i, (x, y) in enumerate(poly.exterior.coords):
                on_x_edge = abs(x - xmin) < tol or abs(x - xmax) < tol
                on_y_edge = abs(y - ymin) < tol or abs(y - ymax) < tol

                if bbox_snap.snap_on_bbox:
                    # snap coordinates that ARE on bbox edges
                    snap_x = x if on_x_edge else None
                    snap_y = y if on_y_edge else None
                else:
                    # snap coordinates that are NOT on bbox edges
                    snap_x = x if not on_x_edge else None
                    snap_y = y if not on_y_edge else None

                if snap_x is not None or snap_y is not None:
                    snap_info[i] = (snap_x, snap_y)

        # Buffer polygon
        buffered = poly.buffer(distance, join_style=2)
        #buffered = shapely.buffer(poly, distance, join_style=2)
        if buffered.is_empty:
            return buffered

        # Snap X or Y components back
        if bbox_snap is not None and snap_info:
            coords = list(buffered.exterior.coords)
            for i, (snap_x, snap_y) in snap_info.items():
                if i < len(coords):
                    x_new = snap_x if snap_x is not None else coords[i][0]
                    y_new = snap_y if snap_y is not None else coords[i][1]
                    coords[i] = (x_new, y_new)
            buffered = Polygon(coords)

        return buffered


class BBoxSnapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "canvas": (IO_Types.PANEL,),
                "snap_if": ("BOOLEAN", {"default": False,
                                        "label_on": "on a bbox's edge", "label_off": "not on a bbox's edge",
                                        "tooltip":
                                            "If True, polygons points coordinates coinciding withthe given canvas' edges are not changed; "
                                            "their source points may still be moved, but do so without leaving the canvas' edges.\n"
                                            "This can be used to add extra space between panels.\n\n"
                                            "If False, only point coordinates that are on the box are moved.\n"
                                            "This can be used to add the page's margins."}),
            }
        }

    CATEGORY = CATEGORY_PATH
    RETURN_TYPES = (IO_Types.BBOX_SNAP,)
    FUNCTION = "func"
    DESCRIPTION = "Optional constraint for the 'Adjust Panel' operation."

    def func(self, canvas, snap_if):
        bbox_snap = BBoxSnap.from_polygon(canvas, snap_if)
        return (bbox_snap,)


class RotatePolygon:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": (IO_Types.PANEL,),
                "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0}),
                "origin": ("STRING", {"default": "center"}),  # "center", "centroid", or (x,y)
            }
        }

    RETURN_TYPES = (IO_Types.PANEL,)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH

    def func(self, polygon, angle, origin):
        new_poly = rotate(polygon, angle, origin=origin)
        return (new_poly,)


class ScalePolygon:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": (IO_Types.PANEL,),
                "xfact": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": .001}),
                "yfact": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": .001}),
                "origin": ("STRING", {"default": "center"}),
            }
        }

    RETURN_TYPES = (IO_Types.PANEL,)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH

    def func(self, polygon, xfact, yfact, origin):
        new_poly = scale(polygon, xfact=xfact, yfact=yfact, origin=origin)
        return (new_poly,)


class TranslatePolygon:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": (IO_Types.PANEL,),
                "xoff": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0}),
                "yoff": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0}),
            }
        }

    RETURN_TYPES = (IO_Types.PANEL,)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH

    def func(self, polygon, xoff, yoff):
        new_poly = translate(polygon, xoff=xoff, yoff=yoff)
        return (new_poly,)


class BevelPolygon:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panel": (IO_Types.PANEL,),
                "curvature": ("INT", {"default": 32, "min": 1, "max": 256}),
                "iterations": ("INT", {"default": 4, "min": 1, "max": 9}),
                "buffer_res": ("INT", {"default": 32, "min": 8, "max": 128}),
            }
        }

    RETURN_TYPES = (IO_Types.PANEL,)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH

    def func(self, panel: Polygon, curvature, iterations, buffer_res):
        p = panel
        for _ in range(iterations):
            out = p.buffer(curvature, join_style=3, resolution=buffer_res)
            back = out.buffer(-curvature, join_style=3, resolution=buffer_res)
            if back.is_empty:
                return (p,)
            if back.geom_type == "Polygon":
                p = back
            else:
                polys = [g for g in getattr(back, "geoms", []) if g.geom_type == "Polygon"]
                if not polys:
                    return p
                p = max(polys, key=lambda g: g.area)
        return (p,)

# endregion  Polygon Operations

# region LIST OPERATIONS


def str_to_slice(slice_str):
    # 1. Validate and clean slice string
    if not re.fullmatch(r"\s*-?\d*\s*(:\s*-?\d*\s*(:\s*-?\d*\s*)?)?", slice_str):
        raise ValueError(f"Invalid slice string: {slice_str}")
    # 2. Parse slice safely into slice object
    if ":" in slice_str:
        parts = [int(p) if p else None for p in slice_str.split(":")]
        sl = slice(*parts)
    else:
        # Single index case, e.g. [3]
        sl = int(slice_str)
        sl = slice(*[sl, sl + 1])

    return sl


class SliceListPanel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panels": (IO_Types.PANEL,),
                "_slice": ("STRING", {"default": "0:", "forceInput": False})
            },
        }

    CATEGORY = CATEGORY_PATH
    INPUT_IS_LIST = True
    RETURN_TYPES = (IO_Types.PANEL,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("Panels (Shapely Polygons)",)
    FUNCTION = "func"

    def func(self, panels, _slice):
        """
        Apply a function to elements of a list selected by a slice string (e.g. [1:5:2], [::-1]).
        Returns a new modified copy of the list.
        """
        sl = str_to_slice(_slice[0])
        result = deepcopy(panels)
        return (result[sl],)


class ListTransferPanel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_panels": (IO_Types.PANEL,),
                "from_panels": (IO_Types.PANEL,),
                "to_slice": ("STRING", {"default": "0:", "forceInput": False}),
                "from_slice": ("STRING", {"default": "0:", "forceInput": False})
            },
        }

    CATEGORY = CATEGORY_PATH
    INPUT_IS_LIST = True
    RETURN_TYPES = (IO_Types.PANEL,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("Panels (Shapely Polygons)",)
    FUNCTION = "func"

    def func(self, to_panels, from_panels, to_slice, from_slice):
        to_slice, from_slice = to_slice[0], from_slice[0]
        to_slice = str_to_slice(to_slice)
        from_slice = str_to_slice(from_slice)

        to_list = deepcopy(to_panels)
        from_list_slice = deepcopy(from_panels[from_slice])

        to_list[to_slice] = from_list_slice
        return (to_list,)


class ListAppendPanel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panels": (IO_Types.PANEL,),
                "to_append": (IO_Types.PANEL,),
            },
        }

    CATEGORY = CATEGORY_PATH
    INPUT_IS_LIST = True
    RETURN_TYPES = (IO_Types.PANEL,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("Panels (Shapely Polygons)",)
    FUNCTION = "func"

    def func(self, panels: list[Polygon], to_append):
        panels = deepcopy(panels)
        panels.extend(deepcopy(to_append))
        return (panels,)

# endregion PANEL LIST OPERATIONS

# region Other Nodes


class DrawPanelsEdges:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panels": (IO_Types.PANEL,),
                "canvas": (IO_Types.PANEL,),
                "stroke_width": ("INT", {"default": 8, "min": 1, "max": 512}),
                "color_alpha": ("INT", {"default": 255, "min": 1, "max": 255}),
                "stroke_color": ("COLOR", {"default": "#000000"}),  # requires bmad or mtb nodes
                "upscale": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip":
                    "The lines are drawn upscaled by this factor to anti-alias the jaggies away."}),
                #"pad": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "Safety margin. Use this when some "
                #    "edge falls outside of the provided canvas."}),
                # TODO is pad really needed? may need to review the code...
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "func"
    INPUT_IS_LIST = True
    CATEGORY = CATEGORY_PATH

    def func(self, panels, canvas, stroke_width, stroke_color, color_alpha, upscale):
        canvas = canvas[0]
        stroke_width = stroke_width[0]
        stroke_color = stroke_color[0]
        color_alpha = color_alpha[0]
        upscale = upscale[0]

        # prepare Color input
        if isinstance(stroke_color, str):
            stroke_color = int(stroke_color.lstrip("#"), 16)
        if isinstance(stroke_color, int):
            color = ((stroke_color & 0xFF0000) >> 16,
                     (stroke_color & 0x00FF00) >> 8,
                     (stroke_color & 0x0000FF),
                     color_alpha)
        else:  # suppose the following without checking -> isinstance(stroke_color, tuple) and len(x) == 3
            color = (stroke_color[0], stroke_color[1], stroke_color[2], color_alpha)

        img = draw_polygon_contours(panels, canvas, stroke_color=color, stroke_width=stroke_width,
                                    pad=0, upscale=upscale)

        i = node_helpers.pillow(ImageOps.exif_transpose, img)
        if i.mode == 'I':
            i = i.point(lambda p: p * (1 / 255))
        image = i

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)

        return (image, mask)


class PolygonUnwrappedBounds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": (IO_Types.PANEL,),
            }
        }

    # Four separate integer outputs
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("min_x", "min_y", "max_x", "max_y")
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH
    DESCRIPTION = "Unwrapped polygon.bounds with rounded values. For potential use with other node packages."

    def func(self, polygon):
        minx, miny, maxx, maxy = polygon.bounds
        return (
            round(minx),
            round(miny),
            round(maxx),
            round(maxy),
        )


class PolygonBounds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": (IO_Types.PANEL,),
            }
        }

    RETURN_TYPES = (IO_Types.BBOX,)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH
    DESCRIPTION = "polygon.bounds"

    def func(self, polygon):
        bbox = polygon.bounds
        return (bbox,)


class BBoxFromInts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xmin": ("INT", {"default": 0}),
                "ymin": ("INT", {"default": 0}),
                "xmax": ("INT", {"default": 64}),
                "ymax": ("INT", {"default": 64}),
            }
        }

    RETURN_TYPES = (IO_Types.BBOX,)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH

    def func(self, xmin, ymin, xmax, ymax):
        return ((float(xmin), float(ymin), float(xmax), float(ymax)),)


class PasteCrops:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),       # (1, H, W, C), float32 [0,1]
                "cropped_images": ("IMAGE",),   # list of crops (1, h, w, C)
                "masks": ("MASK",),             # list of masks (1, h, w) or (1, h, w, 1)
                "bboxes": ("BBOX",),            # list of (x0, y0, x1, y1)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH
    DESCRIPTION = \
        ("Pastes the cropped_images into the base_image in the area defined by the corresponding bboxes.\n"
         "If an image (or mask) does not match its bbox size, it is resized to fit.\n"
         "To avoid quality loss keep the image-mask-bbox pairs with the same dimensions.")

    INPUT_IS_LIST = True

    def func(self, base_image: torch.Tensor, cropped_images, masks, bboxes):
        base = base_image[0].clone()  # (1, H, W, C)

        for crop, mask, bbox in zip(cropped_images, masks, bboxes):
            x0, y0, x1, y1 = unwrap_bbox_as_ints(bbox, base)
            w, h = x1 - x0, y1 - y0

            # Ensure mask has channel dimension
            if mask.ndim == 3:  # (1, H, W)
                mask = mask.unsqueeze(-1)  # (1, H, W, 1)

            # Resize or skip if exact match
            crop_h, crop_w = crop.shape[1:3]
            if (crop_h, crop_w) == (h, w):
                crop_resized = crop
            else:
                crop_resized = F.interpolate(
                    crop.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)

            # Same as previous step for the mask
            mask_h, mask_w = mask.shape[1:3]
            if (mask_h, mask_w) == (h, w):
                mask_resized = mask
            else:
                mask_resized = F.interpolate(
                    mask.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)

            # Blend into base
            region = base[:, y0:y1, x0:x1, :]  # (1, h, w, C)
            blended = region * (1 - mask_resized) + crop_resized * mask_resized
            base[:, y0:y1, x0:x1, :] = blended

        return (base,)


class PolygonToResizedMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polygon": ("POLYGON",),
                "approx_res": (["262144", "1048576", "1638400"], {"default": "1048576"}),
                "pad": (["16", "32", "64", "128"], {"default": "64"}),
            },
            "optional": {
                "image": ("IMAGE",),  # optional image to crop + resize
            }
        }

    RETURN_TYPES = ("MASK", "BBOX", "IMAGE")
    # RETURN_NAMES = ("mask", "bbox", "image")
    FUNCTION = "func"
    CATEGORY = CATEGORY_PATH

    def func(self, polygon, approx_res, pad, image=None):
        approx_res = int(approx_res)
        pad = int(pad)

        # 1. Get bounds and round outward
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny = math.floor(minx), math.floor(miny)
        maxx, maxy = math.ceil(maxx), math.ceil(maxy)

        width = maxx - minx
        height = maxy - miny

        if width <= 0 or height <= 0:
            empty_mask = torch.zeros((1, pad, pad), dtype=torch.float32)
            empty_img = None if image is None else torch.zeros((1, pad, pad, image.shape[-1]), dtype=image.dtype)
            return (empty_mask, (0, 0, 0, 0), empty_img)

        # 2. Rasterize polygon mask
        mask_img = Image.new("L", (width, height), 0)
        shifted_poly = translate(polygon, xoff=-minx, yoff=-miny)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(list(shifted_poly.exterior.coords), outline=1, fill=1)

        mask_np = np.array(mask_img, dtype=np.float32)  # HxW in {0,1}
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # [1,H,W]

        # 3. Determine scaling factor from largest dimension
        H, W = mask_np.shape
        largest_dim = max(H, W)

        ideal_scale = math.sqrt(approx_res / (H * W))
        scaled_largest = int(largest_dim * ideal_scale)

        snapped_largest = max(pad, (scaled_largest // pad) * pad)
        scale = snapped_largest / largest_dim

        new_H = max(1, int(H * scale))
        new_W = max(1, int(W * scale))

        # 4. Resize mask
        mask_resized = F.interpolate(
            mask_t.unsqueeze(1), size=(new_H, new_W), mode="bilinear", align_corners=False
        ).squeeze(1)  # [1,H,W]

        # 5. Pad smaller dimension to multiple of pad (centered)
        pad_H = math.ceil(new_H / pad) * pad
        pad_W = math.ceil(new_W / pad) * pad

        pad_top = (pad_H - new_H) // 2
        pad_bottom = pad_H - new_H - pad_top
        pad_left = (pad_W - new_W) // 2
        pad_right = pad_W - new_W - pad_left

        mask_padded = F.pad(mask_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        # 6. Crop + resize + pad image if provided
        img_out = None
        if image is not None:
            # crop the image to polygon bounds
            img_crop = image[:, miny:maxy, minx:maxx, :]  # [B,h,w,C]

            # resize with same factor
            img_resized = F.interpolate(
                img_crop.permute(0, 3, 1, 2), size=(new_H, new_W), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)  # [B,new_H,new_W,C]

            # pad to match mask
            img_out = F.pad(img_resized, (0, 0, pad_left, pad_right, pad_top, pad_bottom))

        # 7. Recompute bbox on nonzero mask
        nz = (mask_padded[0] > 0.5).nonzero(as_tuple=False)
        if nz.shape[0] > 0:
            ymin, xmin = nz.min(dim=0)[0].tolist()
            ymax, xmax = nz.max(dim=0)[0].tolist()
            bbox = (float(xmin), float(ymin), float(xmax), float(ymax))
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)

        return (mask_padded, bbox, img_out)


class UnpackBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": (IO_Types.BBOX,),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    FUNCTION = "apply"
    CATEGORY = CATEGORY_PATH

    def apply(self, bbox):
        xmin, ymin, xmax, ymax = unwrap_bbox_as_ints(bbox)
        return (xmin, ymin, xmax, ymax)


class CropMaskByBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "bbox": (IO_Types.BBOX,),   # (xmin, ymin, xmax, ymax)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY_PATH

    def apply(self, mask, bbox):
        xmin, ymin, xmax, ymax = unwrap_bbox_as_ints(bbox, mask)
        cropped = mask[:, ymin:ymax, xmin:xmax].clone()
        return (cropped,)


class CropImageByBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": (IO_Types.BBOX,),   # (xmin, ymin, xmax, ymax)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY_PATH

    def apply(self, image, bbox):
        xmin, ymin, xmax, ymax = unwrap_bbox_as_ints(bbox, image)
        cropped = image[:, ymin:ymax, xmin:xmax, :].clone()
        return (cropped,)


# endregion Other Nodes


NODE_CLASS_MAPPINGS = {
    "bmad_CanvasPanel": CanvasPanel,
    "bmad_LoadPanelLayout": LoadPanelLayout,
    "bmad_SavePanelLayout": SavePanelLayout,
    "bmad_StringDecodePanelLayout": StringDecodePanelLayout,
    "bmad_StringEncodePanelLayout": StringEncodePanelLayout,
    "bmad_PreviewPanelLayout": PreviewPanelLayout,
    "bmad_PreviewPanels": PreviewPanels,

    "bmad_OffsetPolygonBounds": OffsetPolygonBounds,
    "bmad_BBoxSnap": BBoxSnapNode,

    "bmad_Panel2Mask": Panel2Mask,
    "bmad_DrawPanelsEdges": DrawPanelsEdges,
    "bmad_RotatePolygon": RotatePolygon,
    "bmad_ScalePolygon": ScalePolygon,
    "bmad_TranslatePolygon": TranslatePolygon,
    "bmad_BevelPolygon": BevelPolygon,

    "bmad_BuildLayoutPanels": BuildLayoutPanels,
    "bmad_RandomPanelLayoutGenerator": RandomPanelLayoutGenerator,
    "bmad_GridPanelLayoutGenerator": GridPanelLayoutGenerator,
    "bmad_MutatePanelLayout": MutatePanelLayout,

    "bmad_PolygonBounds": PolygonBounds,
    "bmad_PolygonUnwrappedBounds": PolygonUnwrappedBounds,
    "bmad_PasteCrops": PasteCrops,
    "bmad_BBoxFromInts": BBoxFromInts,
    "bmad_UnpackBBox": UnpackBBox,
    "bmad_CropMaskByBBox": CropMaskByBBox,

    "bmad_CropImageByBBox": CropImageByBBox,

    "bmad_SliceList_Panels": SliceListPanel,
    "bmad_ListTransferPanel": ListTransferPanel,
    "bmad_ListAppendPanel": ListAppendPanel,

    "bmad_PolygonToResizedMask": PolygonToResizedMask,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "bmad_CanvasPanel": "Canvas Panel",

    "bmad_LoadPanelLayout": "Load Panel Layout",
    "bmad_SavePanelLayout": "Save Panel Layout",
    "bmad_StringDecodePanelLayout": "String Decode Panel Layout",
    "bmad_StringEncodePanelLayout": "String Encode Panel Layout",
    "bmad_PreviewPanelLayout": "Preview Panel Layout",
    "bmad_PreviewPanels": "Preview Panels",

    "bmad_OffsetPolygonBounds": "Offset Polygon Bounds",
    "bmad_BBoxSnap": "BBoxSnap",

    "bmad_Panel2Mask": "Panel to Mask",
    "bmad_DrawPanelsEdges": "Draw Panels Edges",
    "bmad_RotatePolygon": "Rotate Polygon",
    "bmad_ScalePolygon": "Scale Polygon",
    "bmad_TranslatePolygon": "Translate Polygon",
    "bmad_BevelPolygon": "Bevel Polygon",

    "bmad_BuildLayoutPanels": "Build Layout Panels",
    "bmad_RandomPanelLayoutGenerator": "Random Panel Layout Generator",
    "bmad_GridPanelLayoutGenerator": "Grid Panel Layout Generator",
    "bmad_MutatePanelLayout": "Mutate Panel Layout",

    "bmad_PolygonBounds": "Polygon.bounds",
    "bmad_PolygonUnwrappedBounds": "Polygon.bounds (unwrapped)",
    "bmad_PasteCrops": "Paste Crops with Masks",
    "bmad_BBoxFromInts": "BBox from Ints",
    "bmad_UnpackBBox": "Unpack BBox",
    "bmad_CropMaskByBBox": "Crop Mask By BBox",

    "bmad_CropImageByBBox": "Crop Image By BBox",

    "bmad_SliceList_Panels": "Slice Panels List",
    "bmad_ListTransferPanel": "List Transfer Panels",
    "bmad_ListAppendPanel": "List Append Panels",

    "bmad_PolygonToResizedMask": "Polygon To Resized Mask",
}
