"""
FLOORPLAN FACTORY
--------------

This module takes in an environment and generates a randomised 2D floor plan of the environment. 

The factory will be using the "renovation" python package to generate the floor plan. 

We will start by a "simpler" implementation of the factory, where we keep 2 separate visual representations 
of the environment: one for the antenna/obstacle placement and one for the floor plan.

Then we will overlay the two representations to generate the final floor plan with the antenna/obstacle placements.
This is very doable as long as we keep the same scale for both representations, and we can easily convert the coordinates from one representation to the other.

For the purpose of this project we just care about generating the necessary details to train the ML model. 

Hence, we will keep a dict referencing the "occupied" and "unoccupied" areas of the environment, 
and we will use this dict to generate the obstacles in the available free space.

NOTE: this is the best idea I have come up with at the moment, there might be a more optimal way of doing this.

INPUTS & OUTPUTS
--------------  

INPUTS:
    The renovation package needs a yml file determing quite a few parameters to generate the floor plan.
        WALLS:
            - type: wall
            - anchor_point: [-0.2, 6]
            - length: 5.58
            - thickness: 0.2
            - orientation_angle: 270

        DOORS:
            - type: door
            - anchor_point: [2.045, 1.985]
            - doorway_width: 0.9
            - door_width: 0.8
            - thickness: 0.05
            - color: gray
        
        WINDOWS:
            - type: window
            - anchor_point: [0.61, 6]
            - length: 1.65
            - overall_thickness: 0.45
            - single_line_thickness: 0.05

            
OUTPUTS:
    - Optionally, a png image including the 2D representation of the floorplan.
    - A parquet file including the coordinates of the occupied and unoccupied areas of the environment.
        - Classify areas based on obstacle types: walls, doors, and windows. (this is just an idea not sure if its necessary or useful)
        - After we have defined the free space available we can generate humans and other obstacles in the environment.
        - This parquet file will be used to train the ML model to predict the location of a device based on the floor plan and the presence of obstacles.


              
PSEUDOCODE (brainstorming)
--------------------------

1. Retrieve environment details.

2. (SKIP FOR NOW -> nice to have) Randomly determine the type of floor plan we want to generate (e.g. open-area/outdoor || closed-area/indoor plan).
    - outdoor: less rooms per total area, more open space.
    - indoor: more rooms per total area, less open space.

3. Randomly generate the floor plan using the renovation package and the environment details.
    - To keep things completely random and fair, we will have a "patio/open space" element.
      Which if randomly chosen becomes a "no wall zone". So walls can be built on the perimeter of the area but not inside it. 
      Size of this "no-wall-zone" is completely random. (this can give us the "creation" of outdoor and indoor env without defining/justifying 
      specific "max" or "min" occupancy percentages and allowing for randomness to take charge)
    - Create a function to make "rooms" given the available current free-space.
    - Room creation can take a "tree-node-branching" approach, where rooms are created (breath-first || depth-first) right next to that last room that was created. 
    - Room size can be randomly determined based on the total area of the environment, and a min and max room size.
    - Create a function to make "doors" and "windows".
    - Every room will have 1 door and 1 window.
    - Only place windows on the external walls of the floor plan.
    - Only place doors on the internal walls of the floor plan.

4. Output png images and free-space vs occupied areas in the env.


@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import math
import os
import random
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import yaml
from environment_factory import Environment


EPSILON = 1e-6
METERS_PER_INCH = 0.0254
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "floor_planning" / "generated"


@dataclass(frozen=True)
class Rect:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def interval_for_side(self, side: str) -> Tuple[float, float]:
        if side in {"north", "south"}:
            return (self.x_min, self.x_max)
        return (self.y_min, self.y_max)

    def edge_coordinate(self, side: str) -> float:
        if side == "north":
            return self.y_max
        if side == "south":
            return self.y_min
        if side == "east":
            return self.x_max
        if side == "west":
            return self.x_min
        raise ValueError(f"Unsupported side: {side}")

    def touches_building_side(self, building: "Rect", side: str) -> bool:
        return math.isclose(self.edge_coordinate(side), building.edge_coordinate(side), abs_tol=EPSILON)

    def expanded(self, padding: float) -> "Rect":
        return Rect(
            x_min=self.x_min - padding,
            y_min=self.y_min - padding,
            x_max=self.x_max + padding,
            y_max=self.y_max + padding,
        )


@dataclass(frozen=True)
class FloorPlanConfig:
    env_width: float
    env_height: float
    env_type: str = "indoor"
    random_seed: Optional[int] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    yaml_filename: str = "floor_plan.yml"
    parquet_filename: str = "floor_plan_elements.parquet"
    png_dirname: str = "png"
    png_filename: str = "floor_plan.png"
    render_png: bool = False
    dpi: int = 180
    scale_numerator: int = 1
    scale_denominator: int = 220
    min_scale_denominator: int = 10
    max_scale_denominator: int = 400
    building_coverage_range_indoor: Tuple[float, float] = (0.75, 0.95)
    building_coverage_range_outdoor: Tuple[float, float] = (0.25, 0.55)
    corridor_probability_indoor: float = 0.9
    corridor_probability_outdoor: float = 0.35
    patio_probability_indoor: float = 0.18
    patio_probability_outdoor: float = 0.4
    max_split_depth: int = 7
    min_block_width: float = 2.4
    min_block_height: float = 2.4
    min_room_area: float = 6.0
    max_room_area: float = 40.0
    small_room_max_area: float = 12.0
    medium_room_max_area: float = 24.0
    corridor_min_width: float = 2.0
    corridor_max_width: float = 3.5
    corridor_min_aspect_ratio: float = 2.5
    wall_thickness_external: float = 0.24
    wall_thickness_internal: float = 0.12
    doorway_width: float = 1.0
    door_width: float = 0.9
    door_thickness: float = 0.08
    window_min_length: float = 0.9
    window_max_length: float = 1.8
    window_overall_thickness: float = 0.24
    window_single_line_thickness: float = 0.05
    opening_margin: float = 0.2
    opening_gap: float = 0.2
    layout_padding: float = 1.0
    floor_plan_padding_ratio: float = 0.01
    floor_plan_target_pixels_per_meter: float = 36.0
    floor_plan_min_long_edge_pixels: int = 6000
    floor_plan_max_long_edge_pixels: int = 10000
    patio_fill_color: str = "#d9d9d9"
    patio_line_width: float = 0.05

    @property
    def png_dir(self) -> Path:
        return self.output_dir / self.png_dirname

    @property
    def yaml_path(self) -> Path:
        return self.output_dir / self.yaml_filename

    @property
    def parquet_path(self) -> Path:
        return self.output_dir / self.parquet_filename

    @property
    def title(self) -> str:
        return f"{self.env_type.title()} Floor Plan"

    @classmethod
    def from_environment(
        cls,
        env: object,
        *,
        env_type: Optional[str] = None,
        output_dir: Optional[Path] = None,
        random_seed: Optional[int] = None,
        artifact_stem: Optional[str] = None,
        render_png: bool = False,
    ) -> "FloorPlanConfig":
        if isinstance(env, dict):
            width = env["width"]
            height = env["height"]
            resolved_env_type = env_type or env.get("env_type")
        else:
            width = getattr(env, "width")
            height = getattr(env, "height")
            resolved_env_type = env_type or getattr(env, "env_type", None)

        if resolved_env_type is None:
            raise ValueError("Environment must provide env_type, or env_type must be passed explicitly.")

        base_output = output_dir or DEFAULT_OUTPUT_DIR
        config_kwargs: Dict[str, object] = {}
        if artifact_stem:
            config_kwargs.update(
                {
                    "yaml_filename": f"{artifact_stem}.yml",
                    "parquet_filename": f"{artifact_stem}_elements.parquet",
                    "png_filename": f"{artifact_stem}.png",
                }
            )
        return cls(
            env_width=width,
            env_height=height,
            env_type=resolved_env_type,
            output_dir=base_output,
            random_seed=random_seed,
            render_png=render_png,
            **config_kwargs,
        )


@dataclass(frozen=True)
class Block:
    block_id: str
    rect: Rect
    parent_id: Optional[str] = None


@dataclass(frozen=True)
class Room:
    room_id: str
    room_type: str
    rect: Rect


@dataclass(frozen=True)
class Patio:
    patio_id: str
    rect: Rect
    attached_mode: str


@dataclass
class OpeningSpec:
    opening_type: str
    desired_width: float
    door_width: Optional[float] = None
    is_entrance: bool = False


@dataclass
class BoundarySegment:
    segment_id: str
    room_id: str
    side: str
    x0: float
    y0: float
    x1: float
    y1: float
    boundary_type: str
    adjacent_room_id: Optional[str] = None
    patio_id: Optional[str] = None
    openings: List[OpeningSpec] = field(default_factory=list)

    @property
    def is_horizontal(self) -> bool:
        return math.isclose(self.y0, self.y1, abs_tol=EPSILON)

    @property
    def length(self) -> float:
        return abs(self.x1 - self.x0) if self.is_horizontal else abs(self.y1 - self.y0)

    @property
    def is_external(self) -> bool:
        return self.boundary_type == "exterior"

    @property
    def touches_patio(self) -> bool:
        return self.boundary_type == "patio"


@dataclass
class FloorElement:
    element_id: str
    element_type: str
    room_id: str
    x: float
    y: float
    orientation_angle: float
    is_external: bool
    touches_patio: bool
    adjacent_room_id: Optional[str] = None
    patio_id: Optional[str] = None
    is_entrance: bool = False
    length: Optional[float] = None
    thickness: Optional[float] = None
    doorway_width: Optional[float] = None
    door_width: Optional[float] = None
    overall_thickness: Optional[float] = None
    single_line_thickness: Optional[float] = None
    to_the_right: Optional[bool] = None

    def to_renovation_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "type": self.element_type,
            "anchor_point": [round(self.x, 3), round(self.y, 3)],
            "orientation_angle": self.orientation_angle,
        }
        if self.element_type == "wall":
            payload["length"] = round(self.length or 0.0, 3)
            payload["thickness"] = round(self.thickness or 0.0, 3)
        elif self.element_type == "door":
            payload["doorway_width"] = round(self.doorway_width or 0.0, 3)
            payload["door_width"] = round(self.door_width or 0.0, 3)
            payload["thickness"] = round(self.thickness or 0.0, 3)
            if self.to_the_right:
                payload["to_the_right"] = True
        elif self.element_type == "window":
            payload["length"] = round(self.length or 0.0, 3)
            payload["overall_thickness"] = round(self.overall_thickness or 0.0, 3)
            payload["single_line_thickness"] = round(self.single_line_thickness or 0.0, 3)
        return payload


@dataclass
class GeneratedFloorPlan:
    config: FloorPlanConfig
    environment_rect: Rect
    building_rect: Rect
    rooms: List[Room]
    patios: List[Patio]
    blocks: List[Block]
    boundaries: List[BoundarySegment]
    elements: List[FloorElement]
    yaml_path: Path
    png_paths: List[Path]
    parquet_path: Path


def rect_overlap(a0: float, a1: float, b0: float, b1: float) -> Optional[Tuple[float, float]]:
    start = max(a0, b0)
    end = min(a1, b1)
    if end - start <= EPSILON:
        return None
    return (start, end)


def interval_length(interval: Tuple[float, float]) -> float:
    return interval[1] - interval[0]


def normalize_interval(interval: Tuple[float, float]) -> Tuple[float, float]:
    return (min(interval[0], interval[1]), max(interval[0], interval[1]))


def next_id(counter: Dict[str, int], prefix: str) -> str:
    counter[prefix] = counter.get(prefix, 0) + 1
    return f"{prefix}_{counter[prefix]:04d}"


class FloorPlanFactory:
    def __init__(self, config: FloorPlanConfig) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed)
        self._id_counter: Dict[str, int] = {}

    def generate(self) -> GeneratedFloorPlan:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.render_png:
            self.config.png_dir.mkdir(parents=True, exist_ok=True)

        environment_rect = self._build_environment_rect()
        building_rect = self._sample_building_rect(environment_rect)
        corridor_block, base_blocks = self._maybe_create_corridor(building_rect)
        subdivided: List[Block] = []
        for block in base_blocks:
            subdivided.extend(self._subdivide_block(block, depth=0))

        leaf_blocks = [block for block in subdivided if block.rect.area >= self.config.min_room_area - EPSILON]
        if corridor_block is not None:
            leaf_blocks.append(corridor_block)
        if not leaf_blocks:
            raise RuntimeError("Unable to generate any floor-plan blocks.")

        rooms, patios = self._assign_block_types(leaf_blocks, building_rect, corridor_block)
        boundaries = self._build_boundaries(rooms, patios, building_rect)
        self._assign_openings(boundaries, rooms)
        elements = self._build_elements(boundaries)

        yaml_path = self._write_yaml(environment_rect, building_rect, patios, elements)
        png_paths: List[Path] = []
        if self.config.render_png:
            png_paths = self._render_yaml(yaml_path)
        parquet_path = self._write_parquet(elements)

        return GeneratedFloorPlan(
            config=self.config,
            environment_rect=environment_rect,
            building_rect=building_rect,
            rooms=rooms,
            patios=patios,
            blocks=leaf_blocks,
            boundaries=boundaries,
            elements=elements,
            yaml_path=yaml_path,
            png_paths=png_paths,
            parquet_path=parquet_path,
        )

    def _build_environment_rect(self) -> Rect:
        return Rect(
            x_min=-self.config.env_width / 2.0,
            y_min=-self.config.env_height / 2.0,
            x_max=self.config.env_width / 2.0,
            y_max=self.config.env_height / 2.0,
        )

    def _sample_building_rect(self, environment_rect: Rect) -> Rect:
        if self.config.env_type == "outdoor":
            coverage_range = self.config.building_coverage_range_outdoor
        else:
            coverage_range = self.config.building_coverage_range_indoor

        width_ratio = self.rng.uniform(*coverage_range)
        height_ratio = self.rng.uniform(*coverage_range)
        width = environment_rect.width * width_ratio
        height = environment_rect.height * height_ratio
        return Rect(
            x_min=-width / 2.0,
            y_min=-height / 2.0,
            x_max=width / 2.0,
            y_max=height / 2.0,
        )

    def _maybe_create_corridor(self, building_rect: Rect) -> Tuple[Optional[Block], List[Block]]:
        probability = (
            self.config.corridor_probability_outdoor
            if self.config.env_type == "outdoor"
            else self.config.corridor_probability_indoor
        )
        if self.rng.random() > probability:
            return None, [Block(block_id=next_id(self._id_counter, "block"), rect=building_rect)]

        is_horizontal = building_rect.width >= building_rect.height
        corridor_width = self.rng.uniform(self.config.corridor_min_width, self.config.corridor_max_width)
        blocks: List[Block] = []
        corridor_rect: Rect

        if is_horizontal:
            y_mid = (building_rect.y_min + building_rect.y_max) / 2.0
            corridor_rect = Rect(
                x_min=building_rect.x_min,
                y_min=y_mid - corridor_width / 2.0,
                x_max=building_rect.x_max,
                y_max=y_mid + corridor_width / 2.0,
            )
            if corridor_rect.y_min - building_rect.y_min >= self.config.min_block_height:
                blocks.append(
                    Block(
                        block_id=next_id(self._id_counter, "block"),
                        rect=Rect(
                            building_rect.x_min,
                            building_rect.y_min,
                            building_rect.x_max,
                            corridor_rect.y_min,
                        ),
                    )
                )
            if building_rect.y_max - corridor_rect.y_max >= self.config.min_block_height:
                blocks.append(
                    Block(
                        block_id=next_id(self._id_counter, "block"),
                        rect=Rect(
                            building_rect.x_min,
                            corridor_rect.y_max,
                            building_rect.x_max,
                            building_rect.y_max,
                        ),
                    )
                )
        else:
            x_mid = (building_rect.x_min + building_rect.x_max) / 2.0
            corridor_rect = Rect(
                x_min=x_mid - corridor_width / 2.0,
                y_min=building_rect.y_min,
                x_max=x_mid + corridor_width / 2.0,
                y_max=building_rect.y_max,
            )
            if corridor_rect.x_min - building_rect.x_min >= self.config.min_block_width:
                blocks.append(
                    Block(
                        block_id=next_id(self._id_counter, "block"),
                        rect=Rect(
                            building_rect.x_min,
                            building_rect.y_min,
                            corridor_rect.x_min,
                            building_rect.y_max,
                        ),
                    )
                )
            if building_rect.x_max - corridor_rect.x_max >= self.config.min_block_width:
                blocks.append(
                    Block(
                        block_id=next_id(self._id_counter, "block"),
                        rect=Rect(
                            corridor_rect.x_max,
                            building_rect.y_min,
                            building_rect.x_max,
                            building_rect.y_max,
                        ),
                    )
                )

        corridor_block = Block(block_id=next_id(self._id_counter, "block"), rect=corridor_rect)
        return corridor_block, blocks if blocks else [Block(block_id=next_id(self._id_counter, "block"), rect=building_rect)]

    def _subdivide_block(self, block: Block, depth: int) -> List[Block]:
        rect = block.rect
        if depth >= self.config.max_split_depth:
            return [block]
        if rect.area <= self.config.max_room_area and (
            rect.width <= self.config.max_room_area or rect.height <= self.config.max_room_area
        ):
            if rect.area <= self.config.max_room_area or self.rng.random() < 0.3:
                return [block]

        split_orientation = self._choose_split_orientation(rect)
        child_rects = self._split_rect(rect, split_orientation)
        if child_rects is None:
            other = "horizontal" if split_orientation == "vertical" else "vertical"
            child_rects = self._split_rect(rect, other)
        if child_rects is None:
            return [block]

        children = [
            Block(block_id=next_id(self._id_counter, "block"), rect=child_rects[0], parent_id=block.block_id),
            Block(block_id=next_id(self._id_counter, "block"), rect=child_rects[1], parent_id=block.block_id),
        ]
        results: List[Block] = []
        for child in children:
            results.extend(self._subdivide_block(child, depth + 1))
        return results

    def _choose_split_orientation(self, rect: Rect) -> str:
        width_ratio = rect.width / max(rect.height, EPSILON)
        height_ratio = rect.height / max(rect.width, EPSILON)
        if width_ratio >= 1.25:
            return "vertical"
        if height_ratio >= 1.25:
            return "horizontal"
        return self.rng.choice(["horizontal", "vertical"])

    def _split_rect(self, rect: Rect, orientation: str) -> Optional[Tuple[Rect, Rect]]:
        if orientation == "vertical":
            lower = rect.x_min + self.config.min_block_width
            upper = rect.x_max - self.config.min_block_width
            if upper - lower <= EPSILON:
                return None
            split_x = self.rng.uniform(
                max(lower, rect.x_min + rect.width * 0.35),
                min(upper, rect.x_min + rect.width * 0.65),
            )
            return (
                Rect(rect.x_min, rect.y_min, split_x, rect.y_max),
                Rect(split_x, rect.y_min, rect.x_max, rect.y_max),
            )

        lower = rect.y_min + self.config.min_block_height
        upper = rect.y_max - self.config.min_block_height
        if upper - lower <= EPSILON:
            return None
        split_y = self.rng.uniform(
            max(lower, rect.y_min + rect.height * 0.35),
            min(upper, rect.y_min + rect.height * 0.65),
        )
        return (
            Rect(rect.x_min, rect.y_min, rect.x_max, split_y),
            Rect(rect.x_min, split_y, rect.x_max, rect.y_max),
        )

    def _assign_block_types(
        self,
        blocks: List[Block],
        building_rect: Rect,
        corridor_block: Optional[Block],
    ) -> Tuple[List[Room], List[Patio]]:
        patio_probability = (
            self.config.patio_probability_outdoor
            if self.config.env_type == "outdoor"
            else self.config.patio_probability_indoor
        )

        corridor_room: Optional[Room] = None
        regular_blocks: List[Block] = []
        for block in blocks:
            if corridor_block is not None and block.block_id == corridor_block.block_id:
                corridor_room = Room(room_id=next_id(self._id_counter, "room"), room_type="corridor", rect=block.rect)
            else:
                regular_blocks.append(block)

        patios: List[Patio] = []
        room_blocks = list(regular_blocks)
        if regular_blocks:
            max_patios = max(1, math.floor(len(regular_blocks) * patio_probability))
            candidate_pool = regular_blocks[:]
            self.rng.shuffle(candidate_pool)
            for candidate in candidate_pool:
                if len(patios) >= max_patios:
                    break
                if self.rng.random() > patio_probability:
                    continue

                patio_mode = self.rng.choice(["internal", "edge_attached"])
                is_edge = self._rect_touches_building_perimeter(candidate.rect, building_rect)
                if patio_mode == "internal" and is_edge:
                    continue
                if patio_mode == "edge_attached" and not is_edge:
                    continue

                prospective = [block for block in room_blocks if block.block_id != candidate.block_id]
                if corridor_room is None and not prospective:
                    continue

                room_rects = [block.rect for block in prospective]
                if corridor_room is not None:
                    room_rects.append(corridor_room.rect)
                if not self._rects_are_connected(room_rects):
                    continue

                patios.append(
                    Patio(
                        patio_id=next_id(self._id_counter, "patio"),
                        rect=candidate.rect,
                        attached_mode=patio_mode,
                    )
                )
                room_blocks = [block for block in room_blocks if block.block_id != candidate.block_id]

        rooms: List[Room] = []
        if corridor_room is not None:
            rooms.append(corridor_room)

        for block in room_blocks:
            rooms.append(
                Room(
                    room_id=next_id(self._id_counter, "room"),
                    room_type=self._classify_room_type(block.rect),
                    rect=block.rect,
                )
            )

        if not rooms:
            raise RuntimeError("Generated layout has no rooms after patio assignment.")
        if not self._rects_are_connected([room.rect for room in rooms]):
            raise RuntimeError("Generated room graph is disconnected.")

        return rooms, patios

    def _classify_room_type(self, rect: Rect) -> str:
        aspect_ratio = max(rect.width, rect.height) / max(min(rect.width, rect.height), EPSILON)
        if aspect_ratio >= self.config.corridor_min_aspect_ratio and self.rng.random() < 0.25:
            return "corridor"
        if rect.area <= self.config.small_room_max_area:
            return "small"
        if rect.area <= self.config.medium_room_max_area:
            return "medium"
        return "large"

    def _rect_touches_building_perimeter(self, rect: Rect, building_rect: Rect) -> bool:
        return any(
            rect.touches_building_side(building_rect, side)
            for side in ("north", "south", "east", "west")
        )

    def _rects_are_connected(self, rects: Sequence[Rect]) -> bool:
        if not rects:
            return False
        graph = {idx: set() for idx in range(len(rects))}
        for i, left in enumerate(rects):
            for j in range(i + 1, len(rects)):
                if self._rects_share_edge(left, rects[j]):
                    graph[i].add(j)
                    graph[j].add(i)
        queue = deque([0])
        visited = {0}
        while queue:
            node = queue.popleft()
            for nxt in graph[node]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return len(visited) == len(rects)

    def _rects_share_edge(self, first: Rect, second: Rect) -> bool:
        vertical_touch = (
            math.isclose(first.x_max, second.x_min, abs_tol=EPSILON)
            or math.isclose(first.x_min, second.x_max, abs_tol=EPSILON)
        ) and rect_overlap(first.y_min, first.y_max, second.y_min, second.y_max)
        horizontal_touch = (
            math.isclose(first.y_max, second.y_min, abs_tol=EPSILON)
            or math.isclose(first.y_min, second.y_max, abs_tol=EPSILON)
        ) and rect_overlap(first.x_min, first.x_max, second.x_min, second.x_max)
        return bool(vertical_touch or horizontal_touch)

    def _build_boundaries(
        self,
        rooms: Sequence[Room],
        patios: Sequence[Patio],
        building_rect: Rect,
    ) -> List[BoundarySegment]:
        boundaries: List[BoundarySegment] = []
        boundary_keys: set[Tuple[object, ...]] = set()
        room_map = {room.room_id: room for room in rooms}
        patio_map = {patio.patio_id: patio for patio in patios}

        for room in rooms:
            for side in ("north", "south", "east", "west"):
                labeled_neighbors = self._neighbors_for_side(room.rect, side, rooms, patios, room.room_id)
                for start, end, kind, adjacent_id in labeled_neighbors:
                    if interval_length((start, end)) <= EPSILON:
                        continue
                    if kind == "room" and room.room_id > str(adjacent_id):
                        continue
                    x0, y0, x1, y1 = self._side_interval_to_segment(room.rect, side, start, end)
                    key = (
                        room.room_id if kind != "room" else min(room.room_id, str(adjacent_id)),
                        None if kind != "room" else max(room.room_id, str(adjacent_id)),
                        round(x0, 6),
                        round(y0, 6),
                        round(x1, 6),
                        round(y1, 6),
                        kind,
                    )
                    if key in boundary_keys:
                        continue
                    boundary_keys.add(key)
                    boundaries.append(
                        BoundarySegment(
                            segment_id=next_id(self._id_counter, "segment"),
                            room_id=room.room_id,
                            side=side,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            boundary_type="internal" if kind == "room" else kind,
                            adjacent_room_id=str(adjacent_id) if kind == "room" else None,
                            patio_id=str(adjacent_id) if kind == "patio" else None,
                        )
                    )

                if room.rect.touches_building_side(building_rect, side):
                    full_interval = room.rect.interval_for_side(side)
                    uncovered = self._interval_difference(
                        normalize_interval(full_interval),
                        [(start, end) for start, end, _, _ in labeled_neighbors],
                    )
                    for start, end in uncovered:
                        if interval_length((start, end)) <= EPSILON:
                            continue
                        x0, y0, x1, y1 = self._side_interval_to_segment(room.rect, side, start, end)
                        boundaries.append(
                            BoundarySegment(
                                segment_id=next_id(self._id_counter, "segment"),
                                room_id=room.room_id,
                                side=side,
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                                boundary_type="exterior",
                            )
                        )

        if not boundaries:
            raise RuntimeError("No structural boundaries were generated.")

        # Keep lints quiet for now; these maps are useful when extending the module.
        _ = room_map, patio_map
        return boundaries

    def _neighbors_for_side(
        self,
        rect: Rect,
        side: str,
        rooms: Sequence[Room],
        patios: Sequence[Patio],
        self_room_id: str,
    ) -> List[Tuple[float, float, str, str]]:
        labeled: List[Tuple[float, float, str, str]] = []

        for room in rooms:
            if room.room_id == self_room_id:
                continue
            overlap = self._overlap_against_side(rect, room.rect, side)
            if overlap is not None:
                labeled.append((overlap[0], overlap[1], "room", room.room_id))

        for patio in patios:
            overlap = self._overlap_against_side(rect, patio.rect, side)
            if overlap is not None:
                labeled.append((overlap[0], overlap[1], "patio", patio.patio_id))

        labeled.sort(key=lambda item: (item[0], item[1]))
        return labeled

    def _overlap_against_side(self, room_rect: Rect, other_rect: Rect, side: str) -> Optional[Tuple[float, float]]:
        if side == "north" and math.isclose(room_rect.y_max, other_rect.y_min, abs_tol=EPSILON):
            return rect_overlap(room_rect.x_min, room_rect.x_max, other_rect.x_min, other_rect.x_max)
        if side == "south" and math.isclose(room_rect.y_min, other_rect.y_max, abs_tol=EPSILON):
            return rect_overlap(room_rect.x_min, room_rect.x_max, other_rect.x_min, other_rect.x_max)
        if side == "east" and math.isclose(room_rect.x_max, other_rect.x_min, abs_tol=EPSILON):
            return rect_overlap(room_rect.y_min, room_rect.y_max, other_rect.y_min, other_rect.y_max)
        if side == "west" and math.isclose(room_rect.x_min, other_rect.x_max, abs_tol=EPSILON):
            return rect_overlap(room_rect.y_min, room_rect.y_max, other_rect.y_min, other_rect.y_max)
        return None

    def _interval_difference(
        self,
        base: Tuple[float, float],
        covered: Iterable[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        normalized = sorted(normalize_interval(interval) for interval in covered if interval_length(interval) > EPSILON)
        current = base[0]
        leftover: List[Tuple[float, float]] = []
        for start, end in normalized:
            start = max(start, base[0])
            end = min(end, base[1])
            if start - current > EPSILON:
                leftover.append((current, start))
            current = max(current, end)
        if base[1] - current > EPSILON:
            leftover.append((current, base[1]))
        return leftover

    def _side_interval_to_segment(
        self,
        rect: Rect,
        side: str,
        start: float,
        end: float,
    ) -> Tuple[float, float, float, float]:
        if side == "north":
            return (start, rect.y_max, end, rect.y_max)
        if side == "south":
            return (start, rect.y_min, end, rect.y_min)
        if side == "east":
            return (rect.x_max, start, rect.x_max, end)
        if side == "west":
            return (rect.x_min, start, rect.x_min, end)
        raise ValueError(f"Unsupported side: {side}")

    def _assign_openings(self, boundaries: List[BoundarySegment], rooms: Sequence[Room]) -> None:
        graph = {room.room_id: set() for room in rooms}
        internal_candidates: Dict[frozenset[str], BoundarySegment] = {}
        exterior_candidates: List[BoundarySegment] = []
        corridor_ids = {room.room_id for room in rooms if room.room_type == "corridor"}

        for boundary in boundaries:
            if boundary.boundary_type == "internal" and boundary.adjacent_room_id is not None:
                graph[boundary.room_id].add(boundary.adjacent_room_id)
                graph[boundary.adjacent_room_id].add(boundary.room_id)
                key = frozenset({boundary.room_id, boundary.adjacent_room_id})
                current = internal_candidates.get(key)
                if current is None or boundary.length > current.length:
                    internal_candidates[key] = boundary
            if boundary.boundary_type == "exterior":
                exterior_candidates.append(boundary)

        for left_id, right_id in self._spanning_tree_edges(graph, corridor_ids):
            key = frozenset({left_id, right_id})
            boundary = internal_candidates.get(key)
            if boundary is None:
                continue
            boundary.openings.append(
                OpeningSpec(
                    opening_type="door",
                    desired_width=self.config.doorway_width,
                    door_width=self.config.door_width,
                    is_entrance=False,
                )
            )

        entrance_segment = self._choose_entrance_segment(exterior_candidates, corridor_ids)
        if entrance_segment is not None:
            entrance_segment.openings.append(
                OpeningSpec(
                    opening_type="door",
                    desired_width=self.config.doorway_width,
                    door_width=self.config.door_width,
                    is_entrance=True,
                )
            )

        for boundary in boundaries:
            if boundary.boundary_type not in {"exterior", "patio"}:
                continue
            boundary.openings.append(
                OpeningSpec(
                    opening_type="window",
                    desired_width=self.rng.uniform(
                        self.config.window_min_length,
                        self.config.window_max_length,
                    ),
                )
            )

    def _spanning_tree_edges(
        self,
        graph: Dict[str, set[str]],
        corridor_ids: set[str],
    ) -> List[Tuple[str, str]]:
        if not graph:
            return []
        root = next(iter(corridor_ids), next(iter(graph)))
        edges: List[Tuple[str, str]] = []
        queue = deque([root])
        visited = {root}
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph[node]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                edges.append((node, neighbor))
        if len(visited) != len(graph):
            raise RuntimeError("Room graph is disconnected; cannot assign doors.")
        return edges

    def _choose_entrance_segment(
        self,
        candidates: Sequence[BoundarySegment],
        corridor_ids: set[str],
    ) -> Optional[BoundarySegment]:
        if not candidates:
            return None
        min_length = self.config.doorway_width + self.config.window_min_length + 2 * self.config.opening_margin
        corridor_candidates = [candidate for candidate in candidates if candidate.room_id in corridor_ids and candidate.length >= min_length]
        if corridor_candidates:
            return max(corridor_candidates, key=lambda segment: segment.length)
        viable = [candidate for candidate in candidates if candidate.length >= min_length]
        if viable:
            return max(viable, key=lambda segment: segment.length)
        return max(candidates, key=lambda segment: segment.length)

    def _build_elements(self, boundaries: Sequence[BoundarySegment]) -> List[FloorElement]:
        elements: List[FloorElement] = []
        for boundary in boundaries:
            elements.extend(self._elements_for_boundary(boundary))
        if not elements:
            raise RuntimeError("No floor-plan elements were generated.")
        return elements

    def _elements_for_boundary(self, boundary: BoundarySegment) -> List[FloorElement]:
        thickness = (
            self.config.wall_thickness_external
            if boundary.boundary_type in {"exterior", "patio"}
            else self.config.wall_thickness_internal
        )

        laid_out_openings = self._layout_openings(boundary)
        cursor = 0.0
        produced: List[FloorElement] = []
        for offset, opening, opening_width in laid_out_openings:
            if offset - cursor > EPSILON:
                produced.append(self._make_wall_element(boundary, cursor, offset, thickness))
            produced.append(self._make_opening_element(boundary, offset, opening, opening_width))
            cursor = offset + opening_width
        if boundary.length - cursor > EPSILON:
            produced.append(self._make_wall_element(boundary, cursor, boundary.length, thickness))
        return [element for element in produced if element is not None]

    def _layout_openings(
        self,
        boundary: BoundarySegment,
    ) -> List[Tuple[float, OpeningSpec, float]]:
        if not boundary.openings:
            return []

        end_margin = self.config.opening_margin
        gap = self.config.opening_gap
        n = len(boundary.openings)
        usable = boundary.length - 2 * end_margin - gap * max(0, n - 1)
        if usable <= EPSILON:
            end_margin = 0.05
            gap = 0.05
            usable = boundary.length - 2 * end_margin - gap * max(0, n - 1)
        if usable <= EPSILON:
            return []

        fixed = 0.0
        flexible_indices: List[int] = []
        widths: List[float] = []
        for idx, opening in enumerate(boundary.openings):
            if opening.opening_type == "door":
                width = min(opening.desired_width, max(0.5, usable))
                widths.append(width)
                fixed += width
            else:
                widths.append(0.0)
                flexible_indices.append(idx)

        remaining = max(usable - fixed, 0.0)
        if flexible_indices:
            target_each = remaining / len(flexible_indices) if remaining > EPSILON else 0.0
            for idx in flexible_indices:
                widths[idx] = max(
                    0.5,
                    min(
                        boundary.openings[idx].desired_width,
                        max(target_each, 0.5),
                    ),
                )

            total_width = sum(widths)
            if total_width > usable + EPSILON:
                scale = usable / total_width
                widths = [max(0.45, width * scale) for width in widths]

        total_width = sum(widths)
        slots = n + 1
        slack = max(boundary.length - total_width, 0.0)
        step = slack / slots
        cursor = step
        placements: List[Tuple[float, OpeningSpec, float]] = []
        for opening, width in zip(boundary.openings, widths):
            placements.append((cursor, opening, width))
            cursor += width + step
        return placements

    def _make_wall_element(
        self,
        boundary: BoundarySegment,
        start_offset: float,
        end_offset: float,
        thickness: float,
    ) -> Optional[FloorElement]:
        length = end_offset - start_offset
        if length <= EPSILON:
            return None

        side = boundary.side
        if boundary.is_horizontal:
            x_start = min(boundary.x0, boundary.x1) + start_offset
            y = boundary.y0
            if boundary.boundary_type == "internal":
                anchor_x = x_start
                anchor_y = y - thickness / 2.0
            elif side == "south":
                anchor_x = x_start
                anchor_y = y - thickness
            else:
                anchor_x = x_start
                anchor_y = y
            orientation = 0.0
        else:
            y_start = min(boundary.y0, boundary.y1) + start_offset
            y_end = y_start + length
            x = boundary.x0
            if boundary.boundary_type == "internal":
                anchor_x = x - thickness / 2.0
                anchor_y = y_end
                orientation = 270.0
            elif side == "west":
                anchor_x = x - thickness
                anchor_y = y_end
                orientation = 270.0
            else:
                anchor_x = x
                anchor_y = y_start
                orientation = 90.0

        return FloorElement(
            element_id=next_id(self._id_counter, "element"),
            element_type="wall",
            room_id=boundary.room_id,
            adjacent_room_id=boundary.adjacent_room_id,
            patio_id=boundary.patio_id,
            x=anchor_x,
            y=anchor_y,
            length=length,
            thickness=thickness,
            orientation_angle=orientation,
            is_external=boundary.is_external,
            touches_patio=boundary.touches_patio,
        )

    def _make_opening_element(
        self,
        boundary: BoundarySegment,
        offset: float,
        opening: OpeningSpec,
        opening_width: float,
    ) -> FloorElement:
        if boundary.is_horizontal:
            x_start = min(boundary.x0, boundary.x1) + offset
            x_end = x_start + opening_width
            y = boundary.y0
            if boundary.side == "north":
                anchor_x = x_start
                anchor_y = y
                orientation = 0.0
                to_the_right = False
            elif boundary.side == "south":
                anchor_x = x_start
                anchor_y = y
                orientation = 0.0
                to_the_right = False
            else:
                anchor_x = x_start
                anchor_y = y
                orientation = 0.0
                to_the_right = False
        else:
            y_start = min(boundary.y0, boundary.y1) + offset
            y_end = y_start + opening_width
            x = boundary.x0
            if boundary.side == "west":
                anchor_x = x
                anchor_y = y_end
                orientation = 270.0
                to_the_right = True
            elif boundary.side == "east":
                anchor_x = x
                anchor_y = y_start
                orientation = 90.0
                to_the_right = True
            else:
                anchor_x = x
                anchor_y = y_start
                orientation = 90.0
                to_the_right = True

        if opening.opening_type == "door":
            return FloorElement(
                element_id=next_id(self._id_counter, "element"),
                element_type="door",
                room_id=boundary.room_id,
                adjacent_room_id=boundary.adjacent_room_id,
                patio_id=boundary.patio_id,
                x=anchor_x,
                y=anchor_y,
                orientation_angle=orientation,
                is_external=boundary.is_external,
                touches_patio=boundary.touches_patio,
                doorway_width=opening_width,
                door_width=min(opening.door_width or self.config.door_width, max(opening_width - 0.1, 0.45)),
                thickness=self.config.door_thickness,
                to_the_right=to_the_right,
                is_entrance=opening.is_entrance,
            )

        if boundary.is_horizontal:
            anchor_x = x_start
            anchor_y = y if boundary.side != "south" else y - self.config.window_overall_thickness
        else:
            if boundary.side == "west":
                anchor_x = x - self.config.window_overall_thickness
                anchor_y = y_end
            else:
                anchor_x = x
                anchor_y = y_start

        return FloorElement(
            element_id=next_id(self._id_counter, "element"),
            element_type="window",
            room_id=boundary.room_id,
            adjacent_room_id=boundary.adjacent_room_id,
            patio_id=boundary.patio_id,
            x=anchor_x,
            y=anchor_y,
            orientation_angle=orientation,
            is_external=boundary.is_external,
            touches_patio=boundary.touches_patio,
            length=opening_width,
            overall_thickness=self.config.window_overall_thickness,
            single_line_thickness=self.config.window_single_line_thickness,
        )

    def _patio_to_polygon_dict(self, patio: Patio) -> Dict[str, object]:
        rect = patio.rect
        return {
            "type": "polygon",
            "vertices": [
                [round(rect.x_min, 3), round(rect.y_min, 3)],
                [round(rect.x_max, 3), round(rect.y_min, 3)],
                [round(rect.x_max, 3), round(rect.y_max, 3)],
                [round(rect.x_min, 3), round(rect.y_max, 3)],
            ],
            "line_width": self.config.patio_line_width,
            "color": self.config.patio_fill_color,
        }

    def _floor_plan_view_rect(self, environment_rect: Rect, building_rect: Rect) -> Rect:
        padding = max(
            self.config.layout_padding,
            self.config.floor_plan_padding_ratio * max(building_rect.width, building_rect.height),
        )
        view_rect = building_rect.expanded(padding)
        return Rect(
            x_min=max(view_rect.x_min, environment_rect.x_min - self.config.layout_padding),
            y_min=max(view_rect.y_min, environment_rect.y_min - self.config.layout_padding),
            x_max=min(view_rect.x_max, environment_rect.x_max + self.config.layout_padding),
            y_max=min(view_rect.y_max, environment_rect.y_max + self.config.layout_padding),
        )

    def _resolve_scale_denominator(self, view_rect: Rect) -> int:
        long_edge_meters = max(view_rect.width, view_rect.height)
        target_long_edge_pixels = int(
            max(
                self.config.floor_plan_min_long_edge_pixels,
                min(
                    self.config.floor_plan_max_long_edge_pixels,
                    math.ceil(long_edge_meters * self.config.floor_plan_target_pixels_per_meter),
                ),
            )
        )
        raw_denominator = (
            long_edge_meters
            * self.config.dpi
            * self.config.scale_numerator
            / (target_long_edge_pixels * METERS_PER_INCH)
        )
        return max(
            self.config.min_scale_denominator,
            min(self.config.max_scale_denominator, max(1, int(round(raw_denominator)))),
        )

    def _write_yaml(
        self,
        environment_rect: Rect,
        building_rect: Rect,
        patios: Sequence[Patio],
        elements: Sequence[FloorElement],
    ) -> Path:
        view_rect = self._floor_plan_view_rect(environment_rect, building_rect)
        scale_denominator = self._resolve_scale_denominator(view_rect)
        layout = {
            "bottom_left_corner": [
                round(view_rect.x_min, 3),
                round(view_rect.y_min, 3),
            ],
            "top_right_corner": [
                round(view_rect.x_max, 3),
                round(view_rect.y_max, 3),
            ],
            "scale_numerator": self.config.scale_numerator,
            "scale_denominator": scale_denominator,
            "grid_major_step": None,
            "grid_minor_step": None,
        }

        settings = {
            "project": {
                "dpi": self.config.dpi,
                "pdf_file": None,
                "png_dir": str(self.config.png_dir) if self.config.render_png else None,
            },
            "default_layout": layout,
            "reusable_elements": {
                "patio_elements": [self._patio_to_polygon_dict(patio) for patio in patios],
                "generated_elements": [element.to_renovation_dict() for element in elements],
            },
            "floor_plans": [
                {
                    "title": {
                        "text": self.config.title,
                        "font_size": 16,
                    },
                    "inherited_elements": ["patio_elements", "generated_elements"],
                }
            ],
        }

        with self.config.yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(settings, handle, sort_keys=False)
        return self.config.yaml_path

    def _render_yaml(self, yaml_path: Path) -> List[Path]:
        for stale_png in self.config.png_dir.glob("*.png"):
            stale_png.unlink()

        env = os.environ.copy()
        env.update(
            {
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": "/tmp/codex_mpl",
                "XDG_CACHE_HOME": "/tmp",
            }
        )
        result = subprocess.run(
            [sys.executable, "-m", "renovation", "-c", str(yaml_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "renovation rendering failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        rendered_pngs = sorted(self.config.png_dir.glob("*.png"))
        if not rendered_pngs:
            return []

        target_png = self.config.png_dir / self.config.png_filename
        primary_png = rendered_pngs[0]
        if primary_png != target_png:
            if target_png.exists():
                target_png.unlink()
            primary_png.replace(target_png)
        rendered_pngs = [target_png] + [png for png in rendered_pngs[1:] if png != target_png]
        return rendered_pngs

    def _write_parquet(self, elements: Sequence[FloorElement]) -> Path:
        frame = pd.DataFrame([asdict(element) for element in elements])
        frame.to_parquet(self.config.parquet_path, index=False)
        return self.config.parquet_path


def generate_floor_plan_from_environment(
    environment: Environment,
    *,
    output_dir: Optional[Path] = None,
    random_seed: Optional[int] = None,
    artifact_stem: Optional[str] = None,
    render_png: bool = False,
) -> GeneratedFloorPlan:
    config = FloorPlanConfig.from_environment(
        environment,
        output_dir=output_dir,
        random_seed=random_seed,
        artifact_stem=artifact_stem,
        render_png=render_png,
    )
    return FloorPlanFactory(config).generate()
