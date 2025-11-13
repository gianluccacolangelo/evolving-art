import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box
from shapely import affinity
import shapely.ops
from typing import List, Tuple, Union, Any
import io
from PIL import Image
from typing import Optional


from evolution.genome import CompositionGenome, PrimitiveNode, OpNode, PrimitiveGene
from shapes.geometry import ColorAlgebra

COLOR_ALG = ColorAlgebra()

def gene_to_shapely(
    gene: PrimitiveGene
) -> Any:
    if gene.kind == "disk":
        base = Point(0, 0).buffer(1.0, resolution=64) 
    elif gene.kind == "square":
        base = box(-0.5, -0.5, 0.5, 0.5)
    elif gene.kind == "polygon":
        base = Polygon(gene.polygon_vertices)
        if not base.is_valid:
            base = base.buffer(0) 
    else:
        raise ValueError(f"Unknown kind {gene.kind}")
    
    t = gene.transform
    geom = affinity.scale(base, xfact=t.sx, yfact=t.sy, origin=(0, 0))
    geom = affinity.rotate(geom, t.theta, origin=(0, 0), use_radians=True)
    geom = affinity.translate(geom, xoff=t.dx, yoff=t.dy)
    return geom

def shatter_and_blend(
    existing_patches: List[Tuple[Any, np.ndarray]],
    new_geom: Any,
    new_color: np.ndarray,
    blend_mode: str = "or",
) -> List[Tuple[Any, np.ndarray]]:
    """
    Takes a list of existing patches [(geom, color), ...] and a new shape.
    Shatters the shapes into pieces (intersections and differences) to calculate
    the resulting color in each overlapping zone.
    """
    next_gen_patches = []
    shape_to_add = new_geom

    for exist_geom, exist_color in existing_patches:
        if shape_to_add.is_empty:
            next_gen_patches.append((exist_geom, exist_color))
            continue
            
        intersection = exist_geom.intersection(shape_to_add)
        
        if intersection.is_empty:
            next_gen_patches.append((exist_geom, exist_color))
        else:
            diff_exist = exist_geom.difference(shape_to_add)
            if not diff_exist.is_empty:
                next_gen_patches.append((diff_exist, exist_color))
            
            if blend_mode == "or":
                blended_color = COLOR_ALG.or_color(exist_color, new_color)
            elif blend_mode == "and":
                blended_color = COLOR_ALG.and_color(exist_color, new_color)
            else:
                blended_color = new_color
                
            next_gen_patches.append((intersection, blended_color))
            
            shape_to_add = shape_to_add.difference(exist_geom)
            
    if not shape_to_add.is_empty:
        next_gen_patches.append((shape_to_add, new_color))
        
    return next_gen_patches

def process_node(
    node: Union[PrimitiveNode, OpNode]
) -> List[Tuple[Any, np.ndarray]]:
    """
    Returns a list of tuples [(shapely_geometry, numpy_rgb_color)].
    """
    if isinstance(node, PrimitiveNode):
        geom = gene_to_shapely(node.gene)
        return [(geom, node.gene.color_rgb)]

    elif isinstance(node, OpNode):
        children_results = [process_node(child) for child in node.children]
        if not children_results:
            return []

        if node.kind == "union":
            current_patches = children_results[0]
            for next_child_list in children_results[1:]:
                for (new_geom, new_col) in next_child_list:
                    current_patches = shatter_and_blend(current_patches, new_geom, new_col, blend_mode="or")
            return current_patches

        if node.kind == "intersection":
            current_shapes = children_results[0]
            for next_child_list in children_results[1:]:
                new_shapes = []
                for (geom_a, col_a) in current_shapes:
                    for (geom_b, col_b) in next_child_list:
                        inter = geom_a.intersection(geom_b)
                        if not inter.is_empty:
                            new_col = COLOR_ALG.and_color(col_a, col_b)
                            new_shapes.append((inter, new_col))
                current_shapes = new_shapes
            return current_shapes

        if node.kind == "difference":
            current_shapes = children_results[0]
            subtractors = []
            for child_list in children_results[1:]:
                for geom, _ in child_list:
                    subtractors.append(geom)
            
            if subtractors:
                total_subtractor = shapely.ops.unary_union(subtractors)
                final_shapes = []
                for (geom_a, col_a) in current_shapes:
                    diff = geom_a.difference(total_subtractor)
                    if not diff.is_empty:
                        final_shapes.append((diff, col_a))
                return final_shapes
            else:
                return current_shapes
    return []

def draw_genome_on_axis(
    ax: plt.Axes,
    genome: CompositionGenome
) -> None:
    """
    Renders the vector genome directly onto a given Matplotlib axis.
    First clips all geometry to the final calculated bounds.
    """
    shapes_and_colors = process_node(genome.root)
    
    all_geoms = [s[0] for s in shapes_and_colors if not s[0].is_empty]
    
    if not all_geoms:
        ax.set_aspect('equal')
        ax.axis('off')
        return

    total_shape = shapely.ops.unary_union(all_geoms)
    minx, miny, maxx, maxy = total_shape.bounds
    
    width = maxx - minx
    height = maxy - miny
    
    max_dim = max(width, height)
    
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    margin = 0.1
    half_side = (max_dim / 2) + margin
    
    clip_xmin = center_x - half_side
    clip_xmax = center_x + half_side
    clip_ymin = center_y - half_side
    clip_ymax = center_y + half_side
    
    clip_box = box(clip_xmin, clip_ymin, clip_xmax, clip_ymax)
    
    ax.set_aspect('equal')
    ax.set_xlim(clip_xmin, clip_xmax)
    ax.set_ylim(clip_ymin, clip_ymax)
    ax.axis('off')

    for geom, rgb in shapes_and_colors:
        rgb = np.array(rgb).flatten()
        rgba = np.append(rgb[:3], 1.0) 
        
        if hasattr(geom, 'geoms'):
            sub_geoms = geom.geoms
        else:
            sub_geoms = [geom]
            
        for part in sub_geoms:
            clipped_part = part.intersection(clip_box)
            
            if clipped_part.is_empty: continue
            
            if hasattr(clipped_part, 'geoms'):
                draw_geoms = clipped_part.geoms
            else:
                draw_geoms = [clipped_part]
                
            for final_part in draw_geoms:

                if isinstance(final_part, Polygon):
                    x, y = final_part.exterior.xy
                    
                    ax.fill(x, y, fc=rgba, ec=rgba, linewidth=0.5, joinstyle='round') 
                    
                    for interior in final_part.interiors:
                            xi, yi = interior.xy
                            ax.fill(xi, yi, fc='white', ec=None)

def save_genome_as_svg(
    genome: CompositionGenome,
    filename: str
) -> None:
    """
    Saves the genome as an SVG with geometrically blended colors.
    Wraps draw_genome_on_axis to create the figure and save.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    draw_genome_on_axis(ax, genome)
    
    fig.savefig(
        filename, 
        format='svg', 
        bbox_inches='tight', 
        pad_inches=0
    )
    plt.close(fig)

def save_genome_as_png(
    genome: CompositionGenome,
    filename: Optional[str] = None,
    resolution: int = 128
) -> Optional[Image.Image]:
    """
    Saves the genome as a low-res PNG using the vectorizer strategy for faster GUI display, 
    or returns the PIL Image object if filename is None (in-memory rendering).
    """
    # Usamos un tamaño de figura pequeño, y ajustamos el DPI para obtener la resolución final.
    DPI_CALC = resolution / 3.0 
    
    fig, ax = plt.subplots(figsize=(3, 3)) 

    draw_genome_on_axis(ax, genome)
    
    if filename is None:
        # Save to an in-memory buffer
        buffer = io.BytesIO()
        fig.savefig(
            buffer, 
            format='png', 
            dpi=DPI_CALC, 
            bbox_inches='tight', 
            pad_inches=0,
            transparent=False,
            facecolor='white'
        )
        plt.close(fig)
        buffer.seek(0)
        # Load PIL Image from buffer
        return Image.open(buffer) 
    else:
        # Save to file (old behavior)
        fig.savefig(
            filename, 
            format='png', 
            dpi=DPI_CALC, 
            bbox_inches='tight', 
            pad_inches=0,
            transparent=False, 
            facecolor='white'
        )
        plt.close(fig)
        return None