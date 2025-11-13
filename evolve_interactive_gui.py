from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import Optional

import customtkinter
from PIL import Image
import tkinter.messagebox

from evolution import (
    GAConfig,
    initialize_population,
    evolve_one_generation,
)
from evolution.genome import PrimitiveNode, OpNode, PrimitiveGene, TransformParams, CompositionGenome
from plotting import render_to_file
from plotting.renderer import render_population_grid

GUI_IMAGE_SIZE = 250
GUI_IMAGE_RESOLUTION = 500
GUI_ITEM_WIDTH_ESTIMATE = GUI_IMAGE_SIZE + 30 


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments (translated)."""
    p = argparse.ArgumentParser(description="Interactive evolution of geometric compositions (GUI).")
    p.add_argument("--pop", type=int, default=16, help="population size")
    p.add_argument("--genes", type=int, default=4, help="number of primitives per composition")
    p.add_argument("--gens", type=int, default=9999, help="number of generations to run")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--outdir", type=str, default="plots/evolve", help="output directory for grid images")
    p.add_argument("--tag", type=str, default="", help="experiment tag (subfolder) to organize outputs")
    p.add_argument("--save", type=int, nargs="*", default=[], help="indices to save in high quality (besides GUI)")
    p.add_argument("--save-res", type=int, default=1200, help="resolution for high-quality individual renders")
    p.add_argument("--save-dpi", type=int, default=300, help="DPI for high-quality individual renders")
    
    p.add_argument("--hq-format", type=str, default="svg", choices=["png", "svg"], help="Format for high-quality individual renders (png or svg)")
    
    p.add_argument("--checkpoint", type=str, default="", help="path to a *_params.json file to initialize population from")
    p.add_argument("--summary-cols", type=int, default=4, help="columns in the *saved summary grid image*")
    return p.parse_args()


def _gene_to_dict(g) -> dict:
    """Serializes a PrimitiveGene to a dictionary."""
    d = {
        "kind": g.kind,
        "transform": {
            "sx": float(g.transform.sx),
            "sy": float(g.transform.sy),
            "theta": float(g.transform.theta),
            "dx": float(g.transform.dx),
            "dy": float(g.transform.dy),
        },
        "color_rgb": [float(x) for x in (g.color_rgb.tolist() if hasattr(g.color_rgb, "tolist") else g.color_rgb)],
        "polygon_vertices": None,
    }
    if g.polygon_vertices is not None:
        d["polygon_vertices"] = [[float(a), float(b)] for a, b in g.polygon_vertices.tolist()]
    return d


def _node_to_dict(node) -> dict:
    """Serializes a PrimitiveNode or OpNode to a dictionary."""
    if isinstance(node, PrimitiveNode):
        return {"primitive": _gene_to_dict(node.gene)}
    if isinstance(node, OpNode):
        return {
            "op": node.kind,
            "children": [_node_to_dict(ch) for ch in node.children],
        }
    return {"unknown": True}


def _population_to_dict(population) -> dict:
    """Serializes the entire population to a dictionary."""
    return {
        "population": [
            {
                "index": i,
                "composition": _node_to_dict(genome.root),
            }
            for i, genome in enumerate(population)
        ]
    }

def _dict_to_node(d) -> PrimitiveNode | OpNode:
    """Deserializes a dictionary into a PrimitiveNode or OpNode."""
    if "primitive" in d:
        g = d["primitive"]
        t = g["transform"]
        gene = PrimitiveGene(
            kind=g["kind"],
            transform=TransformParams(
                sx=float(t["sx"]),
                sy=float(t["sy"]),
                theta=float(t["theta"]),
                dx=float(t["dx"]),
                dy=float(t["dy"]),
            ),
            color_rgb=np.array(g.get("color_rgb", [0.6, 0.6, 0.6]), dtype=float),
            polygon_vertices=None if g.get("polygon_vertices") is None else np.array(g["polygon_vertices"], dtype=float),
        )
        return PrimitiveNode(gene=gene)
    if "op" in d and "children" in d:
        return OpNode(kind=str(d["op"]), children=[_dict_to_node(c) for c in d["children"]])
    raise ValueError("Invalid composition node in checkpoint JSON")


class EvolutionApp(customtkinter.CTk):
    """
    Main class that handles the GUI and evolution state.
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        self.outdir = args.outdir if not args.tag else os.path.join(args.outdir, args.tag)
        self.temp_dir = os.path.join(self.outdir, "gui_temp_renders")
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.generation = 0
        self.likes = set()
        self.saves_ui = {}
        self.image_buttons = []
        self.num_elites_from_prev_gen = 0
        
        self.ind_frames = []
        self.current_cols = 0

        self.title("Interactive Evolution")
        self.geometry("1100x800")

        self.population, self.cfg, self.rng = self.load_initial_population()

        self.setup_ui()

        self.display_current_generation()

    def load_initial_population(self) -> tuple[list[CompositionGenome], GAConfig, np.random.Generator]:
        """Loads population from checkpoint or initializes a new one."""
        if self.args.checkpoint:
            print(f"Loading checkpoint from {self.args.checkpoint}")
            with open(self.args.checkpoint, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pop_items = meta.get("data", {}).get("population", [])
            population = []
            for item in pop_items:
                comp = item.get("composition")
                if comp is None:
                    continue
                node = _dict_to_node(comp)
                population.append(CompositionGenome(root=node))
            if not population:
                raise ValueError(f"Checkpoint {self.args.checkpoint} contained no valid population")
            cfg = GAConfig(population_size=len(population), num_genes=self.args.genes, random_seed=self.args.seed)
            rng = np.random.default_rng(cfg.random_seed)
            print(f"Loaded {len(population)} individuals.")
        else:
            print("Initializing new population.")
            cfg = GAConfig(population_size=self.args.pop, num_genes=self.args.genes, random_seed=self.args.seed)
            rng, population = initialize_population(cfg)
        return population, cfg, rng

    def setup_ui(self) -> None:
        """Configures the static GUI widgets."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.control_frame = customtkinter.CTkFrame(self, height=50)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.gen_label = customtkinter.CTkLabel(self.control_frame, text="Generation: 0", font=("Arial", 16))
        self.gen_label.pack(side="left", padx=20)

        self.next_gen_button = customtkinter.CTkButton(
            self.control_frame, 
            text="Evolve Next Generation", 
            font=("Arial", 16, "bold"),
            command=self.next_generation_step
        )
        self.next_gen_button.pack(side="right", padx=20, pady=10)

        self.scroll_frame = customtkinter.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.scroll_frame.bind("<Configure>", self.on_resize_grid)

    def render_individual_for_gui(self, genome: CompositionGenome, index: int) -> Image.Image:
        """Renders a single individual in-memory and returns the PIL.Image."""
        
        temp_path = os.path.join(self.temp_dir, f"gen_{self.generation:03d}_ind_{index:02d}.png")
        
        pil_image = render_to_file(
            genome,
            out_path=temp_path,
            resolution=GUI_IMAGE_RESOLUTION,
            dpi=72,
            title=None,
            draw_edges=False,
            show_axes=False,
            show_grid=False,
            frame_only=True,
            format="png",
            transparent=False, 
            return_image=True,
        )
            
        if pil_image is None:
            raise RuntimeError("In-memory rendering failed or returned None.")
            
        return pil_image


    def on_resize_grid(self, event) -> None:
        """Called when the scroll_frame is resized."""
        
        new_cols = max(1, event.width // GUI_ITEM_WIDTH_ESTIMATE)
        
        if new_cols == self.current_cols:
            return
            
        self.current_cols = new_cols
        
        self.re_grid_population()


    def re_grid_population(self) -> None:
        """Re-arranges the existing individual frames into the grid."""
        if not self.ind_frames or self.current_cols == 0:
            return

        for c in range(self.scroll_frame.grid_size()[0]):
             self.scroll_frame.grid_columnconfigure(c, weight=0)

        for c in range(self.current_cols):
            self.scroll_frame.grid_columnconfigure(c, weight=1)

        for i, frame in enumerate(self.ind_frames):
            row = i // self.current_cols
            col = i % self.current_cols
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            

    def display_current_generation(self) -> None:
        """Clears and *creates* widgets for the current population."""
        
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.likes.clear()
        self.saves_ui.clear()
        self.image_buttons.clear()
        self.ind_frames.clear()

        self.gen_label.configure(text=f"Generation: {self.generation}")

        self.save_generation_files()

        for i, genome in enumerate(self.population):
            is_elite = (i < self.num_elites_from_prev_gen)
            border_color = "yellow" if is_elite else "gray20"

            ind_frame = customtkinter.CTkFrame(
                self.scroll_frame, 
                border_width=2, 
                border_color=border_color
            )
            
            try:
                pil_image = self.render_individual_for_gui(genome, i)
                ctk_image = customtkinter.CTkImage(pil_image, size=(GUI_IMAGE_SIZE, GUI_IMAGE_SIZE))
                
                img_button = customtkinter.CTkButton(
                    ind_frame,
                    image=ctk_image,
                    text=f"#{i}",
                    font=("Arial", 12, "bold"),
                    compound="top",
                    fg_color="transparent",
                    border_width=0,
                    command=lambda idx=i: self.toggle_like(idx)
                )
                img_button.pack(padx=5, pady=5)
                self.image_buttons.append(img_button)

                save_check = customtkinter.CTkCheckBox(ind_frame, text="Save HQ")
                save_check.pack(padx=5, pady=(0, 5))
                self.saves_ui[i] = save_check
            
            except Exception as e:
                print(f"Error rendering individual {i}: {e}")
                placeholder = customtkinter.CTkLabel(ind_frame, text=f"Render\nError\n#{i}", width=GUI_IMAGE_SIZE, height=GUI_IMAGE_SIZE, fg_color="gray10")
                placeholder.pack(padx=5, pady=5)
                self.image_buttons.append(None) 

            self.ind_frames.append(ind_frame)

        self.re_grid_population()


    def save_generation_files(self, final: bool = False) -> None:
        """Saves the grid image and parameters JSON."""
        
        g = self.generation
        suffix = f"gen_{g:03d}"
        if final:
            suffix = f"gen_{self.args.gens:03d}_final"

        out_path = os.path.join(self.outdir, f"{suffix}.svg")
        print(f"Saving grid image -> {out_path}")
        
        render_population_grid(self.population, out_path=out_path, cols=self.args.summary_cols, use_vector=True) 
        
        params_path = os.path.join(self.outdir, f"{suffix}_params.json")
        meta = {
            "tag": self.args.tag,
            "generation": g,
            "final": final,
            "config": {
                "population_size": self.cfg.population_size,
                "num_genes": self.cfg.num_genes,
                "random_seed": self.cfg.random_seed,
            },
            "data": _population_to_dict(self.population),
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def toggle_like(self, index: int) -> None:
        """Handles clicking on an image to 'like' it."""
        button = self.image_buttons[index]
        if not button: return

        if index in self.likes:
            self.likes.remove(index)
            button.configure(border_width=0, border_color="gray20")
        else:
            self.likes.add(index)
            button.configure(border_width=4, border_color="green")

    def next_generation_step(self) -> None:
        """Handles the 'Evolve' button click."""
        
        if not self.likes:
            tkinter.messagebox.showwarning(
                "Selection Required", 
                "You have to select at least one image."
            )
            return
        
        self.next_gen_button.configure(state="disabled", text="Generating...")
        self.update_idletasks()
        
        saves_from_ui = {i for i, chk in self.saves_ui.items() if chk.get() == 1}
        
        to_save = sorted(saves_from_ui.union(set(i for i in self.args.save if 0 <= i < len(self.population))))
        
        print(f"Generation {self.generation} completed.")
        print(f"  Saves: {to_save if to_save else 'none'}")

        for idx in to_save:
            try:
                genome = self.population[idx]
                
                ext = self.args.hq_format
                hq_path = os.path.join(self.outdir, f"gen_{self.generation:03d}_idx_{idx:02d}_hq.{ext}")
                
                print(f"  Saving HQ individual #{idx} -> {hq_path}")
                
                render_to_file(
                    genome, 
                    out_path=hq_path,
                    resolution=self.args.save_res,
                    dpi=self.args.save_dpi,
                    title=f"Gen {self.generation} - Indiv #{idx}",
                    draw_edges=False,
                    show_axes=True,
                    show_grid=False,
                    frame_only=True,
                    format=self.args.hq_format,
                    transparent=False,
                )
            except Exception as e:
                print(f"Error saving HQ #{idx}: {e}")

        self.num_elites_from_prev_gen = len(self.likes)
        
        self.population = evolve_one_generation(
            self.rng, 
            self.population, 
            list(self.likes),
            self.cfg
        )
        
        self.generation += 1

        if self.generation > self.args.gens:
            self.gen_label.configure(text=f"Evolution Finished (Total: {self.args.gens} gen)")
            self.next_gen_button.configure(state="disabled", text="Finished")
            print("Evolution complete. Saving final state.")
            self.save_generation_files(final=True)
        else:
            self.display_current_generation()
            
            self.next_gen_button.configure(state="normal", text="Evolve Next Generation")


def main() -> None:
    """Entry point for the GUI application."""
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")
    
    args = parse_args()
    
    app = EvolutionApp(args)
    app.mainloop()
    
    print("Closing application.")


if __name__ == "__main__":
    main()