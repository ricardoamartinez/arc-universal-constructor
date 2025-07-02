#!/usr/bin/env python3
"""
GPU-Accelerated Visualizer for ARC Universal Constructor

Real-time visualization of:
- ARC puzzle grids (input/output/predictions)
- Neural architecture internal states (GNCA, Memory, Attention)
- Blueprint execution with construction arms
- Training metrics and loss curves
- Interactive controls for training speed and debugging
"""

import dearpygui.dearpygui as dpg
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import threading
import queue

# ARC color palette (RGB 0-255 for DearPyGui, with better contrast)
ARC_COLORS = {
    -1: (50, 50, 50),        # Empty (dark gray for constructor canvas)
    0:  (20, 20, 20),        # Black (slightly visible)
    1:  (0, 119, 190),       # Blue
    2:  (255, 51, 51),       # Red
    3:  (0, 204, 0),         # Green
    4:  (255, 255, 0),       # Yellow
    5:  (160, 160, 160),     # Gray
    6:  (255, 0, 255),       # Magenta/Fuchsia
    7:  (255, 165, 0),       # Orange
    8:  (135, 206, 235),     # Sky Blue/Azure
    9:  (139, 69, 19),       # Brown
}

@dataclass
class VisualizationState:
    """Container for all visualization state"""
    # Training state
    is_paused: bool = False
    speed_multiplier: float = 1.0
    current_epoch: int = 0
    current_task: int = 0
    
    # Metrics
    loss_history: deque = None
    reward_history: deque = None
    iou_history: deque = None
    
    # Grid states
    current_input: Optional[torch.Tensor] = None
    current_output: Optional[torch.Tensor] = None
    current_prediction: Optional[torch.Tensor] = None
    construction_canvas: Optional[torch.Tensor] = None
    
    # Neural states
    gnca_activations: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    memory_state: Optional[torch.Tensor] = None
    
    # Blueprint execution
    blueprint: List[int] = None
    blueprint_step: int = 0
    arm_positions: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        self.loss_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.iou_history = deque(maxlen=1000)


class ARCGPUVisualizer:
    """GPU-accelerated visualizer for ARC Universal Constructor"""
    
    def __init__(self, width: int = 1600, height: int = 900):
        self.width = width
        self.height = height
        self.state = VisualizationState()
        self.update_queue = queue.Queue()
        self.is_running = False
        
        # Grid rendering parameters
        self.cell_size = 20
        self.grid_spacing = 0
        
        # Initialize DearPyGui
        dpg.create_context()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        # Configure viewport
        dpg.create_viewport(title="ARC Universal Constructor - GPU Visualizer", 
                          width=self.width, height=self.height)
        dpg.setup_dearpygui()
        
        # Main window
        with dpg.window(label="Main", tag="main_window", 
                       width=self.width, height=self.height,
                       no_title_bar=True, no_move=True, no_resize=True):
            
            # Top control bar
            with dpg.group(horizontal=True):
                dpg.add_button(label="Pause" if not self.state.is_paused else "Resume",
                             callback=self._toggle_pause, tag="pause_button")
                dpg.add_button(label="Step", callback=self._step_forward)
                
                dpg.add_text("Speed:")
                dpg.add_slider_float(default_value=1.0, min_value=0.1, max_value=10.0,
                                   width=100, callback=self._update_speed,
                                   tag="speed_slider")
                
                dpg.add_text("Epoch:")
                dpg.add_text("0", tag="epoch_text")
                dpg.add_text("Task:")
                dpg.add_text("0/0", tag="task_text")
                
            dpg.add_separator()
            
            # Main content area with tabs
            with dpg.tab_bar():
                # Grids tab
                with dpg.tab(label="ARC Grids"):
                    with dpg.group(horizontal=True):
                        # Input grid
                        with dpg.group():
                            dpg.add_text("Input")
                            dpg.add_drawlist(width=300, height=300, tag="input_grid")
                            
                        # Output/Target grid
                        with dpg.group():
                            dpg.add_text("Target Output")
                            dpg.add_drawlist(width=300, height=300, tag="output_grid")
                            
                        # Prediction grid
                        with dpg.group():
                            dpg.add_text("Model Prediction")
                            dpg.add_drawlist(width=300, height=300, tag="prediction_grid")
                            
                        # Construction canvas
                        with dpg.group():
                            dpg.add_text("Construction Canvas")
                            dpg.add_drawlist(width=300, height=300, tag="construction_grid")
                
                # Neural States tab
                with dpg.tab(label="Neural States"):
                    with dpg.group(horizontal=True):
                        # GNCA activations
                        with dpg.group():
                            dpg.add_text("GNCA Activations")
                            with dpg.texture_registry():
                                dpg.add_raw_texture(width=32, height=32, default_value=[0.0]*32*32*4,
                                                  format=dpg.mvFormat_Float_rgba, tag="gnca_texture")
                            dpg.add_image("gnca_texture", width=400, height=300, tag="gnca_image")
                            
                        # Attention weights
                        with dpg.group():
                            dpg.add_text("Attention Weights")
                            with dpg.texture_registry():
                                dpg.add_raw_texture(width=32, height=32, default_value=[0.0]*32*32*4,
                                                  format=dpg.mvFormat_Float_rgba, tag="attention_texture")
                            dpg.add_image("attention_texture", width=400, height=300, tag="attention_image")
                    
                    # Memory state (for DNC/TTM)
                    dpg.add_text("External Memory State")
                    with dpg.texture_registry():
                        dpg.add_raw_texture(width=64, height=32, default_value=[0.0]*64*32*4,
                                          format=dpg.mvFormat_Float_rgba, tag="memory_texture")
                    dpg.add_image("memory_texture", width=800, height=200, tag="memory_image")
                
                # Metrics tab
                with dpg.tab(label="Training Metrics"):
                    with dpg.plot(label="Training Progress", height=400, width=-1):
                        dpg.add_plot_legend()
                        
                        # X-axis
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Steps")
                        
                        # Y-axes
                        y_axis_1 = dpg.add_plot_axis(dpg.mvYAxis, label="Loss/Reward")
                        y_axis_2 = dpg.add_plot_axis(dpg.mvYAxis, label="IoU", tag="iou_axis")
                        
                        # Series
                        dpg.add_line_series([], [], label="Loss", parent=y_axis_1, tag="loss_series")
                        dpg.add_line_series([], [], label="Reward", parent=y_axis_1, tag="reward_series")
                        dpg.add_line_series([], [], label="IoU", parent=y_axis_2, tag="iou_series")
                
                # Blueprint tab
                with dpg.tab(label="Blueprint Execution"):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset", callback=self._reset_blueprint)
                        dpg.add_button(label="Step Blueprint", callback=self._step_blueprint)
                        dpg.add_text("Step: 0/0", tag="blueprint_step_text")
                    
                    # Blueprint visualization
                    dpg.add_text("Blueprint Operations:", tag="blueprint_text")
                    with dpg.child_window(height=100, horizontal_scrollbar=True):
                        dpg.add_text("", tag="blueprint_ops")
                    
                    dpg.add_text("Construction Progress:")
                    dpg.add_drawlist(width=600, height=400, tag="blueprint_canvas")
        
        # Configure theme
        self._setup_theme()
        
        # Show viewport
        dpg.show_viewport()
        
    def _setup_theme(self):
        """Setup dark theme"""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
                
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (0.1, 0.1, 0.1, 1.0))
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (0.15, 0.15, 0.15, 1.0))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0.2, 0.2, 0.2, 1.0))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0.3, 0.3, 0.3, 1.0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0.4, 0.4, 0.4, 1.0))
                
        dpg.bind_theme(global_theme)
    
    def _draw_grid(self, drawlist_tag: str, grid: torch.Tensor, 
                   arm_positions: Optional[List[Tuple[int, int]]] = None):
        """Draw an ARC grid on a drawlist"""
        dpg.delete_item(drawlist_tag, children_only=True)
        
        if grid is None:
            return
        
        # Convert to numpy if needed
        if isinstance(grid, torch.Tensor):
            grid = grid.cpu().numpy()
        
        h, w = grid.shape
        
        # Get drawlist dimensions
        drawlist_width = dpg.get_item_width(drawlist_tag)
        drawlist_height = dpg.get_item_height(drawlist_tag)
        
        # Draw background
        dpg.draw_rectangle((0, 0), (drawlist_width, drawlist_height),
                         fill=(80, 80, 80), parent=drawlist_tag)
        
        # Calculate cell size to fit in drawlist
        cell_size = min(drawlist_width // w, drawlist_height // h) - self.grid_spacing
        
        # Draw cells
        for y in range(h):
            for x in range(w):
                color_idx = int(grid[y, x])
                color = ARC_COLORS.get(color_idx, (128, 128, 128))
                
                x1 = x * (cell_size + self.grid_spacing) + self.grid_spacing
                y1 = y * (cell_size + self.grid_spacing) + self.grid_spacing
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                dpg.draw_rectangle((x1, y1), (x2, y2), 
                                 fill=color, parent=drawlist_tag)
        
        # Draw arms if provided
        if arm_positions:
            for i, (ax, ay) in enumerate(arm_positions):
                if 0 <= ax < w and 0 <= ay < h:
                    cx = ax * (cell_size + self.grid_spacing) + self.grid_spacing + cell_size // 2
                    cy = ay * (cell_size + self.grid_spacing) + self.grid_spacing + cell_size // 2
                    
                    # Draw arm as a circle with number
                    dpg.draw_circle((cx, cy), cell_size // 3, 
                                  color=(255, 255, 0),
                                  fill=(255, 255, 100),
                                  parent=drawlist_tag)
                    dpg.draw_text((cx - 5, cy - 8), f"A{i}", 
                                color=(0, 0, 0),
                                size=12, parent=drawlist_tag)
    
    def _update_heatmap(self, tag: str, data: torch.Tensor):
        """Update a heatmap texture visualization"""
        if data is None:
            return
        
        # Convert to numpy and ensure 2D
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Handle different shapes
        if len(data.shape) == 1:
            # Reshape 1D data to 2D
            n = int(np.sqrt(len(data)))
            if n * n != len(data):
                n = min(len(data) // 8, 32)
                if n * 8 <= len(data):
                    data = data[:n*8].reshape(n, 8)
                else:
                    data = data[:n*n].reshape(n, n)
            else:
                data = data.reshape(n, n)
        elif len(data.shape) > 2:
            # Flatten to 2D if needed
            data = data.reshape(data.shape[0], -1)
        
        # Resize to match texture dimensions
        if tag == "gnca_texture" or tag == "attention_texture":
            target_size = (32, 32)
        elif tag == "memory_texture":
            target_size = (32, 64)  # height, width for memory
        else:
            target_size = (32, 32)
        
        # Resize data if needed
        if data.shape != target_size:
            # Simple resize by taking subset or padding
            h, w = target_size
            if data.shape[0] < h:
                # Pad with zeros
                pad_h = h - data.shape[0]
                data = np.pad(data, ((0, pad_h), (0, 0)), mode='constant')
            else:
                data = data[:h, :]
            
            if data.shape[1] < w:
                # Pad with zeros
                pad_w = w - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_w)), mode='constant')
            else:
                data = data[:, :w]
        
        # Normalize data to 0-1 range
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.ones_like(data) * 0.5  # Middle gray if no variation
        
        # Create RGBA texture data with colormap
        texture_data = []
        for val in data.flatten():
            # Enhanced heat colormap: blue -> cyan -> green -> yellow -> red
            if val < 0.25:
                r = 0.0
                g = 0.0
                b = 1.0
            elif val < 0.5:
                r = 0.0
                g = (val - 0.25) * 4.0
                b = 1.0
            elif val < 0.75:
                r = (val - 0.5) * 4.0
                g = 1.0
                b = 1.0 - (val - 0.5) * 4.0
            else:
                r = 1.0
                g = 1.0 - (val - 0.75) * 4.0
                b = 0.0
            texture_data.extend([r, g, b, 1.0])
        
        # Update texture
        dpg.set_value(tag, texture_data)
    
    def _update_metrics(self):
        """Update metric plots"""
        if len(self.state.loss_history) > 0:
            x_vals = list(range(len(self.state.loss_history)))
            
            # Update loss
            dpg.set_value("loss_series", [x_vals, list(self.state.loss_history)])
            
            # Update reward if available
            if len(self.state.reward_history) > 0:
                dpg.set_value("reward_series", [x_vals[:len(self.state.reward_history)], 
                                               list(self.state.reward_history)])
            
            # Update IoU if available
            if len(self.state.iou_history) > 0:
                dpg.set_value("iou_series", [x_vals[:len(self.state.iou_history)], 
                                            list(self.state.iou_history)])
    
    def _toggle_pause(self):
        """Toggle pause state"""
        self.state.is_paused = not self.state.is_paused
        dpg.set_item_label("pause_button", "Resume" if self.state.is_paused else "Pause")
    
    def _step_forward(self):
        """Step forward one iteration"""
        # This will be connected to the training loop
        pass
    
    def _update_speed(self, sender, value):
        """Update training speed multiplier"""
        self.state.speed_multiplier = value
    
    def _reset_blueprint(self):
        """Reset blueprint execution"""
        self.state.blueprint_step = 0
        self._update_blueprint_display()
    
    def _step_blueprint(self):
        """Step through blueprint execution"""
        if self.state.blueprint and self.state.blueprint_step < len(self.state.blueprint):
            self.state.blueprint_step += 1
            self._update_blueprint_display()
    
    def _update_blueprint_display(self):
        """Update blueprint visualization"""
        if self.state.blueprint:
            # Update step counter
            dpg.set_value("blueprint_step_text", 
                         f"Step: {self.state.blueprint_step}/{len(self.state.blueprint)}")
            
            # Show blueprint operations with current step highlighted
            ops_text = []
            for i, op in enumerate(self.state.blueprint):
                if i == self.state.blueprint_step:
                    ops_text.append(f"[{i}] > {self._get_op_name(op)} <")
                else:
                    ops_text.append(f"[{i}] {self._get_op_name(op)}")
            
            dpg.set_value("blueprint_ops", "\n".join(ops_text))
    
    def _get_op_name(self, op_id: int) -> str:
        """Get operation name from ID"""
        # ConstructorOps enum values start at 1, blueprint uses 0-indexed
        op_names = ["MOVE_ARM", "WRITE", "ERASE", "BRANCH_IF_EMPTY", "FORK_ARM", "HALT"]
        if op_id < len(op_names):
            return op_names[op_id]
        else:
            return f"MACRO_{op_id - len(op_names)}"
    
    def update_state(self, **kwargs):
        """Thread-safe state update"""
        self.update_queue.put(kwargs)
    
    def _process_updates(self):
        """Process queued updates"""
        while not self.update_queue.empty():
            try:
                updates = self.update_queue.get_nowait()
                
                # Update state
                for key, value in updates.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
                
                # Update UI elements
                if "current_epoch" in updates:
                    dpg.set_value("epoch_text", str(updates["current_epoch"]))
                
                if "current_task" in updates:
                    dpg.set_value("task_text", updates["current_task"])
                
                # Update grids
                if "current_input" in updates:
                    self._draw_grid("input_grid", updates["current_input"])
                
                if "current_output" in updates:
                    self._draw_grid("output_grid", updates["current_output"])
                
                if "current_prediction" in updates:
                    self._draw_grid("prediction_grid", updates["current_prediction"])
                    
                if "construction_canvas" in updates:
                    # Update construction grid with arm positions
                    self._draw_grid("construction_grid", updates["construction_canvas"],
                                  self.state.arm_positions)
                    # Initialize prediction grid if empty
                    if self.state.current_prediction is None:
                        empty_grid = torch.full_like(updates["construction_canvas"], -1)
                        self._draw_grid("prediction_grid", empty_grid)
                
                # Update neural states
                if "gnca_activations" in updates:
                    self._update_heatmap("gnca_texture", updates["gnca_activations"])
                
                if "attention_weights" in updates:
                    self._update_heatmap("attention_texture", updates["attention_weights"])
                
                if "memory_state" in updates:
                    self._update_heatmap("memory_texture", updates["memory_state"])
                
                # Update metrics
                if any(k in updates for k in ["loss_history", "reward_history", "iou_history"]):
                    self._update_metrics()
                
                # Update blueprint
                if "blueprint" in updates:
                    self._update_blueprint_display()
                
                # Draw blueprint canvas if we have construction canvas
                if "construction_canvas" in updates and self.state.blueprint:
                    self._draw_grid("blueprint_canvas", updates["construction_canvas"],
                                  self.state.arm_positions)
                
            except queue.Empty:
                break
    
    def run(self):
        """Main render loop"""
        self.is_running = True
        self.setup_ui()
        
        # Main loop
        while dpg.is_dearpygui_running() and self.is_running:
            # Process updates from training thread
            self._process_updates()
            
            # Render frame
            dpg.render_dearpygui_frame()
        
        # Cleanup
        dpg.destroy_context()
    
    def stop(self):
        """Stop the visualizer"""
        self.is_running = False


# Integration functions for the main training loop
def create_visualizer_hooks(visualizer: ARCGPUVisualizer):
    """Create hooks to integrate with training loop"""
    
    def on_task_start(task_idx: int, demo_pairs, test_in, test_out):
        """Called when starting a new task"""
        visualizer.update_state(
            current_task=f"{task_idx}/1000",
            current_input=test_in,
            current_output=test_out
        )
    
    def on_blueprint_generated(blueprint: List[int]):
        """Called when blueprint is generated"""
        visualizer.update_state(blueprint=blueprint, blueprint_step=0)
    
    def on_construction_step(canvas: torch.Tensor, arm_positions: List[Tuple[int, int]]):
        """Called during construction"""
        visualizer.update_state(
            construction_canvas=canvas,
            arm_positions=arm_positions
        )
    
    def on_neural_state_update(gnca_act=None, attention=None, memory=None):
        """Called to update neural states"""
        updates = {}
        if gnca_act is not None:
            updates["gnca_activations"] = gnca_act
        if attention is not None:
            updates["attention_weights"] = attention
        if memory is not None:
            updates["memory_state"] = memory
        
        if updates:
            visualizer.update_state(**updates)
    
    def on_training_step(loss: float, reward: float = None, iou: float = None):
        """Called after each training step"""
        visualizer.state.loss_history.append(loss)
        if reward is not None:
            visualizer.state.reward_history.append(reward)
        if iou is not None:
            visualizer.state.iou_history.append(iou)
        
        visualizer.update_state(
            loss_history=visualizer.state.loss_history,
            reward_history=visualizer.state.reward_history,
            iou_history=visualizer.state.iou_history
        )
    
    def on_epoch_complete(epoch: int):
        """Called at end of epoch"""
        visualizer.update_state(current_epoch=epoch)
    
    def should_pause():
        """Check if training should pause"""
        return visualizer.state.is_paused
    
    def get_speed_multiplier():
        """Get current speed multiplier"""
        return visualizer.state.speed_multiplier
    
    return {
        'on_task_start': on_task_start,
        'on_blueprint_generated': on_blueprint_generated,
        'on_construction_step': on_construction_step,
        'on_neural_state_update': on_neural_state_update,
        'on_training_step': on_training_step,
        'on_epoch_complete': on_epoch_complete,
        'should_pause': should_pause,
        'get_speed_multiplier': get_speed_multiplier
    }


# Example usage
if __name__ == "__main__":
    # Create and run visualizer
    viz = ARCGPUVisualizer()
    
    # In a real scenario, this would run in a separate thread
    # while the training loop runs in the main thread
    viz.run() 