"""
LAS File Visualization Function
Author: AI Assistant
Date: July 2025

Function to plot .las files with specific color coding for classes 14 and 15
"""

import numpy as np
import matplotlib.pyplot as plt
import laspy
import os
from mpl_toolkits.mplot3d import Axes3D


def plot_las_file(las_file_path, output_path=None, figsize=(12, 10), point_size=1, 
                  alpha=0.7, elevation=30, azimuth=45, title=None):
    """
    Plot a .las file with specific color coding for classes 14 and 15.
    
    Parameters:
    -----------
    las_file_path : str
        Path to the .las file to be plotted
    output_path : str, optional
        Path to save the plot. If None, the plot will be shown instead
    figsize : tuple, default (12, 10)
        Figure size (width, height) in inches
    point_size : float, default 1
        Size of the points in the plot
    alpha : float, default 0.7
        Transparency of the points (0=transparent, 1=opaque)
    elevation : float, default 30
        Elevation angle for 3D view
    azimuth : float, default 45
        Azimuth angle for 3D view
    title : str, optional
        Custom title for the plot. If None, uses filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Check if file exists
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"LAS file not found: {las_file_path}")
    
    # Read the LAS file
    try:
        las_data = laspy.read(las_file_path)
        print(f"Successfully loaded LAS file: {os.path.basename(las_file_path)}")
        print(f"Number of points: {len(las_data.x):,}")
    except Exception as e:
        raise Exception(f"Error reading LAS file: {e}")
    
    # Extract coordinates and classification
    x = las_data.x
    y = las_data.y
    z = las_data.z
    classification = las_data.classification
    
    # Get unique classes for statistics
    unique_classes = np.unique(classification)
    print(f"Unique classes found: {sorted(unique_classes)}")
    
    # Count points in classes 14 and 15
    class_14_count = np.sum(classification == 14)
    class_15_count = np.sum(classification == 15)
    total_points = len(classification)
    
    print(f"Points in class 14 (power lines): {class_14_count:,} ({100*class_14_count/total_points:.2f}%)")
    print(f"Points in class 15 (transmission towers): {class_15_count:,} ({100*class_15_count/total_points:.2f}%)")
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for each class
    colors = np.full(len(classification), 'lightgray', dtype=object)  # Default color
    
    # Color mapping for specific classes
    color_map = {
        1: 'brown',        # Class 1
        2: 'tan',          # Class 2  
        3: 'lightgreen',   # Class 3 - vegetation (light green)
        4: 'forestgreen',  # Class 4 - vegetation (medium green)
        5: 'darkgreen',    # Class 5 - vegetation (dark green)
        6: 'orange',       # Class 6
        7: 'black',        # Class 7
        8: 'yellow',       # Class 8
        11: 'pink',        # Class 11
        13: 'cyan',        # Class 13
        14: 'blue',        # Class 14 - power lines
        15: 'purple',      # Class 15 - transmission towers
        16: 'red',         # Class 16
        17: 'magenta',     # Class 17
        18: 'navy',        # Class 18
        135: 'gray'        # Class 135 - noise
    }
    
    # Apply colors based on classification
    for class_id, color in color_map.items():
        colors[classification == class_id] = color
    
    # Plot all points
    scatter = ax.scatter(x, y, z, c=colors, s=point_size, alpha=alpha)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    
    if title is None:
        title = f"LAS Point Cloud: {os.path.basename(las_file_path)}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Create custom legend for key classes
    from matplotlib.patches import Patch
    
    # Count points for each class
    class_counts = {}
    for class_id in unique_classes:
        class_counts[class_id] = np.sum(classification == class_id)
    
    # Create legend elements for the most important classes
    legend_elements = []
    
    # Always show classes 14 and 15 if they exist
    if 14 in class_counts and class_counts[14] > 0:
        legend_elements.append(Patch(facecolor='blue', label=f'Class 14 - Power Lines ({class_counts[14]:,})'))
    if 15 in class_counts and class_counts[15] > 0:
        legend_elements.append(Patch(facecolor='purple', label=f'Class 15 - Transmission Towers ({class_counts[15]:,})'))
    
    # Show vegetation classes (3, 4, 5) if they exist
    vegetation_labels = {3: 'Low Vegetation', 4: 'Medium Vegetation', 5: 'High Vegetation'}
    vegetation_colors = {3: 'lightgreen', 4: 'forestgreen', 5: 'darkgreen'}
    for class_id in [3, 4, 5]:
        if class_id in class_counts and class_counts[class_id] > 0:
            legend_elements.append(Patch(facecolor=vegetation_colors[class_id], 
                                       label=f'Class {class_id} - {vegetation_labels[class_id]} ({class_counts[class_id]:,})'))
    
    # Add other significant classes (>1% of points)
    other_class_names = {
        1: 'Unclassified', 2: 'Ground', 6: 'Building', 7: 'Low Points', 8: 'Ground Key Points',
        11: 'Air Points', 13: 'Other Ground', 16: 'Wall', 17: 'Above Building', 18: 'Other Towers', 135: 'Noise'
    }
    
    for class_id, count in class_counts.items():
        if class_id not in [3, 4, 5, 14, 15] and count > total_points * 0.01:  # Show if >1% of points
            color = color_map.get(class_id, 'lightgray')
            class_name = other_class_names.get(class_id, f'Class {class_id}')
            legend_elements.append(Patch(facecolor=color, label=f'Class {class_id} - {class_name} ({count:,})'))
    
    # Add "Other Classes" if there are remaining small classes
    other_count = sum(count for class_id, count in class_counts.items() 
                     if class_id not in [c for c in class_counts.keys() if class_counts[c] > total_points * 0.01 or c in [3, 4, 5, 14, 15]])
    if other_count > 0:
        legend_elements.append(Patch(facecolor='lightgray', label=f'Other Classes ({other_count:,})'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_las_file_2d_views(las_file_path, output_path=None, figsize=(15, 5), point_size=1, alpha=0.7):
    """
    Plot a .las file in 2D views (XY, XZ, YZ) with specific color coding for classes 14 and 15.
    
    Parameters:
    -----------
    las_file_path : str
        Path to the .las file to be plotted
    output_path : str, optional
        Path to save the plot. If None, the plot will be shown instead
    figsize : tuple, default (15, 5)
        Figure size (width, height) in inches
    point_size : float, default 1
        Size of the points in the plot
    alpha : float, default 0.7
        Transparency of the points (0=transparent, 1=opaque)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Check if file exists
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"LAS file not found: {las_file_path}")
    
    # Read the LAS file
    try:
        las_data = laspy.read(las_file_path)
        print(f"Successfully loaded LAS file: {os.path.basename(las_file_path)}")
    except Exception as e:
        raise Exception(f"Error reading LAS file: {e}")
    
    # Extract coordinates and classification
    x = las_data.x
    y = las_data.y
    z = las_data.z
    classification = las_data.classification
    
    # Define colors for each class (same as 3D function)
    colors = np.full(len(classification), 'lightgray', dtype=object)  # Default color
    
    # Color mapping for specific classes
    color_map = {
        1: 'brown',        # Class 1
        2: 'tan',          # Class 2  
        3: 'lightgreen',   # Class 3 - vegetation (light green)
        4: 'forestgreen',  # Class 4 - vegetation (medium green)
        5: 'darkgreen',    # Class 5 - vegetation (dark green)
        6: 'orange',       # Class 6
        7: 'black',        # Class 7
        8: 'yellow',       # Class 8
        11: 'pink',        # Class 11
        13: 'cyan',        # Class 13
        14: 'blue',        # Class 14 - power lines
        15: 'purple',      # Class 15 - transmission towers
        16: 'red',         # Class 16
        17: 'magenta',     # Class 17
        18: 'navy',        # Class 18
        135: 'gray'        # Class 135 - noise
    }
    
    # Apply colors based on classification
    for class_id, color in color_map.items():
        colors[classification == class_id] = color
    
    # Create subplots for 2D views
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # XY view (top view)
    axes[0].scatter(x, y, c=colors, s=point_size, alpha=alpha)
    axes[0].set_xlabel('X (meters)')
    axes[0].set_ylabel('Y (meters)')
    axes[0].set_title('Top View (XY)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # XZ view (side view)
    axes[1].scatter(x, z, c=colors, s=point_size, alpha=alpha)
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Z (meters)')
    axes[1].set_title('Side View (XZ)')
    axes[1].grid(True, alpha=0.3)
    
    # YZ view (front view)
    axes[2].scatter(y, z, c=colors, s=point_size, alpha=alpha)
    axes[2].set_xlabel('Y (meters)')
    axes[2].set_ylabel('Z (meters)')
    axes[2].set_title('Front View (YZ)')
    axes[2].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f"2D Views: {os.path.basename(las_file_path)}", fontsize=14, fontweight='bold')
    
    # Create legend for 2D views
    from matplotlib.patches import Patch
    
    # Get unique classes and their counts
    unique_classes = np.unique(classification)
    class_counts = {}
    for class_id in unique_classes:
        class_counts[class_id] = np.sum(classification == class_id)
    
    total_points = len(classification)
    
    # Create legend elements for key classes
    legend_elements = []
    
    # Always show classes 14 and 15 if they exist
    if 14 in class_counts and class_counts[14] > 0:
        legend_elements.append(Patch(facecolor='blue', label=f'Class 14 - Power Lines ({class_counts[14]:,})'))
    if 15 in class_counts and class_counts[15] > 0:
        legend_elements.append(Patch(facecolor='purple', label=f'Class 15 - Transmission Towers ({class_counts[15]:,})'))
    
    # Show vegetation classes (3, 4, 5) if they exist
    vegetation_labels = {3: 'Low Vegetation', 4: 'Medium Vegetation', 5: 'High Vegetation'}
    vegetation_colors = {3: 'lightgreen', 4: 'forestgreen', 5: 'darkgreen'}
    for class_id in [3, 4, 5]:
        if class_id in class_counts and class_counts[class_id] > 0:
            legend_elements.append(Patch(facecolor=vegetation_colors[class_id], 
                                       label=f'Class {class_id} - {vegetation_labels[class_id]} ({class_counts[class_id]:,})'))
    
    # Add other significant classes (>1% of points)
    other_class_names = {
        1: 'Unclassified', 2: 'Ground', 6: 'Building', 7: 'Low Points', 8: 'Ground Key Points',
        11: 'Air Points', 13: 'Other Ground', 16: 'Wall', 17: 'Above Building', 18: 'Other Towers', 135: 'Noise'
    }
    
    for class_id, count in class_counts.items():
        if class_id not in [3, 4, 5, 14, 15] and count > total_points * 0.01:  # Show if >1% of points
            color = color_map.get(class_id, 'lightgray')
            class_name = other_class_names.get(class_id, f'Class {class_id}')
            legend_elements.append(Patch(facecolor=color, label=f'Class {class_id} - {class_name} ({count:,})'))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_elements), 4), fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Save or show the plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"2D views plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_multiple_las_files(las_file_paths, output_dir, plot_type='3d', **kwargs):
    """
    Plot multiple .las files and save them to a directory.
    
    Parameters:
    -----------
    las_file_paths : list
        List of paths to .las files
    output_dir : str
        Directory to save the plots
    plot_type : str, default '3d'
        Type of plot ('3d' or '2d')
    **kwargs : dict
        Additional arguments to pass to the plotting function
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for las_file_path in las_file_paths:
        try:
            filename = os.path.basename(las_file_path).replace('.las', '')
            output_path = os.path.join(output_dir, f"{filename}_{plot_type}_plot.png")
            
            if plot_type == '3d':
                plot_las_file(las_file_path, output_path, **kwargs)
            elif plot_type == '2d':
                plot_las_file_2d_views(las_file_path, output_path, **kwargs)
            else:
                raise ValueError("plot_type must be '3d' or '2d'")
                
        except Exception as e:
            print(f"Error plotting {las_file_path}: {e}")


# Example usage
if __name__ == "__main__":
    # Example usage of the functions
    
    # Single file 3D plot
    # las_file = "/path/to/your/file.las"
    # plot_las_file(las_file, output_path="output_3d.png")
    
    # Single file 2D views
    # plot_las_file_2d_views(las_file, output_path="output_2d.png")
    
    # Multiple files
    # las_files = ["/path/to/file1.las", "/path/to/file2.las"]
    # plot_multiple_las_files(las_files, "output_directory/", plot_type='3d')
    
    print("LAS visualization functions loaded successfully!")
    print("\nUsage examples:")
    print("1. plot_las_file('path/to/file.las', 'output.png')")
    print("2. plot_las_file_2d_views('path/to/file.las', 'output_2d.png')")
    print("3. plot_multiple_las_files(['file1.las', 'file2.las'], 'output_dir/', '3d')")
