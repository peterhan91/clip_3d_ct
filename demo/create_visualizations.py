#!/usr/bin/env python3
"""
Create MP4 video visualizations from preprocessed CT volumes.
This script reads the HDF5 file and creates slice-by-slice MP4 videos for each CT volume.
"""

import os
import argparse
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import matplotlib.cm as cm


def create_ct_video(volume, output_path, volume_name, fps=10, cmap='gray'):
    """
    Create an MP4 video from a 3D CT volume by animating through slices.
    
    Args:
        volume: 3D numpy array (D, H, W) representing the CT volume
        output_path: Path to save the MP4 file
        volume_name: Name of the volume for title
        fps: Frames per second for the video
        cmap: Colormap for visualization
    """
    D, H, W = volume.shape
    
    # Create temporary directory for frames
    temp_dir = Path(output_path).parent / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    # Create frames
    print(f"Creating frames for {volume_name}...")
    frame_files = []
    
    for slice_idx in tqdm(range(D), desc="Generating frames"):
        # Get slice and normalize to 0-255 range
        slice_img = volume[slice_idx, :, :]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(slice_img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(f'{volume_name} - Slice {slice_idx+1}/{D}', fontsize=14, pad=20)
        ax.axis('off')
        
        # Add slice information
        ax.text(0.02, 0.98, f'Slice: {slice_idx+1}/{D}', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save frame
        frame_file = temp_dir / f"frame_{slice_idx:04d}.png"
        plt.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        frame_files.append(str(frame_file))
    
    # Create video using OpenCV
    print(f"Creating video: {output_path}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_file in tqdm(frame_files, desc="Writing video"):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    
    # Release video writer
    video_writer.release()
    
    # Clean up temporary frames
    for frame_file in frame_files:
        os.remove(frame_file)
    temp_dir.rmdir()
    
    print(f"✓ Video saved: {output_path}")


def create_comparison_grid(volume, output_path, volume_name, num_slices=9):
    """
    Create a static grid image showing multiple slices from the CT volume.
    """
    D, H, W = volume.shape
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    # Create subplot grid
    rows = int(np.sqrt(num_slices))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    fig.suptitle(f'{volume_name} - Multi-slice View', fontsize=16, y=0.95)
    
    for idx, slice_idx in enumerate(slice_indices):
        row = idx // cols
        col = idx % cols
        
        slice_img = volume[slice_idx, :, :]
        axes[row][col].imshow(slice_img, cmap='gray', vmin=0, vmax=255)
        axes[row][col].set_title(f'Slice {slice_idx+1}/{D}', fontsize=12)
        axes[row][col].axis('off')
    
    # Hide empty subplots
    for idx in range(num_slices, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Grid image saved: {output_path}")


def process_h5_file(h5_path, output_dir, fps=10, create_grids=True):
    """
    Process HDF5 file and create visualizations for all volumes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    video_dir = output_dir / "videos"
    grid_dir = output_dir / "grids"
    video_dir.mkdir(exist_ok=True)
    if create_grids:
        grid_dir.mkdir(exist_ok=True)
    
    print(f"Reading HDF5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Get volume data and names
        volumes = f['ct_volumes'][:]
        volume_names = []
        
        if 'volume_names' in f:
            volume_names = [name.decode() if isinstance(name, bytes) else name 
                          for name in f['volume_names'][:]]
        else:
            volume_names = [f"volume_{i+1}" for i in range(len(volumes))]
        
        print(f"Found {len(volumes)} volumes in HDF5 file")
        print(f"Volume shape: {volumes[0].shape}")
        print()
        
        # Process each volume
        for i, (volume, vol_name) in enumerate(zip(volumes, volume_names)):
            print(f"Processing {i+1}/{len(volumes)}: {vol_name}")
            
            # Clean volume name for filename
            clean_name = vol_name.replace('.nii.gz', '').replace('/', '_').replace(' ', '_')
            
            # Create MP4 video
            video_path = video_dir / f"{clean_name}.mp4"
            create_ct_video(volume, str(video_path), clean_name, fps=fps)
            
            # Create grid image
            if create_grids:
                grid_path = grid_dir / f"{clean_name}_grid.png"
                create_comparison_grid(volume, str(grid_path), clean_name)
            
            print()
    
    # Create summary
    create_summary_report(h5_path, output_dir, volumes.shape, volume_names)


def create_summary_report(h5_path, output_dir, volume_shape, volume_names):
    """Create a summary report of the processing."""
    report_path = output_dir / "processing_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=== CT-RATE Demo Visualization Summary ===\n\n")
        f.write(f"Input HDF5 file: {h5_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Number of volumes: {len(volume_names)}\n")
        f.write(f"Volume dimensions: {volume_shape[1:]}\n")
        f.write(f"Data type: uint8\n\n")
        
        f.write("Processed volumes:\n")
        for i, name in enumerate(volume_names):
            f.write(f"  {i+1}. {name}\n")
        
        f.write(f"\nOutput files:\n")
        f.write(f"  Videos: {len(volume_names)} MP4 files in videos/\n")
        f.write(f"  Grids: {len(volume_names)} PNG files in grids/\n")
        
        f.write(f"\nVisualization details:\n")
        f.write(f"  - Each MP4 shows axial slices from inferior to superior\n")
        f.write(f"  - Grid images show 9 representative slices\n")
        f.write(f"  - All images use grayscale colormap\n")
        f.write(f"  - Intensity range: 0-255 (normalized from HU values)\n")
    
    print(f"✓ Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Create MP4 visualizations from CT HDF5 file")
    parser.add_argument('--h5_path', type=str,
                        default='/cbica/home/hanti/codes/clip_3d_ct/demo/demo_ct_volumes.h5',
                        help="Path to input HDF5 file")
    parser.add_argument('--output_dir', type=str,
                        default='/cbica/home/hanti/codes/clip_3d_ct/demo/visualizations',
                        help="Output directory for videos and images")
    parser.add_argument('--fps', type=int, default=10,
                        help="Frames per second for MP4 videos")
    parser.add_argument('--no_grids', action='store_true',
                        help="Skip creating grid images")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.h5_path):
        print(f"Error: HDF5 file not found: {args.h5_path}")
        print("Please run the preprocessing script first.")
        return
    
    print("=== CT-RATE Demo Visualization ===")
    print(f"Input HDF5: {args.h5_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Video FPS: {args.fps}")
    print(f"Create grids: {not args.no_grids}")
    print()
    
    # Process the file
    process_h5_file(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        fps=args.fps,
        create_grids=not args.no_grids
    )
    
    print("=== Visualization Complete ===")
    print(f"Check {args.output_dir} for:")
    print("  - videos/: MP4 animations of each CT volume")
    print("  - grids/: Multi-slice overview images")
    print("  - processing_summary.txt: Detailed summary")


if __name__ == "__main__":
    main()