import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

def generate_paint_by_numbers(image_path, num_colors=24, min_region_size=100, output_dir='.'):
    # Step 1: Load and preprocess image
    img = mpimg.imread(image_path)
    if img.shape[2] == 4:  # Remove alpha if present
        img = img[:, :, :3]
    # Resize while preserving aspect ratio (target max dimension = 800)
    max_dimension = 800
    aspect_ratio = img.shape[1] / img.shape[0]  # width / height
    if aspect_ratio > 1:  # Landscape image
        new_width = max_dimension
        new_height = int(max_dimension / aspect_ratio)
    else:  # Portrait or square image
        new_height = max_dimension
        new_width = int(max_dimension * aspect_ratio)
    img_resized = ndimage.zoom(img, (new_height/img.shape[0], new_width/img.shape[1], 1), order=1)
    # Sharpen and enhance contrast (simple filter)
    sharpened = ndimage.gaussian_filter(img_resized, sigma=1)
    contrast_enhanced = np.clip((img_resized - sharpened) * 1.5 + sharpened, 0, 1)
    
    # Step 2: Color reduction with k-means
    pixels = contrast_enhanced.reshape(-1, 3)
    centroids, _ = kmeans(pixels, num_colors)
    labels, _ = vq(pixels, centroids)
    quantized = centroids[labels].reshape(contrast_enhanced.shape)
    
    # Step 3: Region segmentation
    labeled_regions, num_regions = ndimage.label(quantized[:,:,0] > -1)  # Dummy label; refine with color diffs
    # Merge small regions
    sizes = ndimage.sum(np.ones_like(labeled_regions), labeled_regions, index=np.arange(num_regions + 1))
    small_mask = sizes < min_region_size
    for i in np.where(small_mask)[0]:
        labeled_regions[labeled_regions == i] = 0  # Merge to background or nearest; simplify here
    
    # Relabel after merging
    labeled_regions, num_regions = ndimage.label(labeled_regions > 0)
    
    # Step 4: Generate outputs
    # Reference image
    plt.imsave(f'{output_dir}/reference.png', quantized)
    
    # Outline with numbers
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.ones_like(quantized), cmap='gray')  # White background
    edges = ndimage.sobel(np.mean(quantized, axis=2))
    ax.contour(edges, levels=[0.1], colors='black', linewidths=0.5)
    for region_id in range(1, num_regions + 1):
        mask = labeled_regions == region_id
        if np.sum(mask) > 0:
            centroid = ndimage.center_of_mass(mask)
            ax.text(centroid[1], centroid[0], str(region_id), fontsize=8, color='red', ha='center')
    ax.axis('off')
    plt.savefig(f'{output_dir}/numbered_outline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Palette legend
    fig, ax = plt.subplots(figsize=(4, num_colors * 0.5))
    for i, color in enumerate(centroids):
        ax.add_patch(Rectangle((0, i), 1, 1, color=color))
        ax.text(1.1, i + 0.5, f'Color {i+1}: RGB{np.round(color*255).astype(int)}', va='center')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, num_colors)
    ax.axis('off')
    plt.savefig(f'{output_dir}/palette_legend.png', bbox_inches='tight')
    plt.close()
    
    print(f'Paint by numbers package saved to {output_dir}: reference.png, numbered_outline.png, palette_legend.png')
    print(f'Use {num_colors} colors; print numbered_outline.png on canvas at 300 DPI.')

# Example usage: generate_paint_by_numbers('path/to/your/image.jpg', num_colors=24)
if __name__ == "__main__":
    generate_paint_by_numbers(r"C:\Users\Kynde\OneDrive\Desktop\PaintByNumbers\EiffelTower_ChiChi.png", num_colors=24, output_dir='output')
