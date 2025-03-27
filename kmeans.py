import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class KMeansVisualizer:
    def __init__(self, n_clusters=4, max_iter=20, random_state=0, sub_steps=5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.sub_steps = sub_steps
        self.history = {'centroids': [], 'labels': [], 'inertia': []}
        
    def _init_centroids(self, X):
        # Initialize centroids to be in terrible locations - all far from data
        # and far from each other
        np.random.seed(self.random_state)
        
        # Find data boundaries
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        data_range = max_vals - min_vals
        
        # Create a set of terrible initial centroids
        centroids = []
        
        # Place centroids in different corners outside the data
        for i in range(self.n_clusters):
            # Create different bad positions in different areas
            corner_position = np.zeros(2)
            
            # Choose different corners for each centroid
            corner_position[0] = min_vals[0] - data_range[0] * (i % 2)
            corner_position[1] = min_vals[1] - data_range[1] * ((i // 2) % 2)
            
            # Add some randomness
            corner_position += np.random.randn(2) * data_range * 0.1
            
            centroids.append(corner_position)
        
        return np.array(centroids)
        
    def fit(self, X):
        # Reset history
        self.history = {'centroids': [], 'labels': [], 'inertia': []}
        
        # Initialize centroids
        centers = self._init_centroids(X)
        
        # Capture initial state
        distances = np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances**2, axis=1))
        
        self.history['centroids'].append(centers.copy())
        self.history['labels'].append(labels.copy())
        self.history['inertia'].append(inertia)
        
        # Run K-means iterations
        converged = False
        for i in range(self.max_iter):
            if converged:
                break
                
            # Create a copy of current centers for comparison
            old_centers = centers.copy()
                
            # Update centroids
            new_centers = np.zeros_like(centers)
            for j in range(self.n_clusters):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = np.mean(X[mask], axis=0)
                else:
                    # If a cluster is empty, place it randomly within the data
                    new_centers[j] = X[np.random.randint(0, X.shape[0])]
            
            # Add intermediate steps for smoother animation
            for step in range(1, self.sub_steps + 1):
                # Interpolate between old and new centers
                alpha = step / self.sub_steps
                interp_centers = old_centers * (1 - alpha) + new_centers * alpha
                
                # Reassign points to clusters with interpolated centers
                distances = np.linalg.norm(X[:, np.newaxis, :] - interp_centers, axis=2)
                interp_labels = np.argmin(distances, axis=1)
                interp_inertia = np.sum(np.min(distances**2, axis=1))
                
                # Store this sub-step
                self.history['centroids'].append(interp_centers.copy())
                self.history['labels'].append(interp_labels.copy())
                self.history['inertia'].append(interp_inertia)
            
            # Apply the centroid update
            centers = new_centers.copy()
            
            # Final reassignment with new centers
            distances = np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2)
            new_labels = np.argmin(distances, axis=1)
            inertia = np.sum(np.min(distances**2, axis=1))
            
            # Check for convergence - no label changes
            if np.array_equal(labels, new_labels):
                print(f"Converged after {i+1} iterations")
                converged = True
            
            labels = new_labels.copy()
            
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        
        return self
    
    def visualize(self, X, interval=100, save_path=None):
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up color map for consistency
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))
        
        def update(frame):
            ax.clear()
            centers = self.history['centroids'][frame]
            labels = self.history['labels'][frame]
            inertia = self.history['inertia'][frame]
            
            # Plot data points
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7)
            
            # Plot centroids with matching colors but different marker
            for i, center in enumerate(centers):
                ax.scatter(center[0], center[1], c=[colors[i]], 
                          marker='X', s=200, edgecolor='black', linewidth=2)
            
            # Plot lines from points to centroids (with matching colors)
            for i, (point, label) in enumerate(zip(X, labels)):
                ax.plot([point[0], centers[label, 0]], 
                       [point[1], centers[label, 1]], 
                       c=colors[label], alpha=0.05)
            
            # Add iteration info
            real_iter = frame // (self.sub_steps + 1)
            sub_step = frame % (self.sub_steps + 1)
            title = f'Iteration {real_iter}'
            if sub_step > 0:
                title += f' (Step {sub_step}/{self.sub_steps})'
            title += f' - Inertia: {inertia:.2f}'
            
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Set consistent axis limits that include both data and all centroids
            # throughout the entire animation
            if frame == 0:
                # On first frame, calculate bounds that work for all frames
                all_centroids = np.vstack(self.history['centroids'])
                all_points = np.vstack([X, all_centroids])
                x_min, y_min = np.min(all_points, axis=0) - 1
                x_max, y_max = np.max(all_points, axis=0) + 1
                
                # Store bounds as figure attributes
                fig.x_bounds = (x_min, x_max)
                fig.y_bounds = (y_min, y_max)
            
            # Use the stored bounds
            ax.set_xlim(fig.x_bounds)
            ax.set_ylim(fig.y_bounds)
        
        # Create animation
        animation = FuncAnimation(fig, update, frames=len(self.history['centroids']), 
                                 interval=interval, repeat=True)
        
        if save_path:
            animation.save(save_path)
        
        # Show the animation in a window
        plt.show()
        
        return animation

# Generate challenging data with clear clusters
def generate_challenging_data(random_state=0):
    np.random.seed(random_state)
    
    # Create three distinct clusters
    n_samples = 500
    
    # First cluster: center
    cluster1 = np.random.randn(n_samples // 3, 2) * 1.0
    
    # Second cluster: to the right
    cluster2 = np.random.randn(n_samples // 3, 2) * 1.5 + np.array([8, 0])
    
    # Third cluster: above
    cluster3 = np.random.randn(n_samples // 3, 2) * 1.2 + np.array([4, 8])
    
    # Combine data
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Shuffle
    np.random.shuffle(X)
    
    return X

# Create and run the visualizer
if __name__ == "__main__":
    # Generate data
    X = generate_challenging_data(random_state=42)
    
    print("Running K-means algorithm with visualization...")
    kmeans_viz = KMeansVisualizer(n_clusters=3, max_iter=20, random_state=42, sub_steps=3)
    kmeans_viz.fit(X)
    
    print(f"Visualization includes {len(kmeans_viz.history['centroids'])} frames")
    animation = kmeans_viz.visualize(X, interval=200)
    
    print("Done!")