"""
Interactive Graph Data Extraction Tool
Click on points in the graph image and specify their actual values.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GraphExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.img_array = np.array(self.img)
        
        self.calibration_points = []
        self.data_points = []
        self.clicks = []
        
        self.fig = None
        self.ax = None
        
        self.x_slope = None
        self.x_intercept = None
        self.y_slope = None
        self.y_intercept = None
        
        self.current_mode = None
        
    def onclick(self, event):
        """Handle mouse clicks on the image."""
        if event.inaxes != self.ax:
            return
        
        x_pixel, y_pixel = event.xdata, event.ydata
        
        # Mark the clicked point
        self.ax.plot(x_pixel, y_pixel, 'rx', markersize=15, markeredgewidth=3)
        self.fig.canvas.draw()
        
        print(f"\nClicked at pixel coordinates: X={x_pixel:.1f}, Y={y_pixel:.1f}")
        self.clicks.append((x_pixel, y_pixel))
        
    def get_float_input(self, prompt):
        """Get validated float input from user."""
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def calibrate_axes(self):
        """Calibrate axes by clicking on two reference corners."""
        print("\n" + "="*60)
        print("STEP 1: CALIBRATE AXES")
        print("="*60)
        print("\nClick on TWO corners of your plot area to calibrate.")
        print("Typically: bottom-left corner and top-right corner")
        print("(or any two opposite corners where you know the exact values)")
        print("\nClick TWICE on the image, then close the window.")
        
        input("\nPress Enter to start clicking...")
        
        self.clicks = []
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(self.img_array)
        self.ax.set_title('CLICK on TWO CORNERS (e.g., bottom-left, then top-right), then close window', 
                         fontsize=14, fontweight='bold', color='red')
        self.ax.set_xlabel('Pixel X')
        self.ax.set_ylabel('Pixel Y')
        
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.tight_layout()
        plt.show()
        
        if len(self.clicks) < 2:
            print(f"Error: Only {len(self.clicks)} click(s) detected. Need 2 corners. Exiting.")
            return False
        
        # Get the first two clicks
        x_pixel_1, y_pixel_1 = self.clicks[0]
        x_pixel_2, y_pixel_2 = self.clicks[1]
        
        print(f"\nCorner 1: Pixel X={x_pixel_1:.1f}, Y={y_pixel_1:.1f}")
        x_value_1 = self.get_float_input("  Enter the WAVELENGTH at corner 1: ")
        y_value_1 = self.get_float_input("  Enter the TRANSMISSION at corner 1: ")
        
        print(f"\nCorner 2: Pixel X={x_pixel_2:.1f}, Y={y_pixel_2:.1f}")
        x_value_2 = self.get_float_input("  Enter the WAVELENGTH at corner 2: ")
        y_value_2 = self.get_float_input("  Enter the TRANSMISSION at corner 2: ")
        
        # Calculate transformation coefficients
        self.x_slope = (x_value_2 - x_value_1) / (x_pixel_2 - x_pixel_1)
        self.x_intercept = x_value_1 - self.x_slope * x_pixel_1
        
        self.y_slope = (y_value_2 - y_value_1) / (y_pixel_2 - y_pixel_1)
        self.y_intercept = y_value_1 - self.y_slope * y_pixel_1
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print(f"Wavelength: λ = {self.x_slope:.6f} * X_pixel + {self.x_intercept:.6f}")
        print(f"Transmission: T = {self.y_slope:.6f} * Y_pixel + {self.y_intercept:.6f}")
        print("="*60)
        
        return True
    
    def extract_curve_points(self):
        """Extract data points by clicking all points at once - auto-calculates values."""
        print("\n" + "="*60)
        print("STEP 2: EXTRACT DATA POINTS FROM THE CURVE")
        print("="*60)
        print("\nNow click on ALL points along the curve in one go.")
        print("The wavelength and transmission will be calculated automatically.")
        
        num_points = int(self.get_float_input("\nHow many points do you want to extract? "))
        
        print(f"\nClick on {num_points} points on the curve, then close the window.")
        input("Press Enter to start clicking...")
        
        self.clicks = []
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(self.img_array)
        self.ax.set_title(f'CLICK on {num_points} points along the curve, then close window', 
                        fontsize=14, fontweight='bold', color='red')
        self.ax.set_xlabel('Pixel X')
        self.ax.set_ylabel('Pixel Y')
        
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.tight_layout()
        plt.show()
        
        if len(self.clicks) == 0:
            print("No clicks detected. Exiting.")
            return None, None, None, None
        
        print(f"\n✓ Captured {len(self.clicks)} points")
        
        # Calculate wavelength and transmission for all points
        pixel_x_data = []
        pixel_y_data = []
        wavelength_data = []
        transmission_data = []
        
        print("\nCalculated values:")
        print("-" * 60)
        for i, (px, py) in enumerate(self.clicks, 1):
            wavelength = self.x_slope * px + self.x_intercept
            transmission = self.y_slope * py + self.y_intercept
            
            pixel_x_data.append(px)
            pixel_y_data.append(py)
            wavelength_data.append(wavelength)
            transmission_data.append(transmission)
            
            print(f"Point {i:2d}: λ={wavelength:8.3f}, T={transmission:6.3f}")
        
        return np.array(wavelength_data), np.array(transmission_data), pixel_x_data, pixel_y_data
    
    def fit_and_plot(self, wavelength_data, transmission_data, pixel_x_data, pixel_y_data):
        """Show extracted points with labels and save to CSV."""
        print("\n" + "="*60)
        print("STEP 3: DISPLAY AND SAVE RESULTS")
        print("="*60)
        
        # Sort by wavelength
        sorted_indices = np.argsort(wavelength_data)
        wavelength_sorted = wavelength_data[sorted_indices]
        transmission_sorted = transmission_data[sorted_indices]
        pixel_x_sorted = [pixel_x_data[i] for i in sorted_indices]
        pixel_y_sorted = [pixel_y_data[i] for i in sorted_indices]
        
        # Display table
        print("\nExtracted Data (sorted by wavelength):")
        print("-" * 60)
        print(f"{'Point':<8}{'Wavelength':<15}{'Transmission':<15}")
        print("-" * 60)
        for i, (wl, tr) in enumerate(zip(wavelength_sorted, transmission_sorted), 1):
            print(f"{i:<8}{wl:<15.4f}{tr:<15.6f}")
        print("-" * 60)
        
        # Save to CSV immediately
        output_file = 'extracted_transmission_data.csv'
        with open(output_file, 'w') as f:
            f.write("wavelength,transmission\n")
            for wl, tr in zip(wavelength_sorted, transmission_sorted):
                f.write(f"{wl},{tr}\n")
        print(f"\n✓ Data saved to {output_file}")
        
        # Plot: Original image with X marks and text labels
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(self.img_array)
        
        # Plot X marks
        ax.scatter(pixel_x_data, pixel_y_data, color='red', s=200, 
                  marker='x', linewidth=3, label='Extracted points', zorder=5)
        
        # Add text labels with wavelength and transmission
        for px, py, wl, tr in zip(pixel_x_data, pixel_y_data, wavelength_data, transmission_data):
            label_text = f'λ={wl:.2f}\nT={tr:.3f}'
            ax.text(px, py - 20, label_text, color='yellow', fontsize=9, 
                   fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='red'))
        
        ax.set_xlabel('Pixel X', fontsize=12)
        ax.set_ylabel('Pixel Y', fontsize=12)
        ax.set_title(f'Original Graph with {len(pixel_x_data)} Extracted Points', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE!")
        print("="*60)

def main():
    image_path = 'emccd_QE.png'
    
    try:
        extractor = GraphExtractor(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find {image_path}")
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE GRAPH DATA EXTRACTION")
    print("="*60)
    print("\nThis tool will help you extract transmission vs wavelength data")
    print("from your graph image by clicking on points.")
    
    # Calibrate axes
    if not extractor.calibrate_axes():
        return
    
    # Extract data points
    wavelength_data, transmission_data, pixel_x_data, pixel_y_data = extractor.extract_curve_points()
    
    if wavelength_data is None or len(wavelength_data) == 0:
        print("\nError: No data points extracted.")
        return
    
    # Display and save
    extractor.fit_and_plot(wavelength_data, transmission_data, pixel_x_data, pixel_y_data)

if __name__ == "__main__":
    main()
