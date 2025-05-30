import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pywt
import os

class ImageEnhancementAnalyzer:
    def __init__(self, image_path):
        """
        Initialize the analyzer with an image
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("Could not load image. Please check the path.")
        
        # Convert BGR to RGB for display
        self.original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Create output directory for saving figures
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_image_properties(self):
        """
        Analyze basic image properties and create visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Analisis Citra - Properti Gambar', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(self.original_rgb)
        axes[0, 0].set_title('Gambar Asli')
        axes[0, 0].axis('off')
        
        # Grayscale
        axes[0, 1].imshow(self.image_gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # Histogram RGB
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.original_rgb], [i], None, [256], [0, 256])
            axes[0, 2].plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} Channel')
        axes[0, 2].set_title('Histogram RGB')
        axes[0, 2].set_xlabel('Pixel Intensity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Histogram Grayscale
        hist_gray = cv2.calcHist([self.image_gray], [0], None, [256], [0, 256])
        axes[1, 0].plot(hist_gray, color='black')
        axes[1, 0].set_title('Histogram Grayscale')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Edge detection (Canny)
        edges = cv2.Canny(self.image_gray, 50, 150)
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Edge Detection (Canny)')
        axes[1, 1].axis('off')
        
        # Image statistics
        stats_text = f"""
        Dimensi: {self.original_rgb.shape[1]} x {self.original_rgb.shape[0]}
        Channels: {self.original_rgb.shape[2]}
        
        Statistik Intensitas:
        Mean: {np.mean(self.image_gray):.2f}
        Std: {np.std(self.image_gray):.2f}
        Min: {np.min(self.image_gray)}
        Max: {np.max(self.image_gray)}
        
        Estimasi Noise Level:
        {self.estimate_noise_level():.2f}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                        verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Statistik Gambar')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_image_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def estimate_noise_level(self):
        """
        Estimate noise level using Laplacian variance method
        """
        laplacian_var = cv2.Laplacian(self.image_gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_metrics(self, original, processed):
        """
        Calculate image quality metrics
        """
        # Ensure images are in the same data type and range
        if original.dtype != processed.dtype:
            processed = processed.astype(original.dtype)
        
        # PSNR
        psnr = peak_signal_noise_ratio(original, processed)
        
        # SSIM - updated for newer scikit-image versions
        try:
            # Try new parameter name first
            if len(original.shape) == 3:
                ssim = structural_similarity(original, processed, channel_axis=2)
            else:
                ssim = structural_similarity(original, processed)
        except TypeError:
            # Fallback to older parameter name
            if len(original.shape) == 3:
                ssim = structural_similarity(original, processed, multichannel=True)
            else:
                ssim = structural_similarity(original, processed)
        
        # Mean Squared Error
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        
        # Signal-to-Noise Ratio estimation
        signal_power = np.mean(processed.astype(float) ** 2)
        noise_power = mse
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'SNR': snr
        }
    
    def gaussian_smoothing(self, sigma=1.0):
        """
        Apply Gaussian smoothing
        """
        # Apply to RGB image
        smoothed_rgb = np.zeros_like(self.original_rgb)
        for i in range(3):
            smoothed_rgb[:, :, i] = ndimage.gaussian_filter(self.original_rgb[:, :, i], sigma=sigma)
        
        return smoothed_rgb.astype(np.uint8)
    
    def median_filtering(self, kernel_size=5):
        """
        Apply median filtering
        """
        filtered_rgb = np.zeros_like(self.original_rgb)
        for i in range(3):
            filtered_rgb[:, :, i] = ndimage.median_filter(self.original_rgb[:, :, i], size=kernel_size)
        
        return filtered_rgb.astype(np.uint8)
    
    def bilateral_filtering(self, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filtering
        """
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(self.original_rgb, cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.bilateralFilter(bgr_image, d, sigma_color, sigma_space)
        # Convert back to RGB
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        return filtered_rgb
    
    def wavelet_denoising(self, wavelet='db4', mode='soft'):
        """
        Apply wavelet denoising
        """
        denoised_rgb = np.zeros_like(self.original_rgb)
        
        for i in range(3):
            # Convert to float [0, 1]
            channel = self.original_rgb[:, :, i].astype(float) / 255.0
            
            # Estimate noise level - updated for newer scikit-image versions
            try:
                # Try new parameter name first
                sigma = restoration.estimate_sigma(channel, channel_axis=None, average_sigmas=True)
            except TypeError:
                try:
                    # Fallback to older parameter name
                    sigma = restoration.estimate_sigma(channel, multichannel=False, average_sigmas=True)
                except TypeError:
                    # If both fail, use simple estimation
                    sigma = 0.1
            
            # Apply wavelet denoising
            try:
                denoised_channel = restoration.denoise_wavelet(
                    channel, method='BayesShrink', mode=mode, 
                    wavelet=wavelet, rescale_sigma=True
                )
            except:
                # Fallback method using pywt directly
                coeffs = pywt.wavedec2(channel, wavelet, level=4)
                threshold = sigma * np.sqrt(2 * np.log(channel.size))
                coeffs_thresh = coeffs[0], tuple([pywt.threshold(detail, threshold, mode) for detail in coeffs[1:]])
                denoised_channel = pywt.waverec2(coeffs_thresh, wavelet)
            
            # Convert back to uint8
            denoised_rgb[:, :, i] = np.clip(denoised_channel * 255, 0, 255)
        
        return denoised_rgb.astype(np.uint8)
    
    def compare_noise_reduction_techniques(self):
        """
        Compare all noise reduction techniques
        """
        # Apply all techniques
        gaussian_result = self.gaussian_smoothing(sigma=1.0)
        median_result = self.median_filtering(kernel_size=5)
        bilateral_result = self.bilateral_filtering()
        wavelet_result = self.wavelet_denoising()
        
        # Calculate metrics for each technique
        techniques = {
            'Original': self.original_rgb,
            'Gaussian Smoothing': gaussian_result,
            'Median Filtering': median_result,
            'Bilateral Filtering': bilateral_result,
            'Wavelet Denoising': wavelet_result
        }
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Perbandingan Teknik Noise Reduction', fontsize=16, fontweight='bold')
        
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        metrics_data = {}
        
        for i, (name, image) in enumerate(techniques.items()):
            if i < len(positions):
                row, col = positions[i]
                axes[row, col].imshow(image)
                axes[row, col].set_title(name, fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
                
                # Calculate metrics (compare with original for processed images)
                if name != 'Original':
                    metrics = self.calculate_metrics(self.original_rgb, image)
                    metrics_data[name] = metrics
                    
                    # Add metrics text
                    metrics_text = f"PSNR: {metrics['PSNR']:.2f} dB\nSSIM: {metrics['SSIM']:.3f}"
                    axes[row, col].text(0.02, 0.98, metrics_text, 
                                      transform=axes[row, col].transAxes,
                                      verticalalignment='top', fontsize=9,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide the last subplot if not used
        if len(techniques) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_noise_reduction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create metrics comparison chart
        self.plot_metrics_comparison(metrics_data)
        
        return techniques, metrics_data
    
    def plot_metrics_comparison(self, metrics_data):
        """
        Create a detailed metrics comparison chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Perbandingan Metrik Kualitas Gambar', fontsize=16, fontweight='bold')
        
        techniques = list(metrics_data.keys())
        
        # PSNR comparison
        psnr_values = [metrics_data[tech]['PSNR'] for tech in techniques]
        bars1 = axes[0, 0].bar(techniques, psnr_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[0, 0].set_title('Peak Signal-to-Noise Ratio (PSNR)')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, psnr_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # SSIM comparison
        ssim_values = [metrics_data[tech]['SSIM'] for tech in techniques]
        bars2 = axes[0, 1].bar(techniques, ssim_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[0, 1].set_title('Structural Similarity Index (SSIM)')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        for bar, value in zip(bars2, ssim_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # MSE comparison
        mse_values = [metrics_data[tech]['MSE'] for tech in techniques]
        bars3 = axes[1, 0].bar(techniques, mse_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 0].set_title('Mean Squared Error (MSE)')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, mse_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # SNR comparison
        snr_values = [metrics_data[tech]['SNR'] for tech in techniques]
        bars4 = axes[1, 1].bar(techniques, snr_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_title('Signal-to-Noise Ratio (SNR)')
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, snr_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detailed_analysis_per_technique(self, techniques, metrics_data):
        """
        Create detailed analysis for each technique
        """
        # Create before/after comparison for each technique
        technique_names = ['Gaussian Smoothing', 'Median Filtering', 'Bilateral Filtering', 'Wavelet Denoising']
        
        for i, technique_name in enumerate(technique_names):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Analisis Detail: {technique_name}', fontsize=16, fontweight='bold')
            
            # Original
            axes[0].imshow(self.original_rgb)
            axes[0].set_title('Gambar Asli')
            axes[0].axis('off')
            
            # Processed
            processed_image = techniques[technique_name]
            axes[1].imshow(processed_image)
            axes[1].set_title(f'Setelah {technique_name}')
            axes[1].axis('off')
            
            # Difference
            diff = np.abs(self.original_rgb.astype(float) - processed_image.astype(float))
            diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
            axes[2].imshow(diff_normalized)
            axes[2].set_title('Perbedaan (Amplified)')
            axes[2].axis('off')
            
            # Add metrics text
            metrics = metrics_data[technique_name]
            metrics_text = f"""
            Metrik Kualitas:
            PSNR: {metrics['PSNR']:.2f} dB
            SSIM: {metrics['SSIM']:.3f}
            MSE: {metrics['MSE']:.2f}
            SNR: {metrics['SNR']:.2f} dB
            """
            
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/04_{technique_name.lower().replace(" ", "_")}_detailed.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_report(self, metrics_data):
        """
        Generate a summary report of the analysis
        """
        print("\n" + "="*80)
        print("LAPORAN ANALISIS IMAGE ENHANCEMENT")
        print("="*80)
        
        print(f"\nInformasi Gambar Asli:")
        print(f"- Dimensi: {self.original_rgb.shape[1]} x {self.original_rgb.shape[0]} pixels")
        print(f"- Channels: {self.original_rgb.shape[2]} (RGB)")
        print(f"- Estimasi Noise Level: {self.estimate_noise_level():.2f}")
        
        print(f"\nPerbandingan Teknik Noise Reduction:")
        print("-" * 60)
        
        # Find best technique for each metric
        best_psnr = max(metrics_data.items(), key=lambda x: x[1]['PSNR'])
        best_ssim = max(metrics_data.items(), key=lambda x: x[1]['SSIM'])
        best_mse = min(metrics_data.items(), key=lambda x: x[1]['MSE'])  # Lower is better
        best_snr = max(metrics_data.items(), key=lambda x: x[1]['SNR'])
        
        for technique, metrics in metrics_data.items():
            print(f"\n{technique}:")
            print(f"  PSNR: {metrics['PSNR']:.2f} dB {'🏆' if technique == best_psnr[0] else ''}")
            print(f"  SSIM: {metrics['SSIM']:.3f} {'🏆' if technique == best_ssim[0] else ''}")
            print(f"  MSE:  {metrics['MSE']:.2f} {'🏆' if technique == best_mse[0] else ''}")
            print(f"  SNR:  {metrics['SNR']:.2f} dB {'🏆' if technique == best_snr[0] else ''}")
        
        print(f"\nRekomendasi:")
        print(f"- Terbaik untuk PSNR: {best_psnr[0]} ({best_psnr[1]['PSNR']:.2f} dB)")
        print(f"- Terbaik untuk SSIM: {best_ssim[0]} ({best_ssim[1]['SSIM']:.3f})")
        print(f"- Terbaik untuk MSE: {best_mse[0]} ({best_mse[1]['MSE']:.2f})")
        print(f"- Terbaik untuk SNR: {best_snr[0]} ({best_snr[1]['SNR']:.2f} dB)")
        
        print(f"\nKesimpulan:")
        print("- Bilateral Filtering: Baik untuk mempertahankan edge sambil mengurangi noise")
        print("- Gaussian Smoothing: Efektif untuk noise reduction tapi bisa blur detail")
        print("- Median Filtering: Excellent untuk salt-and-pepper noise")
        print("- Wavelet Denoising: Advanced technique dengan preservasi detail yang baik")
        
        print("\n" + "="*80)

def main():
    """
    Main function to run the image enhancement analysis
    """
    # Load and analyze image
    print("Image Enhancement Analysis - Noise Reduction Comparison")
    print("="*60)
    
    image_path = './images/kuda.jpg'  
      
    try:
        # Initialize analyzer
        analyzer = ImageEnhancementAnalyzer(image_path)
        
        print("1. Analyzing image properties...")
        analyzer.analyze_image_properties()
        
        print("2. Comparing noise reduction techniques...")
        techniques, metrics_data = analyzer.compare_noise_reduction_techniques()
        
        print("3. Creating detailed analysis...")
        analyzer.detailed_analysis_per_technique(techniques, metrics_data)
        
        print("4. Generating summary report...")
        analyzer.generate_summary_report(metrics_data)
        
        print(f"\nAll results saved in '{analyzer.output_dir}' directory")
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the image path is correct and the image file exists.")

def create_demo_image():
    """
    Create a demo noisy image for testing purposes
    """
    print("Creating demo image...")
    
    # Create a simple synthetic image
    height, width = 400, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some geometric shapes
    cv2.rectangle(image, (50, 50), (200, 200), (100, 150, 200), -1)
    cv2.circle(image, (400, 150), 80, (200, 100, 150), -1)
    cv2.ellipse(image, (300, 300), (100, 50), 45, 0, 360, (150, 200, 100), -1)
    
    # Add gradient background
    for i in range(height):
        for j in range(width):
            if image[i, j, 0] == 0:  # If pixel is black (background)
                image[i, j] = [int(50 + 100 * i / height), 
                              int(30 + 80 * j / width), 
                              int(70 + 60 * (i + j) / (height + width))]
    
    # Add noise
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Save demo image
    cv2.imwrite('demo_noisy_image.jpg', cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
    print("Demo image 'demo_noisy_image.jpg' created successfully!")

if __name__ == "__main__":
    main()