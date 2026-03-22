# Import necessary libraries
import nibabel as nib  # For handling NIfTI files
import numpy as np  # For numerical operations
import vtk  # For 3D visualization
from vtk.util import numpy_support  # For converting between NumPy and VTK arrays
from nilearn import masking  # For brain extraction
import tensorflow as tf  # For deep learning
from tensorflow.keras.models import load_model  # To load saved models
from tensorflow.keras.layers import Layer, Dense  # For custom layers
from tensorflow.keras.utils import register_keras_serializable  # For custom layer serialization
from scipy.ndimage import binary_dilation  # For morphological operations
from skimage.transform import resize  # For image resizing
from scipy.ndimage import zoom  # For image zooming

# Function to load and preprocess NIfTI files
def load_and_process_nifti(nifti_file_path, is_masked=False):
    # Load NIfTI file
    nifti_image = nib.load(nifti_file_path)
    volume = nifti_image.get_fdata()  # Get volume data
    if np.all(volume == 0):
        print("Detected zero volume - reorienting...")
        volume = np.rot90(volume, k=2, axes=(0,1))  # Rotate if empty
    
    # Get image metadata
    spacing = nifti_image.header.get_zooms()  # Voxel spacing
    affine = nifti_image.affine  # Affine transformation matrix

    print("Voxel spacing (x, y, z):", spacing)
    print("Affine matrix:\n", affine)

    # Brain extraction if needed
    if is_masked:
        print("Dataset is already masked. Skipping brain extraction.")
        brain_volume = volume
    else:
        print("Dataset is not masked. Performing brain extraction using nilearn.")
        try:
            # Compute brain mask using nilearn
            brain_mask = masking.compute_brain_mask(nifti_file_path)
            brain_volume = volume * brain_mask.get_fdata()  # Apply mask
            if np.sum(brain_volume) == 0:
                raise ValueError("Brain mask is empty. Check the input dataset.")
        except Exception as e:
            print(f"Error computing brain mask: {e}")
            print("Falling back to simple thresholding.")
            threshold = np.percentile(volume, 50)
            brain_mask = volume > threshold
            brain_volume = volume * brain_mask

    return brain_volume, spacing, affine

# Function to apply segmentation model to volume
def apply_segmentation(model, volume):
    """Apply segmentation model to the volume with proper preprocessing"""
    try:
        # Check model input shape requirements
        target_shape = model.input_shape[1:-1]  # Get (z,y,x) dimensions
        
        # Resize volume to match model expected input size
        if volume.shape != target_shape:
            print(f"Resizing volume from {volume.shape} to {target_shape}")
            volume = resize(volume, target_shape, preserve_range=True)
        
        # Normalize volume and add channel dimension
        volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-6)
        volume_input = np.expand_dims(volume_normalized, axis=-1)
        
        # For 4-channel model, duplicate single channel input
        if model.input_shape[-1] == 4:
            volume_input = np.repeat(volume_input, 4, axis=-1)
        
        # Add batch dimension
        volume_input = np.expand_dims(volume_input, axis=0)
        
        # Predict segmentation
        print("Running segmentation...")
        segmentation = model.predict(volume_input)[0]
        
        # Convert to discrete labels (argmax for multi-class)
        segmentation_labels = np.argmax(segmentation, axis=-1).astype(np.uint8)
        
        # Map class 3 (enhancing tumor) back to 4 for BraTS compatibility
        segmentation_labels[segmentation_labels == 3] = 4
        
        # Print segmentation statistics
        unique_labels, counts = np.unique(segmentation_labels, return_counts=True)
        print("Segmentation Summary:")
        print(f"Labels present: {unique_labels}")
        print(f"Voxel counts: {counts}")
        
        return segmentation_labels
        
    except Exception as e:
        print(f"Error in segmentation: {e}")
        return None

# Function to create segmentation overlay on MRI slice
def create_segmentation_overlay(mri_slice, seg_slice):
    """Create overlay with high-contrast tumor colors"""
    # Normalize MRI slice
    mri_slice = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-6)
    rgb = np.stack([mri_slice]*3, axis=-1)  # Convert to RGB
    
    if seg_slice is not None:
        # Ultra-high contrast colors for different tumor regions
        tumor_colors = {
            1: [1.0, 0.75, 0.0],  # Amber - necrotic core
            2: [0.5, 0.0, 0.5],   # Purple - edema
            3: [0.0, 1.0, 1.0]    # Cyan - enhancing tumor
        }
        
        # Create overlay image
        overlay = np.zeros_like(rgb)
        for class_idx, color in tumor_colors.items():
            mask = (seg_slice == class_idx).astype(np.uint8)
            overlay[mask == 1] = color
        
        # 50-50 blend for clear visualization
        rgb = 0.5*rgb + 0.5*overlay
    
    return (rgb * 255).astype(np.uint8)  # Convert to 8-bit RGB
    
# Global variable to control auto-rotation
auto_rotate = True

# Timer callback class for automatic rotation
class TimerCallback:
    def __init__(self, renderer_original, renderer_clip):
        self.renderer_original = renderer_original  # Original view renderer
        self.renderer_clip = renderer_clip  # Clipped view renderer
        self.timer_count = 0  # Counter for timer events

    def execute(self, obj, event):
        if auto_rotate:
            # Rotate both cameras
            camera_original = self.renderer_original.GetActiveCamera()
            camera_original.Azimuth(2)  # Rotate by 2 degrees
            camera_clip = self.renderer_clip.GetActiveCamera()
            camera_clip.Azimuth(2)
            obj.GetRenderWindow().Render()  # Trigger re-render
        self.timer_count += 1  # Increment counter

# Custom Channel Attention Layer for the neural network
@register_keras_serializable()
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        # Initialize layer with reduction ratio
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio  # Channel reduction ratio
        self.supports_masking = True  # Support masking in Keras

    def build(self, input_shape):
        # Build layer components based on input shape
        channel = input_shape[-1]  # Get number of channels from input
        # Create shared dense layers for channel attention
        self.shared_dense = tf.keras.Sequential([
            Dense(channel//self.ratio, activation='relu', use_bias=True),  # First dense layer with reduction
            Dense(channel, use_bias=True)  # Second dense layer to expand back
        ])
        super(ChannelAttention, self).build(input_shape)  # Call parent build method

    def call(self, inputs):
        # Forward pass implementation
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)  # Global average pooling
        max_pool = tf.reduce_max(inputs, axis=[1, 2, 3], keepdims=True)  # Global max pooling
        
        # Process pooled features through shared dense layers
        avg_out = self.shared_dense(avg_pool)
        max_out = self.shared_dense(max_pool)
        
        # Combine attention maps with sigmoid activation
        scale = tf.sigmoid(avg_out + max_out)
        return inputs * scale  # Apply attention weights to input

    def get_config(self):
        # Get layer configuration for serialization
        config = super().get_config()
        config.update({'ratio': self.ratio})  # Add ratio to config
        return config

# Custom loss function combining Dice and Focal loss
@register_keras_serializable()
def dice_focal_loss(y_true, y_pred, alpha=[0.1, 0.5, 0.7, 0.9], gamma=2.0):
    # Dice loss component
    smooth = 1e-7  # Small constant for numerical stability
    y_true_f = tf.cast(tf.reshape(y_true, [-1, 4]), tf.float32)  # Flatten and cast true labels
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, 4]), tf.float32)  # Flatten and cast predictions

    # Calculate intersection and Dice coefficient
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth)
    dice_loss = 1.0 - dice  # Convert to loss

    # Focal loss component
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)  # Clip predictions for stability
    focal = -y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)  # Focal loss calculation
    focal = tf.reduce_sum(focal * tf.constant(alpha), axis=-1)  # Apply class weights

    # Combined loss (Dice + Focal)
    return tf.reduce_mean(dice_loss * tf.constant(alpha)) + tf.reduce_mean(focal)

# Function to load segmentation model with custom components
def load_segmentation_model(model_path):
    # Define custom objects needed for loading
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'dice_focal_loss': dice_focal_loss
    }
    
    try:
        # First attempt with standard loading
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Verify layer configurations match training setup
        for layer in model.layers:
            if isinstance(layer, ChannelAttention):
                for dense_layer in layer.shared_dense.layers:
                    if hasattr(dense_layer, 'use_bias') and not dense_layer.use_bias:
                        print(f"Warning: {dense_layer.name} has use_bias=False but expected True")
                        dense_layer.use_bias = True  # Force match training configuration
        
        # Recompile with safe settings
        model.compile(optimizer='adam', loss=dice_focal_loss)
        return model
        
    except Exception as e:
        print(f"Primary loading failed: {str(e)}")
        print("Attempting weight transfer approach...")
        
        try:
            # Create fresh model with correct architecture
            new_model = build_enhanced_unet()  # Use your training function
            
            # Load weights only
            new_model.load_weights(model_path)
            
            # Verify weights loaded properly
            for layer in new_model.layers:
                if not layer.weights:
                    print(f"Warning: No weights loaded for {layer.name}")
            
            new_model.compile(optimizer='adam', loss=dice_focal_loss)
            return new_model
            
        except Exception as e:
            print(f"Weight transfer failed: {str(e)}")
            print("Attempting direct weight assignment...")
            
            try:
                # Last resort - manual weight assignment
                new_model = build_enhanced_unet()
                old_model = tf.keras.models.load_model(model_path, compile=False)
                
                # Copy weights layer by layer
                for new_layer, old_layer in zip(new_model.layers, old_model.layers):
                    if new_layer.weights and old_layer.weights:
                        if len(new_layer.weights) == len(old_layer.weights):
                            new_layer.set_weights(old_layer.get_weights())
                        else:
                            print(f"Skipping {new_layer.name} - weight count mismatch")
                
                new_model.compile(optimizer='adam', loss=dice_focal_loss)
                return new_model
                
            except Exception as e:
                print(f"Complete loading failure: {str(e)}")
                return None

# Main 3D visualization function with cutting tool
def render_3d_volume_with_cutting_tool(volume, spacing, segmentation=None):
    """Render the 3D volume with cutting tool and optional segmentation"""
    try:
        # Convert NumPy array to VTK image data
        vtk_data = numpy_support.numpy_to_vtk(volume.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])  # Set dimensions (x,y,z)
        vtk_image.SetSpacing(spacing[2], spacing[1], spacing[0])  # Set voxel spacing
        vtk_image.GetPointData().SetScalars(vtk_data)  # Set volume data

        # Create volume mapper
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(vtk_image)

        # Create volume property (rendering properties)
        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()  # Enable shading
        volume_property.SetInterpolationTypeToLinear()  # Linear interpolation
        volume_property.SetAmbient(0.3)  # Ambient lighting
        volume_property.SetDiffuse(0.7)  # Diffuse lighting
        volume_property.SetSpecular(0.5)  # Specular highlights
        volume_property.SetSpecularPower(30)  # Specular power
        volume_property.SetShade(1)  # Enable shading
        volume_property.SetScalarOpacityUnitDistance(0.1)  # Opacity adjustment

        # Color transfer function (maps intensity to color)
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)  # Black at 0
        color_transfer_function.AddRGBPoint(0.1, 0.6, 0.8, 1.0)   # Blue at 0.1
        color_transfer_function.AddRGBPoint(0.3, 0.4, 0.4, 0.4)   # Gray at 0.3
        color_transfer_function.AddRGBPoint(0.7, 1.0, 1.0, 1.0)   # White at 0.7
        color_transfer_function.AddRGBPoint(0.6, 1.0, 0.0, 0.0)   # Red at 0.6
        color_transfer_function.AddRGBPoint(1.0, 1.0, 1.0, 0.0)   # Yellow at 1.0
        
        # Opacity transfer function (maps intensity to opacity)
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0.0, 0.0)  # Transparent at 0
        opacity_transfer_function.AddPoint(0.2, 0.2)  # 20% opaque at 0.2
        opacity_transfer_function.AddPoint(0.4, 0.6)  # 60% opaque at 0.4
        opacity_transfer_function.AddPoint(0.5, 0.8)  # 80% opaque at 0.5
        opacity_transfer_function.AddPoint(0.6, 1.0)  # 100% opaque at 0.6
        opacity_transfer_function.AddPoint(1.0, 1.0)  # 100% opaque at 1.0

        # Apply transfer functions to volume properties
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)

        # Create volume actor
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(volume_mapper)  # Set mapper
        volume_actor.SetProperty(volume_property)  # Set properties

        # Create renderer for original brain (left view)
        renderer_original = vtk.vtkRenderer()
        renderer_original.AddVolume(volume_actor)  # Add volume actor
        renderer_original.SetBackground(0, 0, 0)  # Black background
        renderer_original.SetViewport(0, 0, 0.5, 1)  # Left half of window

        # Create plane for clipping
        plane = vtk.vtkPlane()
        plane.SetOrigin(vtk_image.GetCenter())  # Center plane at volume center
        plane.SetNormal(0, 0, 1)  # Initial normal along Z-axis

        # Create cutter for clipped portion (top right view)
        cutter = vtk.vtkCutter()
        cutter.SetInputData(vtk_image)  # Set input volume
        cutter.SetCutFunction(plane)  # Set cutting plane
        cutter.Update()  # Update cutter

        cutter_mapper = vtk.vtkPolyDataMapper()
        cutter_mapper.SetInputConnection(cutter.GetOutputPort())  # Connect to cutter

        cutter_actor = vtk.vtkActor()
        cutter_actor.SetMapper(cutter_mapper)  # Set mapper

        renderer_cutter = vtk.vtkRenderer()
        renderer_cutter.AddActor(cutter_actor)  # Add cutter actor
        renderer_cutter.SetBackground(0, 0, 0)  # Black background
        renderer_cutter.SetViewport(0.5, 0.5, 1, 1)  # Top right quadrant

        # Create clip filter for bottom right view
        clip = vtk.vtkExtractVOI()
        clip.SetInputData(vtk_image)  # Set input volume
        clip.SetVOI(0, volume.shape[2]-1, 0, volume.shape[1]-1, 0, volume.shape[0]-1)  # Set volume of interest
        clip.Update()  # Update clip filter

        clip_mapper = vtk.vtkSmartVolumeMapper()
        clip_mapper.SetInputConnection(clip.GetOutputPort())  # Connect to clip filter

        clip_actor = vtk.vtkVolume()
        clip_actor.SetMapper(clip_mapper)  # Set mapper
        clip_actor.SetProperty(volume_property)  # Share properties with main volume

        renderer_clip = vtk.vtkRenderer()
        renderer_clip.AddVolume(clip_actor)  # Add clipped volume
        renderer_clip.SetBackground(0, 0, 0)  # Black background
        renderer_clip.SetViewport(0.5, 0, 1, 0.5)  # Bottom right quadrant

        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer_original)  # Add original view
        render_window.AddRenderer(renderer_cutter)  # Add cutter view
        render_window.AddRenderer(renderer_clip)  # Add clipped view
        render_window.SetSize(1200, 600)  # Set window size

        # Add segmentation visualization if available
        if segmentation is not None:
            # Verify segmentation first
            unique_labels = np.unique(segmentation)
            print(f"Detected labels: {unique_labels}")
            print(f"Voxel counts: {np.bincount(segmentation.ravel())}")
            
            # Convert segmentation to VTK format
            seg_data = numpy_support.numpy_to_vtk(segmentation.ravel(), 
                                                deep=True, 
                                                array_type=vtk.VTK_UNSIGNED_CHAR)
            
            vtk_seg = vtk.vtkImageData()
            vtk_seg.SetDimensions(segmentation.shape[2], 
                                segmentation.shape[1], 
                                segmentation.shape[0])
            vtk_seg.SetSpacing(spacing[2], spacing[1], spacing[0])
            vtk_seg.GetPointData().SetScalars(seg_data)
            
            # High-contrast color mapping for segmentation
            seg_color = vtk.vtkColorTransferFunction()
            seg_color.AddRGBPoint(0, 0.0, 0.0, 0.0)   # Black background
            seg_color.AddRGBPoint(1, 1.0, 0.75, 0.0)  # Amber - necrotic
            seg_color.AddRGBPoint(2, 0.5, 0.0, 0.5)   # Purple - edema  
            seg_color.AddRGBPoint(3, 0.0, 1.0, 1.0)   # Cyan - enhancing
            
            # High opacity for all tumor classes
            seg_opacity = vtk.vtkPiecewiseFunction()
            seg_opacity.AddPoint(0, 0.0)  # Transparent background
            seg_opacity.AddPoint(1, 1.0)  # 100% opaque
            seg_opacity.AddPoint(2, 1.0)
            seg_opacity.AddPoint(3, 1.0)
            
            # Set segmentation properties
            seg_property = vtk.vtkVolumeProperty()
            seg_property.SetColor(seg_color)
            seg_property.SetScalarOpacity(seg_opacity)
            seg_property.ShadeOn()
            
            # Create segmentation mapper
            seg_mapper = vtk.vtkSmartVolumeMapper()
            seg_mapper.SetInputData(vtk_seg)
            
            # Create segmentation actor
            seg_actor = vtk.vtkVolume()
            seg_actor.SetMapper(seg_mapper)
            seg_actor.SetProperty(seg_property)
            
            # Configure segmentation view
            renderer_seg = vtk.vtkRenderer()
            renderer_seg.AddVolume(seg_actor)
            renderer_seg.SetBackground(0, 0, 0)  # Black background
            renderer_seg.SetViewport(0.5, 0.5, 1, 1)  # Top right quadrant
            
            # Force direct frontal view for segmentation
            camera = renderer_seg.GetActiveCamera()
            camera.SetPosition(0, -1, 0)  # Looking along Y-axis
            camera.SetViewUp(0, 0, 1)     # Z-axis up
            camera.SetFocalPoint(0, 0, 0)
            renderer_seg.ResetCamera()
            
            render_window.AddRenderer(renderer_seg)

        # Create interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # Callback function for updating cutting plane
        def update_cutting_plane(obj, event):
            plane_origin = obj.GetOrigin()  # Get current plane origin
            plane_normal = obj.GetNormal()  # Get current plane normal

            plane.SetOrigin(plane_origin)  # Update cutting plane origin
            plane.SetNormal(plane_normal)  # Update cutting plane normal
            cutter.Update()  # Update cutter with new plane

            # Calculate valid Z range
            z_min = 0
            z_max = volume.shape[0] - 1
            z_cut = int(plane_origin[2])
            z_cut = max(z_min, min(z_max, z_cut))  # Clamp to valid range

            # Update clip filter with new Z range
            clip.SetVOI(0, volume.shape[2]-1, 0, volume.shape[1]-1, z_min, z_cut)
            clip.Update()

            render_window.Render()  # Trigger re-render

        # Create plane widget for interactive cutting
        plane_widget = vtk.vtkImplicitPlaneWidget()
        plane_widget.SetInteractor(interactor)  # Set interactor
        plane_widget.SetPlaceFactor(1.0)  # Set placement factor
        plane_widget.PlaceWidget(vtk_image.GetBounds())  # Set bounds
        plane_widget.SetOrigin(vtk_image.GetCenter())  # Start at center
        plane_widget.SetNormal(0, 0, 1)  # Initial normal along Z-axis
        plane_widget.AddObserver("InteractionEvent", update_cutting_plane)  # Add callback
        plane_widget.On()  # Enable widget

        # Function to setup frontal camera view
        def setup_frontal_camera(renderer):
            camera = renderer.GetActiveCamera()
            camera.SetPosition(0, -1, 0)  # Face the slice towards viewer
            camera.SetViewUp(0, 0, 1)     # Z-axis up
            camera.SetFocalPoint(0, 0, 0)
            renderer.ResetCamera()

        # Setup frontal views for all renderers
        setup_frontal_camera(renderer_original)
        setup_frontal_camera(renderer_clip)
        if segmentation is not None:
            setup_frontal_camera(renderer_seg)
        else:
            setup_frontal_camera(renderer_cutter)

        # Function to rotate and reset camera
        def rotate_and_reset_camera(renderer):
            camera = renderer.GetActiveCamera()
            camera.Roll(80)  # Roll camera
            camera.Azimuth(80)  # Rotate azimuth
            camera.Elevation(360)  # Rotate elevation
            renderer.ResetCamera()  # Reset camera

        # Apply initial rotation to all views
        rotate_and_reset_camera(renderer_original)
        rotate_and_reset_camera(renderer_clip)
        if segmentation is not None:
            rotate_and_reset_camera(renderer_seg)
        else:
            rotate_and_reset_camera(renderer_cutter)

        # Initialize interaction
        interactor.Initialize()

        # Timer callback for auto-rotation
        timer_callback = TimerCallback(renderer_original, renderer_clip)
        interactor.AddObserver('TimerEvent', timer_callback.execute)
        timer_id = interactor.CreateRepeatingTimer(50)  # 50ms timer interval

        # Key press callback function
        def key_pressed_callback(obj, event):
            global auto_rotate
            key = obj.GetKeySym()  # Get pressed key
            if key == 'r' or key == 'R':
                auto_rotate = not auto_rotate  # Toggle auto-rotation
                print("Auto-rotation:", "On" if auto_rotate else "Off")

        interactor.AddObserver('KeyPressEvent', key_pressed_callback)

        # Configure render quality
        render_window.SetMultiSamples(8)  # Enable multi-sampling
        volume_mapper.SetSampleDistance(0.5)  # Set sampling distance
        
        # Hover functionality setup
        picker = vtk.vtkVolumePicker()
        picker.SetTolerance(0.005)  # Set pick tolerance
        
        # Create hover text actor
        hover_text = vtk.vtkTextActor()
        hover_text.GetTextProperty().SetFontSize(24)  # Set font size
        hover_text.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
        hover_text.VisibilityOff()  # Initially hidden
        renderer_clip.AddActor(hover_text)  # Add to renderer
        
        # Brain region intensity ranges and names
        brain_regions = {
            (0.0, 0.1): "CSF",
            (0.1, 0.3): "Gray Matter",
            (0.3, 0.5): "White Matter",
            (0.5, 0.8): "Abnormalities",
            (0.8, 1.0): "High-Intensity Areas"
        }
        
        # Hover callback function
        def hover_callback(obj, event):
            mouse_x, mouse_y = interactor.GetEventPosition()  # Get mouse position
            picker.Pick(mouse_x, mouse_y, 0, renderer_clip)  # Perform pick
            picked_pos = picker.GetPickPosition()  # Get picked position
        
            if picker.GetPointId() >= 0:  # If picked something
                voxel_x, voxel_y, voxel_z = map(int, picked_pos)  # Get voxel coordinates
        
                # Check if within volume bounds
                if 0 <= voxel_x < volume.shape[2] and 0 <= voxel_y < volume.shape[1] and 0 <= voxel_z < volume.shape[0]:
                    picked_intensity = volume[voxel_z, voxel_y, voxel_x]  # Get intensity
        
                    label = None
                    text_color = (1, 1, 1)  # Default white
                    # Find which region the intensity belongs to
                    for (low, high), name in brain_regions.items():
                        if low <= picked_intensity < high:
                            label = name
                            # Set color based on region
                            if name == "CSF":
                                text_color = (0.6, 0.8, 1.0)  # Light blue
                            elif name == "Gray Matter":
                                text_color = (0.4, 0.4, 0.4)  # Gray
                            elif name == "White Matter":
                                text_color = (1.0, 1.0, 1.0)  # White
                            elif name == "Abnormalities":
                                text_color = (1.0, 0.0, 0.0)  # Red
                            elif name == "High-Intensity Areas":
                                text_color = (1.0, 1.0, 0.0)  # Yellow
                            break
        
                    if label:
                        # Update hover text
                        hover_text.SetInput(f"{label}")
                        hover_text.SetPosition(20, 580)  # Position in window
                        hover_text.VisibilityOn()
                        text_property = hover_text.GetTextProperty()
                        text_property.SetColor(*text_color)  # Set color
                        text_property.SetBackgroundColor(0, 0, 0)  # Black background
                        text_property.SetBackgroundOpacity(0.7)  # 70% opaque
                        text_property.SetBold(True)  # Bold text
                        text_property.SetFontSize(24)  # Font size
                        text_property.SetShadow(True)  # Text shadow
                        hover_text.Modified()
                        render_window.Render()
                        return
        
            # If nothing picked or invalid, hide text
            hover_text.VisibilityOff()
            render_window.Render()

        interactor.AddObserver("MouseMoveEvent", hover_callback)  # Add hover callback

        render_window.Render()  # Initial render
        interactor.Start()  # Start interaction loop

    except Exception as e:
        print("An error occurred:", str(e))  # Print any errors

# Main execution block
if __name__ == "__main__":
    # Configuration
    model_path = "/Users/koduruchandrakanthreddy/Downloads/final_model3.keras"  # Model path
    nifti_path = "/Users/koduruchandrakanthreddy/Downloads/BraTS20_Training_001_t1ce.nii"  # MRI data path
    # nifti_path = "/Users/koduruchandrakanthreddy/Downloads/archive/5.nii"
    
    MODEL = True
    
    # 1. Load model only if we're using it
    model = load_segmentation_model(model_path) if MODEL else None
    
    # 2. Load and process NIfTI
    brain_volume, spacing, affine = load_and_process_nifti(nifti_path, is_masked=False)
    
    # Normalize volume
    min_val = np.min(brain_volume)
    max_val = np.max(brain_volume)
    brain_volume_normalized = (brain_volume - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(brain_volume)
    
    # 3. Predict segmentation only if using model
    if MODEL:
        print("Using model for segmentation...")
        segmentation = apply_segmentation(model, brain_volume_normalized)
        
        # Additional verification step
        if segmentation is not None:
            tumor_voxels = np.sum(segmentation > 0)  # Count tumor voxels
            total_voxels = segmentation.size  # Total voxels
            tumor_ratio = tumor_voxels / total_voxels  # Calculate ratio
            print(f"Tumor volume ratio: {tumor_ratio:.4f}")
            
            if tumor_ratio < 0.001:  # If very small tumor
                print("Very small tumor detected - likely false positive")
                segmentation = None  # Discard segmentation
    else:
        print("Skipping model segmentation")
        segmentation = None
    
    # 4. Visualize
    render_3d_volume_with_cutting_tool(brain_volume_normalized, spacing, segmentation)