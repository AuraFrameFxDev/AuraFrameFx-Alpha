package dev.aurakai.auraframefx.ui.viewmodel

import androidx.lifecycle.ViewModel
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.system.overlay.model.OverlayShape // Assuming this is the correct import
import dev.aurakai.auraframefx.ui.model.ImageResource // Assuming this is the correct import
import dev.aurakai.auraframefx.ui.model.ShapeType // Assuming this is the correct import for ShapeType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

// Placeholder for a ShapeManager if it's not defined elsewhere yet
// This might need to be a proper class/interface injected into the ViewModel
class ShapeManager @Inject constructor() {
    /**
     * Creates a custom overlay shape with the specified type and properties.
     *
     * @param type The type of shape to create.
     * @param properties A map of properties defining the shape's characteristics.
     * @return A new OverlayShape instance with a unique ID and the given properties.
     */
    fun createCustomShape(type: ShapeType, properties: Map<String, Any>): OverlayShape {
        // Stub implementation
        return OverlayShape(id = "custom_${type.name}_${System.currentTimeMillis()}", shapeType = type.name, properties = properties)
    }
}

@HiltViewModel
class ImageShapeManagerViewModel @Inject constructor(
    // Assuming ShapeManager might be injected or created internally
    val shapeManager: ShapeManager // Made public val based on usage in AddShapeDialog
) : ViewModel() {

    private val _availableImages = MutableStateFlow<List<ImageResource>>(emptyList())
    val availableImages: StateFlow<List<ImageResource>> = _availableImages

    private val _customImages = MutableStateFlow<List<ImageResource>>(emptyList())
    val customImages: StateFlow<List<ImageResource>> = _customImages

    private val _shapes = MutableStateFlow<List<OverlayShape>>(emptyList())
    val shapes: StateFlow<List<OverlayShape>> = _shapes

    private val _selectedImage = MutableStateFlow<ImageResource?>(null)
    val selectedImage: StateFlow<ImageResource?> = _selectedImage

    private val _selectedShape = MutableStateFlow<OverlayShape?>(null)
    val selectedShape: StateFlow<OverlayShape?> = _selectedShape

    // Dialog states
    private val _showAddImageDialog = MutableStateFlow(false)
    // val showAddImageDialog: StateFlow<Boolean> = _showAddImageDialog // Example if needed by screen

    private val _showAddShapeDialog = MutableStateFlow(false)
    // val showAddShapeDialog: StateFlow<Boolean> = _showAddShapeDialog // Example if needed by screen

    private val _editingImage = MutableStateFlow<ImageResource?>(null)
    // val editingImage: StateFlow<ImageResource?> = _editingImage // Example

    private val _editingShape = MutableStateFlow<OverlayShape?>(null)
    /**
     * Opens the dialog for adding a new image by setting the corresponding visibility flag.
     */

    fun openAddImageDialog() {
        _showAddImageDialog.value = true
        // In a real implementation, you might set editingImage to null here
    }

    /**
     * Opens the dialog for adding a new shape by setting the corresponding visibility flag.
     */
    fun openAddShapeDialog() {
        _showAddShapeDialog.value = true
        // In a real implementation, you might set editingShape to null here
    }

    /**
     * Sets the specified image as the currently selected image.
     *
     * @param image The image to select.
     */
    fun selectImage(image: ImageResource) {
        _selectedImage.value = image
    }

    /**
     * Opens the dialog for editing the specified image and sets it as the current image being edited.
     *
     * @param image The image to edit.
     */
    fun openEditImageDialog(image: ImageResource) {
        _editingImage.value = image
        _showAddImageDialog.value = true // Or a separate edit dialog state
    }

    /**
     * Removes the specified image from both the custom and available image lists.
     *
     * If the image is currently selected, the selection is cleared.
     *
     * @param image The image to be deleted.
     */
    fun deleteImage(image: ImageResource) {
        _customImages.value = _customImages.value.filterNot { it.id == image.id }
        _availableImages.value = _availableImages.value.filterNot { it.id == image.id }
        if (_selectedImage.value?.id == image.id) {
            _selectedImage.value = null
        }
    }

    /**
     * Sets the specified shape as the currently selected shape.
     *
     * @param shape The shape to select.
     */
    fun selectShape(shape: OverlayShape) {
        _selectedShape.value = shape
    }

    /**
     * Opens the dialog for editing the specified shape and sets it as the shape being edited.
     *
     * @param shape The shape to be edited.
     */
    fun openEditShapeDialog(shape: OverlayShape) {
        _editingShape.value = shape
        _showAddShapeDialog.value = true // Or a separate edit dialog state
    }

    /**
     * Removes the specified shape from the list of managed shapes and clears the selection if it was selected.
     *
     * @param shape The shape to be deleted.
     */
    fun deleteShape(shape: OverlayShape) {
        _shapes.value = _shapes.value.filterNot { it.id == shape.id }
        if (_selectedShape.value?.id == shape.id) {
            _selectedShape.value = null
        }
    }

    /**
     * Closes all add and edit dialogs and clears any editing state for images and shapes.
     */
    fun dismissAllDialogs() {
        _showAddImageDialog.value = false
        _showAddShapeDialog.value = false
        _editingImage.value = null
        _editingShape.value = null
    }
}
