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
    // val editingShape: StateFlow<OverlayShape?> = _editingShape // Example

    fun openAddImageDialog() {
        _showAddImageDialog.value = true
        // In a real implementation, you might set editingImage to null here
    }

    fun openAddShapeDialog() {
        _showAddShapeDialog.value = true
        // In a real implementation, you might set editingShape to null here
    }

    fun selectImage(image: ImageResource) {
        _selectedImage.value = image
    }

    fun openEditImageDialog(image: ImageResource) {
        _editingImage.value = image
        _showAddImageDialog.value = true // Or a separate edit dialog state
    }

    fun deleteImage(image: ImageResource) {
        _customImages.value = _customImages.value.filterNot { it.id == image.id }
        _availableImages.value = _availableImages.value.filterNot { it.id == image.id }
        if (_selectedImage.value?.id == image.id) {
            _selectedImage.value = null
        }
    }

    fun selectShape(shape: OverlayShape) {
        _selectedShape.value = shape
    }

    fun openEditShapeDialog(shape: OverlayShape) {
        _editingShape.value = shape
        _showAddShapeDialog.value = true // Or a separate edit dialog state
    }

    fun deleteShape(shape: OverlayShape) {
        _shapes.value = _shapes.value.filterNot { it.id == shape.id }
        if (_selectedShape.value?.id == shape.id) {
            _selectedShape.value = null
        }
    }

    // Call this from dialogs when they are dismissed
    fun dismissAllDialogs() {
        _showAddImageDialog.value = false
        _showAddShapeDialog.value = false
        _editingImage.value = null
        _editingShape.value = null
    }
}
