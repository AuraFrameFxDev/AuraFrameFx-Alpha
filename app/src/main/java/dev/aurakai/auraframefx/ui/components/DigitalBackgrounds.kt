package dev.aurakai.auraframefx.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import kotlin.math.cos
import kotlin.math.sin

/**
 * Digital landscape background component
 */
@Composable
fun DigitalLandscapeBackground(
    modifier: Modifier = Modifier,
    color: Color = Color(0xFF00FFFF).copy(alpha = 0.3f)
) {
    Canvas(modifier = modifier) {
        drawDigitalLandscape(color)
    }
}

/**
 * Hexagon grid background component
 */
@Composable
fun HexagonGridBackground(
    modifier: Modifier = Modifier,
    alpha: Float = 0.2f,
    color: Color = Color(0xFF00FFFF).copy(alpha = alpha)
) {
    Canvas(modifier = modifier) {
        drawHexagonGrid(color)
    }
}

private fun DrawScope.drawDigitalLandscape(color: Color) {
    // Simple grid pattern for digital landscape
    val spacing = 50f
    for (i in 0 until (size.width / spacing).toInt()) {
        drawLine(
            color = color,
            start = androidx.compose.ui.geometry.Offset(i * spacing, 0f),
            end = androidx.compose.ui.geometry.Offset(i * spacing, size.height),
            strokeWidth = 1f
        )
    }
    for (i in 0 until (size.height / spacing).toInt()) {
        drawLine(
            color = color,
            start = androidx.compose.ui.geometry.Offset(0f, i * spacing),
            end = androidx.compose.ui.geometry.Offset(size.width, i * spacing),
            strokeWidth = 1f
        )
    }
}

private fun DrawScope.drawHexagonGrid(color: Color) {
    // Simple hexagon grid pattern
    val radius = 30f
    val spacing = radius * 1.5f
    
    for (row in 0 until (size.height / spacing).toInt()) {
        for (col in 0 until (size.width / spacing).toInt()) {
            val x = col * spacing + if (row % 2 == 1) spacing / 2 else 0f
            val y = row * spacing
            
            if (x < size.width && y < size.height) {
                drawHexagon(
                    center = androidx.compose.ui.geometry.Offset(x, y),
                    radius = radius * 0.8f,
                    color = color
                )
            }
        }
    }
}

private fun DrawScope.drawHexagon(
    center: androidx.compose.ui.geometry.Offset,
    radius: Float,
    color: Color
) {
    val path = androidx.compose.ui.graphics.Path()
    for (i in 0..5) {
        val angle = i * 60.0 * Math.PI / 180.0
        val x = center.x + radius * cos(angle).toFloat()
        val y = center.y + radius * sin(angle).toFloat()
        
        if (i == 0) {
            path.moveTo(x, y)
        } else {
            path.lineTo(x, y)
        }
    }
    path.close()
    
    drawPath(
        path = path,
        color = color,
        style = Stroke(width = 1f)
    )
}
