package dev.aurakai.auraframefx.ui.debug.component

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.*
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset as ComposeOffset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import dev.aurakai.auraframefx.ui.debug.model.GraphNode
import dev.aurakai.auraframefx.ui.debug.model.Connection
import dev.aurakai.auraframefx.ui.debug.model.NodeType
import dev.aurakai.auraframefx.ui.debug.model.Offset as GraphOffset
import kotlin.math.*

@Composable
fun InteractiveGraph(
    nodes: List<GraphNode>,
    selectedNodeId: String? = null,
    onNodeSelected: (String) -> Unit = {},
    modifier: Modifier = Modifier,
    contentPadding: PaddingValues = PaddingValues(16.dp)
) {
    var scale by remember { mutableStateOf(1f) }
    var translation by remember { mutableStateOf(ComposeOffset.Zero) }
    val infiniteTransition = rememberInfiniteTransition()
    val pulse by infiniteTransition.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        )
    )

    BoxWithConstraints(
        modifier = modifier
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.1f))
            .clip(MaterialTheme.shapes.medium)
    ) {
        val canvasWidth = constraints.maxWidth.toFloat()
        val canvasHeight = constraints.maxHeight.toFloat()
        
        // Calculate content bounds for centering
        val contentWidth = 1000f * scale
        val contentHeight = 800f * scale
        
        val offsetX = (canvasWidth - contentWidth) / 2 + translation.x
        val offsetY = (canvasHeight - contentHeight) / 2 + translation.y
        
        // Convert GraphOffset to ComposeOffset for rendering
        fun GraphOffset.toCompose() = ComposeOffset(x.toFloat(), y.toFloat())

        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTransformGestures(
                        onGesture = { _, pan, zoom, _ ->
                            scale = (scale * zoom).coerceIn(0.5f, 3f)
                            translation += pan / scale
                        }
                    )
                }
        ) {
            // Draw grid
            drawGrid(scale, translation)
            
            // Draw connections first (behind nodes)
            nodes.forEach { node ->
                node.connections.forEach { connection ->
                    val targetNode = nodes.find { it.id == connection.targetId }
                    targetNode?.let { drawConnection(node, it, connection) }
                }
            }
            
            // Draw nodes
            nodes.forEach { node ->
                val isSelected = node.id == selectedNodeId
                val nodeScale = if (isSelected) pulse else 1f
                
                withTransform({
                    scale(scale, scale, ComposeOffset(offsetX, offsetY) + node.position.toCompose() * scale)
                    scale(nodeScale, nodeScale, center = node.position.toCompose())
                }) {
                    drawNode(node, isSelected)
                }
            }
        }
    }
}

private fun DrawScope.drawGrid(scale: Float, translation: ComposeOffset) {
    val gridSize = 40f / scale
    val gridColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
    
    // Draw vertical lines
    for (x in 0 until size.width.toInt() step gridSize.toInt()) {
        drawLine(
            color = gridColor,
            start = Offset(x.toFloat(), 0f) - translation,
            end = Offset(x.toFloat(), size.height) - translation,
            strokeWidth = 1f / scale
        )
    }
    
    // Draw horizontal lines
    for (y in 0 until size.height.toInt() step gridSize.toInt()) {
        drawLine(
            color = gridColor,
            start = Offset(0f, y.toFloat()) - translation,
            end = Offset(size.width, y.toFloat()) - translation,
            strokeWidth = 1f / scale
        )
    }
}

private fun DrawScope.drawNode(node: GraphNode, isSelected: Boolean) {
    val nodeSize = node.type.defaultSize.toPx()
    val center = node.position
    
    // Draw glow/selection ring
    if (isSelected) {
        val ringWidth = 4.dp.toPx()
        drawCircle(
            color = node.type.color.copy(alpha = 0.5f),
            radius = nodeSize * 0.7f,
            center = center,
            style = Stroke(width = ringWidth * 2)
        )
    }
    
    // Draw node background
    drawCircle(
        color = node.type.color.copy(alpha = 0.2f),
        radius = nodeSize * 0.6f,
        center = center
    )
    
    // Draw node border
    drawCircle(
        color = node.type.color,
        radius = nodeSize * 0.6f,
        center = center,
        style = Stroke(width = 2.dp.toPx())
    )
    
    // Draw node icon background
    val iconSize = nodeSize * 0.5f
    val iconBgRadius = iconSize * 0.8f
    
    // Draw icon background circle
    drawCircle(
        color = node.type.color,
        radius = iconBgRadius,
        center = center
    )
    
    // Draw the icon (simplified - actual icon rendering would require more complex handling)
    // For now, we'll just draw a smaller circle as a placeholder
    drawCircle(
        color = Color.White,
        radius = iconBgRadius * 0.5f,
        center = center
    )
    
    // Draw node label
    drawContext.canvas.nativeCanvas.apply {
        drawText(
            node.name,
            center.x,
            center.y + nodeSize * 0.8f,
            android.graphics.Paint().apply {
                color = android.graphics.Color.WHITE
                textSize = 12.dp.toPx()
                textAlign = android.graphics.Paint.Align.CENTER
            }
        )
    }
}

private fun DrawScope.drawConnection(
    from: GraphNode,
    to: GraphNode,
    connection: Connection
) {
    val fromCenter = from.position
    val toCenter = to.position
    val direction = toCenter - fromCenter
    val distance = sqrt(direction.x * direction.x + direction.y * direction.y)
    val directionNormalized = direction / distance
    
    val fromRadius = from.type.defaultSize.toPx() * 0.6f
    val toRadius = to.type.defaultSize.toPx() * 0.6f
    
    val start = fromCenter + directionNormalized * fromRadius
    val end = toCenter - directionNormalized * toRadius
    
    // Draw connection line
    val strokeWidth = 2.dp.toPx()
    val color = when (connection.type) {
        ConnectionType.DIRECT -> Color.White.copy(alpha = 0.7f)
        ConnectionType.BIDIRECTIONAL -> Color.Green.copy(alpha = 0.7f)
        ConnectionType.DASHED -> Color.Yellow.copy(alpha = 0.7f)
    }
    
    if (connection.type == ConnectionType.DASHED) {
        // Draw dashed line
        val dashLength = 10f
        val gapLength = 5f
        val totalLength = distance - fromRadius - toRadius
        val dashCount = (totalLength / (dashLength + gapLength)).toInt()
        
        for (i in 0 until dashCount) {
            val dashStart = start + directionNormalized * (i * (dashLength + gapLength))
            val dashEnd = dashStart + directionNormalized * dashLength
            drawLine(
                color = color,
                start = dashStart,
                end = dashEnd,
                strokeWidth = strokeWidth
            )
        }
    } else {
        // Draw solid line
        drawLine(
            color = color,
            start = start,
            end = end,
            strokeWidth = strokeWidth
        )
    }
    
    // Draw arrow head
    if (connection.type != ConnectionType.BIDIRECTIONAL || true) {
        val arrowSize = 10.dp.toPx()
        val arrowAngle = Math.PI.toFloat() / 4f
        
        val arrowDir = if (connection.type == ConnectionType.BIDIRECTIONAL) {
            -directionNormalized
        } else {
            directionNormalized
        }
        
        val arrowP1 = end.rotate(
            angle = arrowAngle,
            pivot = end + arrowDir * arrowSize,
            pivotOffset = end
        )
        
        val arrowP2 = end.rotate(
            angle = -arrowAngle,
            pivot = end + arrowDir * arrowSize,
            pivotOffset = end
        )
        
        drawPath(
            path = Path().apply {
                moveTo(end.x, end.y)
                lineTo(arrowP1.x, arrowP1.y)
                lineTo(arrowP2.x, arrowP2.y)
                close()
            },
            color = color
        )
    }
}

private operator fun Offset.plus(other: Offset): Offset {
    return Offset(x + other.x, y + other.y)
}

private operator fun Offset.minus(other: Offset): Offset {
    return Offset(x - other.x, y - other.y)
}

private operator fun Offset.div(scalar: Float): Offset {
    return Offset(x / scalar, y / scalar)
}

private fun ComposeOffset.rotate(angle: Float, pivot: ComposeOffset, pivotOffset: ComposeOffset = ComposeOffset.Zero): ComposeOffset {
    val cos = cos(angle)
    val sin = sin(angle)
    
    val translatedX = x - pivot.x
    val translatedY = y - pivot.y
    
    val rotatedX = translatedX * cos - translatedY * sin
    val rotatedY = translatedX * sin + translatedY * cos
    
    return Offset(rotatedX + pivot.x + pivotOffset.x, rotatedY + pivot.y + pivotOffset.y)
}
