package dev.aurakai.auraframefx.ui.debug.model

import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import dev.aurakai.auraframefx.model.agent_states.ProcessingState
import dev.aurakai.auraframefx.model.agent_states.VisionState
import java.util.*

@Immutable
data class GraphNode(
    val id: String,
    val name: String,
    val type: NodeType,
    val position: Offset = Offset(0f, 0f),
    var state: Any? = null,
    val lastUpdated: Long = System.currentTimeMillis(),
    val connections: List<Connection> = emptyList()
) {
    fun withUpdatedState(newState: Any?): GraphNode {
        return copy(state = newState, lastUpdated = System.currentTimeMillis())
    }

    fun withPosition(x: Float, y: Float): GraphNode {
        return copy(position = Offset(x, y))
    }
}

@Immutable
data class Offset(val x: Float, val y: Float) {
    operator fun plus(other: Offset): Offset = Offset(x + other.x, y + other.y)
    operator fun minus(other: Offset): Offset = Offset(x - other.x, y - other.y)
    operator fun times(factor: Float): Offset = Offset(x * factor, y * factor)
    fun distanceTo(other: Offset): Float {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

@Immutable
data class Connection(
    val targetId: String,
    val type: ConnectionType = ConnectionType.DIRECT,
    val label: String = ""
)

enum class NodeType(
    val color: Color,
    val icon: ImageVector,
    val defaultSize: Dp = 48.dp
) {
    VISION(
        color = Color(0xFF03DAC6),
        icon = Icons.Default.Visibility,
        defaultSize = 56.dp
    ),
    PROCESSING(
        color = Color(0xFFBB86FC),
        icon = Icons.Default.Settings,
        defaultSize = 56.dp
    ),
    AGENT(
        color = Color(0xFFCF6679),
        icon = Icons.Default.Person,
        defaultSize = 64.dp
    ),
    DATA(
        color = Color(0xFF018786),
        icon = Icons.Default.Storage,
        defaultSize = 56.dp
    )
}

enum class ConnectionType {
    DIRECT,
    BIDIRECTIONAL,
    DASHED
}

// Extension properties for Dp
val Int.dp: Dp
    get() = Dp(this.toFloat())

val Float.dp: Dp
    get() = Dp(this)
