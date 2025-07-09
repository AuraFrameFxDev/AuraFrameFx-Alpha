package dev.aurakai.auraframefx.ui.debug

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import dev.aurakai.auraframefx.ai.agents.CascadeAgent
import dev.aurakai.auraframefx.model.agent_states.ProcessingState
import dev.aurakai.auraframefx.model.agent_states.VisionState
import dev.aurakai.auraframefx.ui.debug.component.InteractiveGraph
import dev.aurakai.auraframefx.ui.debug.model.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import javax.inject.Inject
import javax.inject.Singleton
import dev.aurakai.auraframefx.ui.debug.model.Offset as GraphOffset

/**
 * ViewModel for managing the Cascade debug state and interactions.
 */
@Singleton
class CascadeDebugViewModel @Inject constructor(
    private val cascadeAgent: CascadeAgent,
) {
    val visionState = cascadeAgent.visionState
    val processingState = cascadeAgent.processingState

    /**
     * Updates the Cascade agent's vision state with the provided value.
     *
     * @param newState The new vision state to set for the agent.
     */
    fun updateVisionState(newState: VisionState) {
        cascadeAgent.updateVisionState(newState)
    }

    fun updateProcessingState(newState: ProcessingState) {
        cascadeAgent.updateProcessingState(newState)
    }
}

/**
 * Renders an interactive force-directed graph visualizing the Cascade agent's components and their current states.
 *
 * Users can explore the relationships between agent subsystems (such as vision, processing, and data store), select nodes to view detailed state information, and observe real-time updates as the agent's state changes.
 *
 * @param viewModel Supplies the agent's state data and manages updates for the visualization.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CascadeZOrderPlayground(
    viewModel: CascadeDebugViewModel = hiltViewModel(),
    modifier: Modifier = Modifier
) {
    val coroutineScope = rememberCoroutineScope()
    var selectedNodeId by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Simulate loading
    LaunchedEffect(Unit) {
        coroutineScope.launch {
            delay(800) // Simulate network/data loading
            isLoading = false
        }
    }

    // Sample graph nodes - in a real app, these would be derived from your actual agent state
    val nodes = remember {
        try {
            mutableStateListOf(
                GraphNode(
                    id = "vision",
                    name = "Vision",
                    type = NodeType.VISION,
                    position = GraphOffset(200f, 200f),
                    state = "Active",
                    connections = listOf(
                        Connection(
                            targetId = "processing",
                            type = ConnectionType.DIRECT,
                            label = "sends to"
                        )
                    )
                ),
                GraphNode(
                    id = "processing",
                    name = "Processing",
                    type = NodeType.PROCESSING,
                    position = GraphOffset(500f, 200f),
                    state = "Idle",
                    connections = listOf(
                        Connection(
                            targetId = "agent",
                            type = ConnectionType.DIRECT,
                            label = "updates"
                        )
                    )
                ),
                GraphNode(
                    id = "agent",
                    name = "Agent",
                    type = NodeType.AGENT,
                    position = GraphOffset(800f, 200f),
                    state = "Ready",
                    connections = listOf(
                        Connection(
                            targetId = "datastore",
                            type = ConnectionType.BIDIRECTIONAL,
                            label = "reads/writes"
                        )
                    )
                ),
                GraphNode(
                    id = "datastore",
                    name = "Data Store",
                    type = NodeType.DATA,
                    position = GraphOffset(500f, 400f),
                    state = "Connected",
                    connections = listOf(
                        Connection(
                            targetId = "vision",
                            type = ConnectionType.DIRECT,
                            label = "feeds"
                        )
                    )
                )
            )
        } catch (e: Exception) {
            errorMessage = "Failed to load nodes: ${e.localizedMessage}"
            mutableStateListOf()
        }
    }

    // Collect state updates with error handling
    val visionState by viewModel.visionState.collectAsState()
    val processingState by viewModel.processingState.collectAsState()

    // Update node states when view model states change
    LaunchedEffect(visionState, processingState) {
        try {
            nodes.find { it.type == NodeType.VISION }?.let { node ->
                node.state = visionState
                node.lastUpdated = System.currentTimeMillis()
            }
            nodes.find { it.type == NodeType.PROCESSING }?.let { node ->
                node.state = processingState
                node.lastUpdated = System.currentTimeMillis()
            }
        } catch (e: Exception) {
            errorMessage = "Error updating node states: ${e.localizedMessage}"
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Agent State Visualizer",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant,
                    titleContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    actionIconContentColor = MaterialTheme.colorScheme.onSurfaceVariant
                ),
                actions = {
                    if (errorMessage != null) {
                        IconButton(onClick = { errorMessage = null }) {
                            Icon(
                                imageVector = Icons.Default.ErrorOutline,
                                contentDescription = "Error occurred. Click to dismiss.",
                                tint = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                    IconButton(onClick = {
                        isLoading = true
                        coroutineScope.launch {
                            delay(500) // Simulate refresh
                            isLoading = false
                        }
                    }) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Refresh"
                        )
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .padding(padding)
                .then(modifier)
        ) {
            // Show loading indicator
            if (isLoading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.7f)),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }

            // Show error message if any
            errorMessage?.let { message ->
                Snackbar(
                    modifier = Modifier.padding(16.dp),
                    action = {
                        TextButton(onClick = { errorMessage = null }) {
                            Text("DISMISS")
                        }
                    }
                ) {
                    Text(message)
                }
            }
            Column(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                // Graph Visualization with loading state
                val graphAlpha by animateFloatAsState(
                    targetValue = if (isLoading) 0.5f else 1f,
                    label = "graphAlpha"
                )

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .background(
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.1f),
                            shape = RoundedCornerShape(8.dp)
                        )
                        .padding(8.dp)
                        .alpha(graphAlpha)
                ) {
                    InteractiveGraph(
                        nodes = nodes,
                        selectedNodeId = selectedNodeId,
                        onNodeSelected = { nodeId ->
                            selectedNodeId = nodeId
                        },
                        modifier = Modifier.fillMaxSize()
                    )
                }

                // State Details Panel with animation
                val detailsAlpha by animateFloatAsState(
                    targetValue = if (isLoading) 0.5f else 1f,
                    label = "detailsAlpha"
                )

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .padding(top = 8.dp)
                        .alpha(detailsAlpha),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                ) {
                    selectedNodeId?.let { nodeId ->
                        val node = nodes.find { it.id == nodeId }
                        if (node != null) {
                            NodeDetails(node = node)
                        } else {
                            EmptyState("Node not found")
                        }
                    } ?: EmptyState("Select a node to view details")
                }
            }
        }
    }
}

/**
 * Shows a detailed panel with information about the given graph node, including its name, state, last updated time, and connections.
 *
 * @param node The graph node whose details are displayed.
 */
@Composable
private fun NodeDetails(node: GraphNode) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Node header with icon and name
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 8.dp)
        ) {
            // Using a simple circle with a colored background and icon
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(40.dp)
                    .background(
                        node.type.color.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(20.dp)
                    )
            ) {
                // Using a simple text as icon for now
                Text(
                    text = node.name.take(1).uppercase(),
                    style = MaterialTheme.typography.titleMedium,
                    color = node.type.color
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = node.name,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        Divider(
            modifier = Modifier.padding(vertical = 8.dp),
            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
        )

        // State information
        Column(modifier = Modifier.padding(vertical = 4.dp)) {
            Text(
                text = "State",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = node.state ?: "No data",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }

        // Last updated
        Column(modifier = Modifier.padding(vertical = 4.dp)) {
            Text(
                text = "Last Updated",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = SimpleDateFormat("MMM d, yyyy HH:mm:ss", Locale.getDefault())
                    .format(Date(node.lastUpdated)),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // Connections section
        if (node.connections.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Connections",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 4.dp)
            )

            node.connections.forEach { connection ->
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(vertical = 2.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Circle,
                        contentDescription = null,
                        tint = node.type.color,
                        modifier = Modifier.size(8.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${connection.targetId} (${connection.label})",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

/**
 * Displays a preview of the Cascade agent state visualizer UI within a Material theme.
 *
 * Renders the interactive graph visualization and details panel for design-time inspection in the Compose preview.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Preview(showBackground = true, device = "spec:width=411dp,height=891dp")
@Composable
fun CascadeZOrderPlaygroundPreview() {
    MaterialTheme {
        CascadeZOrderPlayground()
    }
}

@Composable
private fun EmptyState(
    message: String,
    modifier: Modifier = Modifier
) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier.fillMaxSize()
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun NodeDetailsPreview() {
    val sampleNode = GraphNode(
        id = "sample-1",
        name = "Vision Processor",
        type = NodeType.VISION,
        state = "Active",
        connections = listOf(
            Connection("data-1", ConnectionType.DIRECT, "Input"),
            Connection("agent-1", ConnectionType.DIRECT, "Output")
        )
    )

    MaterialTheme {
        Surface {
            NodeDetails(node = sampleNode)
        }
    }
}
