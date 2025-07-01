package dev.aurakai.auraframefx.ui.screens

// Assuming theme colors like NeonBlue, NeonGreen, NeonRed are defined in ui.theme package
// If not, use MaterialTheme.colorScheme alternatives or simple Color values.
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import dev.aurakai.auraframefx.ui.theme.NeonBlue
import dev.aurakai.auraframefx.ui.theme.NeonGreen
import dev.aurakai.auraframefx.ui.theme.NeonRed
import dev.aurakai.auraframefx.viewmodel.DiagnosticsViewModel

/**
 * Displays a diagnostics screen showing system status and live logs.
 *
 * Presents a top app bar with a refresh action, a card listing key-value pairs of system status (with color-coded values), and a scrollable area for live logs. The logs automatically scroll to the latest entry when updated. Loading and empty states are handled for both status and logs. The UI uses Material3 theming and custom neon colors for a cyberpunk aesthetic.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DiagnosticsScreen(viewModel: DiagnosticsViewModel = hiltViewModel()) {
    val logs by viewModel.currentLogs.collectAsState()
    val systemStatus by viewModel.systemStatus.collectAsState()
    val scrollState = rememberScrollState()

    // Automatically scroll to bottom when logs change, if desired
    LaunchedEffect(logs) {
        scrollState.animateScrollTo(scrollState.maxValue)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "AuraFX Diagnostics",
                        color = Color.White
                    )
                }, // Or MaterialTheme.colorScheme.onPrimary
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary,
                    actionIconContentColor = MaterialTheme.colorScheme.onPrimary
                ),
                actions = {
                    IconButton(onClick = { viewModel.refreshLogs() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh Logs")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(MaterialTheme.colorScheme.background) // Use theme background
                .padding(8.dp) // Add overall padding for the content area
        ) {
            // System Status Section
            Text(
                "System Status:",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.padding(bottom = 8.dp, start = 8.dp, end = 8.dp)
            )
            Card( // Use Card for better visual grouping of status
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    if (systemStatus.isEmpty()) {
                        Text(
                            "Loading system status...",
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    } else {
                        systemStatus.entries.toList().sortedBy { it.key }
                            .forEach { (key, value) -> // Sort for consistent order
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 2.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        "$key:",
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.weight(0.4f),
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        value,
                                        color = when (value.lowercase()) { // Case-insensitive check for status
                                            "online" -> NeonGreen // Assuming NeonGreen is defined
                                            "offline (or check error)", "offline" -> NeonRed // Assuming NeonRed is defined
                                            else -> MaterialTheme.colorScheme.onSurface
                                        },
                                        modifier = Modifier.weight(0.6f),
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                }
                                Divider(modifier = Modifier.padding(vertical = 4.dp)) // Add divider
                            }
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            // Log Display Section
            Text(
                "Live Logs (Today):",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.padding(bottom = 8.dp, start = 8.dp, end = 8.dp)
            )

            Surface( // Use Surface for the log area background
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f) // Takes remaining space
                    .padding(horizontal = 8.dp),
                shape = MaterialTheme.shapes.medium,
                color = Color.Black.copy(alpha = 0.8f), // Dark background for logs
                tonalElevation = 2.dp
            ) {
                Box(modifier = Modifier.padding(8.dp)) { // Inner padding for log text
                    Text(
                        text = logs,
                        color = NeonBlue, // Cyberpunk console look (ensure NeonBlue is defined)
                        fontSize = 10.sp, // Smaller font for logs
                        fontFamily = FontFamily.Monospace, // Monospace for logs
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(scrollState)
                    )
                    if (logs.isEmpty() || logs == "Loading logs..." || logs.startsWith("No logs")) {
                        Text(
                            logs, // Show "Loading..." or "No logs..." centered
                            color = Color.Gray,
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.align(Alignment.Center)
                        )
                    }
                }
            }
        }
    }
}
