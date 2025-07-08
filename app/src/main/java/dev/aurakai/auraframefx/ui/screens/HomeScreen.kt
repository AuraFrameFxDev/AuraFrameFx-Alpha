package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import dev.aurakai.auraframefx.R // Already present, good.
import dev.aurakai.auraframefx.ui.animation.*
import dev.aurakai.auraframefx.ui.components.CyberMenuItem
import dev.aurakai.auraframefx.ui.components.CyberpunkText
import dev.aurakai.auraframefx.ui.components.FloatingCyberWindow
import dev.aurakai.auraframefx.ui.components.AuraSparkleButton
import dev.aurakai.auraframefx.ui.components.CornerStyle
import dev.aurakai.auraframefx.ui.components.BackgroundStyle
import dev.aurakai.auraframefx.ui.components.HexagonGridBackground
import dev.aurakai.auraframefx.ui.components.DigitalLandscapeBackground
import dev.aurakai.auraframefx.ui.components.digitalGlitchEffect // Explicit import
import dev.aurakai.auraframefx.ui.components.cyberEdgeGlow // Explicit import for consistency
import dev.aurakai.auraframefx.ui.navigation.NavDestination
import dev.aurakai.auraframefx.ui.theme.*


/**
 * Home screen for the AuraFrameFX app with cyberpunk-style floating UI
 *
 * Features a digital landscape background with floating transparent windows
 * and hexagonal UI elements inspired by futuristic cyberpunk interfaces.
 */
/**
 * Displays the AuraFrameFX home screen with a cyberpunk-themed floating UI, layered backgrounds, interactive navigation menu, action buttons, and system status panels.
 *
 * The screen features a digital landscape and hexagon grid background, a stylized title header, a main navigation menu with selectable items and AI chat access, action buttons for system functions, and a status panel showing neural and quantum system states. Navigation actions are triggered based on user interaction with menu items and buttons.
 */
/**
 * Displays the main home screen of the AuraFrameFX app with a cyberpunk-themed UI.
 *
 * The screen features layered animated backgrounds, a floating window header, a navigation menu with selectable items and an AI chat button, action buttons for system navigation, and a status panel showing neural and quantum system states. Navigation to other screens is triggered by user interaction with menu items and buttons.
 */
/**
 * Displays the AuraFrameFX home screen with a cyberpunk-themed floating interface.
 *
 * Renders layered animated backgrounds, a floating title header, an interactive navigation menu, action buttons, and a system status panel. Users can navigate to different app sections, including AI chat, profile, settings, and Oracle Drive Control, through visually distinct menu items and buttons. The screen features custom visual effects and thematic text styling to create a futuristic user experience.
 */
@Composable
fun HomeScreen(navController: NavController) {
    // Track selected menu item
    var selectedMenuItem by remember { mutableStateOf("") }

    // Background with digital landscape and hexagon grid
    Box(modifier = Modifier.fillMaxSize()) {
        // Digital landscape background like in image reference 4
        DigitalLandscapeBackground(
            modifier = Modifier.fillMaxSize()
        )

        // Animated hexagon grid overlay like in image reference 1
        HexagonGridBackground(
            modifier = Modifier.fillMaxSize(),
            alpha = 0.2f
        )

        // Main content with floating windows
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(40.dp))

            // Title header like in image reference 4
            FloatingCyberWindow(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(120.dp),
                cornerStyle = CornerStyle.HEXAGON,
                title = stringResource(R.string.app_title),
                backgroundStyle = BackgroundStyle.MATRIX
            ) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CyberpunkText(
                        text = stringResource(R.string.app_name),
                        color = CyberpunkTextColor.Primary,
                        style = CyberpunkTextStyle.Label
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    CyberpunkText(
                        text = "Neural Interface Active",
                        color = CyberpunkTextColor.Secondary,
                        style = CyberpunkTextStyle.Body
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Main navigation menu like in image reference 1
            FloatingCyberWindow(
                modifier = Modifier
                    .fillMaxWidth()
                    .cyberEdgeGlow(),
                title = stringResource(R.string.virtual_monitorization),
                cornerStyle = CornerStyle.SHARP
            ) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    // Menu items like in image reference 1
                    listOf(
                        stringResource(R.string.menu_ui_engine),
                        stringResource(R.string.menu_aurashield),
                        stringResource(R.string.menu_aurakaiecosys),
                        stringResource(R.string.menu_conference_room)
                    ).forEach { menuItem ->
                        CyberMenuItem(
                            text = menuItem,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                                .digitalPixelEffect(visible = selectedMenuItem == menuItem)
                                .clickable {
                                    selectedMenuItem = menuItem
                                    if (menuItem == stringResource(R.string.menu_conference_room)) {
                                        navController.navigate(NavDestination.AiChat.route)
                                    }
                                },
                            isSelected = selectedMenuItem == menuItem
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    AuraSparkleButton(
                        text = stringResource(R.string.ai_chat_placeholder),
                        onClick = { 
                            selectedMenuItem = "ai_chat"
                            navController.navigate(NavDestination.AiChat.route)
                        }
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Warning message like in image reference 4
                    if (selectedMenuItem != stringResource(R.string.menu_conference_room)) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 16.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CyberpunkText(
                                text = stringResource(R.string.xhancement_caution),
                                color = CyberpunkTextColor.Warning,
                                style = CyberpunkTextStyle.Glitch,
                                enableGlitch = true
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Action buttons - like in image reference 3
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                // These buttons match the style in reference image 3
                FloatingCyberWindow(
                    modifier = Modifier
                        .size(80.dp)
                        .cyberEdgeGlow(
                            primaryColor = NeonPink,
                            secondaryColor = NeonBlue
                        )
                        .clickable { navController.navigate(NavDestination.Profile.route) },
                    cornerStyle = CornerStyle.ROUNDED,
                    title = "System Status",
                    backgroundStyle = BackgroundStyle.SOLID
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CyberpunkText(
                            text = "System Online",
                            color = CyberpunkTextColor.Warning,
                            style = CyberpunkTextStyle.Body
                        )
                    }
                }

                FloatingCyberWindow(
                    modifier = Modifier
                        .size(80.dp)
                        .cyberEdgeGlow(
                            primaryColor = NeonCyan,
                            secondaryColor = NeonBlue
                        )
                        .clickable { navController.navigate(NavDestination.Settings.route) },
                    cornerStyle = CornerStyle.ROUNDED,
                    title = "Performance Metrics",
                    backgroundStyle = BackgroundStyle.SOLID
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CyberpunkText(
                            text = "Ready for Input",
                            color = CyberpunkTextColor.Primary,
                            style = CyberpunkTextStyle.Body
                        )
                    }
                }

                FloatingCyberWindow(
                    modifier = Modifier
                        .size(80.dp)
                        .cyberEdgeGlow(
                            primaryColor = NeonGreen,
                            secondaryColor = NeonBlue
                        )
                        .clickable { navController.navigate(NavDestination.OracleDriveControl.route) },
                    cornerStyle = CornerStyle.ROUNDED,
                    title = "Network Status",
                    backgroundStyle = BackgroundStyle.SOLID
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CyberpunkText(
                            text = "Aura Framework Active",
                            color = CyberpunkTextColor.Secondary,
                            style = CyberpunkTextStyle.Body
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Status panel based on image reference 5
            FloatingCyberWindow(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(140.dp)
                    .digitalGlitchEffect(),
                cornerStyle = CornerStyle.HEXAGON,
                title = "Digital Matrix",
                backgroundStyle = BackgroundStyle.GRADIENT
            ) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CyberpunkText(
                        text = stringResource(R.string.aura_shield_active),
                        color = CyberpunkTextColor.Primary,
                        style = CyberpunkTextStyle.Body // Added style
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CyberpunkText(
                                text = stringResource(R.string.neural),
                                color = CyberpunkTextColor.White,
                                style = CyberpunkTextStyle.Label // Added style
                            )
                            CyberpunkText(
                                text = stringResource(R.string.active),
                                color = CyberpunkTextColor.Primary,
                                style = CyberpunkTextStyle.Body // Added style
                            )
                        }

                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CyberpunkText(
                                text = stringResource(R.string.quantum),
                                color = CyberpunkTextColor.White,
                                style = CyberpunkTextStyle.Label // Added style
                            )
                            CyberpunkText(
                                text = stringResource(R.string.quantum_percent),
                                color = CyberpunkTextColor.Primary,
                                style = CyberpunkTextStyle.Body // Added style
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}
