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
import dev.aurakai.auraframefx.R
import dev.aurakai.auraframefx.ui.animation.*
import dev.aurakai.auraframefx.ui.components.*
import dev.aurakai.auraframefx.ui.navigation.NavDestination
import dev.aurakai.auraframefx.ui.theme.*

/**
 * Home screen for the AuraFrameFX app with cyberpunk-style floating UI
 *
 * Features a digital landscape background with floating transparent windows
 * and hexagonal UI elements inspired by futuristic cyberpunk interfaces.
 */
/**
 * Displays the AuraFrameFX home screen with a cyberpunk-themed floating UI, interactive navigation menu, action buttons, and system status panel.
 *
 * The screen features layered digital landscape and hexagonal grid backgrounds, a floating header, a vertical menu for navigation, action buttons for quick access to key screens, and a status panel showing system information. User interactions update the selected menu item and trigger navigation to other screens via the provided navigation controller.
 */
/**
 * Displays the AuraFrameFX home screen with a cyberpunk-themed floating interface.
 *
 * The screen features layered digital landscape and hexagonal grid backgrounds, a floating header, an interactive vertical navigation menu, action buttons for quick access to key areas, and a system status panel. User interactions update the selected menu item and trigger navigation actions via the provided navigation controller.
 */
/**
 * Displays the main home screen with a cyberpunk-themed floating interface, layered digital backgrounds, and interactive navigation.
 *
 * The screen features a digital landscape and hexagonal grid background, a floating header, a vertical navigation menu, action buttons for quick access to other screens, and a system status panel. User interactions update the selected menu item and trigger navigation actions via the provided navigation controller.
 */
/**
 * Displays the main home screen of the AuraFrameFX app with a cyberpunk-themed floating UI.
 *
 * Presents layered digital landscape and hexagonal grid backgrounds, a floating header, a vertical navigation menu, action buttons, and a system status panel. User interactions update the selected menu item and trigger navigation to other screens via the provided navigation controller.
 */
/**
 * Displays the main home screen of the AuraFrameFX app with a layered cyberpunk-themed interface.
 *
 * Renders a digital landscape and hexagonal grid background, a floating header, a vertical navigation menu, action buttons, and a system status panel. User interactions update the selected menu item and trigger navigation to other screens via the provided navigation controller.
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
                cornerStyle = CornerStyle.Hex,
                title = stringResource(R.string.app_title),
                backgroundStyle = BackgroundStyle.HexGrid
            ) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CyberpunkText(
                        text = stringResource(R.string.creativity_engine),
                        color = CyberpunkTextColor.Secondary,
                        style = CyberpunkTextStyle.Label
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    CyberpunkText(
                        text = stringResource(R.string.neural_interface_active),
                        color = CyberpunkTextColor.Warning,
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
                cornerStyle = CornerStyle.Angled
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
                    cornerStyle = CornerStyle.Rounded,
                    backgroundStyle = BackgroundStyle.HexGrid
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CyberpunkText(
                            text = stringResource(R.string.profile),
                            color = CyberpunkTextColor.Secondary,
                            style = CyberpunkTextStyle.Label
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
                    cornerStyle = CornerStyle.Rounded,
                    backgroundStyle = BackgroundStyle.HexGrid
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CyberpunkText(
                            text = stringResource(R.string.config),
                            color = CyberpunkTextColor.Primary,
                            style = CyberpunkTextStyle.Label
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
                    cornerStyle = CornerStyle.Rounded,
                    backgroundStyle = BackgroundStyle.HexGrid
                ) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CyberpunkText(
                            text = stringResource(R.string.oracledrive),
                            color = CyberpunkTextColor.Primary,
                            style = CyberpunkTextStyle.Label
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
                cornerStyle = CornerStyle.Hex,
                title = stringResource(R.string.system_status),
                backgroundStyle = BackgroundStyle.Transparent
            ) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CyberpunkText(
                        text = stringResource(R.string.aura_shield_active),
                        color = CyberpunkTextColor.Primary
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CyberpunkText(
                                text = stringResource(R.string.neural),
                                color = CyberpunkTextColor.White
                            )
                            CyberpunkText(
                                text = stringResource(R.string.active),
                                color = CyberpunkTextColor.Primary
                            )
                        }

                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CyberpunkText(
                                text = stringResource(R.string.quantum),
                                color = CyberpunkTextColor.White
                            )
                            CyberpunkText(
                                text = stringResource(R.string.quantum_percent),
                                color = CyberpunkTextColor.Primary
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}
