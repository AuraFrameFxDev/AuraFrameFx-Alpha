package dev.aurakai.auraframefx.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import dev.aurakai.auraframefx.R // Corrected import
import dev.aurakai.auraframefx.ui.theme.AppDimensions

/**
 * Settings screen for the AuraFrameFX app
 */
/**
 * Displays the settings screen with options for theme, notifications, and privacy.
 *
 * Presents a vertically arranged list of settings, each in a card with a title, description, and toggle switch.
 * Each setting maintains its own toggle state locally within the screen.
 */
@Composable
fun SettingsScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(AppDimensions.spacing_medium)
    ) {
        Text(
            text = stringResource(R.string.settings),
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onSurface
        )

        Spacer(modifier = Modifier.height(AppDimensions.spacing_medium))

        SettingsCard(
            title = stringResource(R.string.theme),
            description = stringResource(R.string.dark_mode)
        ) {
            var darkThemeEnabled by remember { mutableStateOf(true) }
            Switch(
                checked = darkThemeEnabled,
                onCheckedChange = { darkThemeEnabled = it }
            )
        }

        Spacer(modifier = Modifier.height(AppDimensions.spacing_medium))

        SettingsCard(
            title = stringResource(R.string.notifications),
            description = stringResource(R.string.enable_notifications)
        ) {
            var notificationsEnabled by remember { mutableStateOf(true) }
            Switch(
                checked = notificationsEnabled,
                onCheckedChange = { notificationsEnabled = it }
            )
        }

        Spacer(modifier = Modifier.height(AppDimensions.spacing_medium))

        SettingsCard(
            title = stringResource(R.string.privacy),
            description = stringResource(R.string.private_account)
        ) {
            var privacyEnabled by remember { mutableStateOf(false) }
            Switch(
                checked = privacyEnabled,
                onCheckedChange = { privacyEnabled = it }
            )
        }

        Spacer(modifier = Modifier.height(AppDimensions.spacing_medium))

        IncantationSettingsCard()
    }
}

@Composable
private fun IncantationSettingsCard() {
    var wakeWordEnabled by remember { mutableStateOf(false) }
    var customWakeWord by remember { mutableStateOf("Aura Activate") } // Default wake-word

    SettingsCard(
        title = "Incantation & Voice", // TODO: Use stringResource
        description = "Configure the \"Aura Activate\" wake-word and voice settings." // TODO: Use stringResource
    ) {
        Column(horizontalAlignment = Alignment.End) {
            Switch(
                checked = wakeWordEnabled,
                onCheckedChange = { wakeWordEnabled = it }
            )
        }
    }
    // Add TextField and Button below the card, or integrate more deeply if design allows
    // For simplicity, adding below the card for now.
    // A more integrated approach might pass more complex content to SettingsCard's content lambda.
    if (wakeWordEnabled) {
        Spacer(modifier = Modifier.height(AppDimensions.spacing_small))
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(AppDimensions.spacing_medium)) {
                androidx.compose.material3.OutlinedTextField(
                    value = customWakeWord,
                    onValueChange = { customWakeWord = it },
                    label = { Text("Custom Wake-Word") }, // TODO: Use stringResource
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
                Spacer(modifier = Modifier.height(AppDimensions.spacing_small))
                androidx.compose.material3.Button(
                    onClick = {
                        // TODO: Implement save logic for the wake-word
                        // This would typically involve calling a ViewModel method
                        // which then saves to DataStore or other persistence.
                        // For now, it just updates local state if needed, but state is already updated.
                        println("Wake-word saved: $customWakeWord (Enabled: $wakeWordEnabled)")
                    },
                    modifier = Modifier.align(Alignment.End)
                ) {
                    Text("Save Wake-Word") // TODO: Use stringResource
                }
            }
        }
    }
}

/**
 * Displays a card containing a title, description, and a customizable content slot.
 *
 * Typically used to present a settings option with descriptive text and an interactive element such as a toggle switch.
 *
 * @param title The heading text displayed at the top of the card.
 * @param description Additional information shown below the title.
 * @param content A composable slot for custom UI, usually an interactive control.
 */
@Composable
private fun SettingsCard(
    title: String,
    description: String,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(AppDimensions.spacing_medium)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                content()
            }
        }
    }
}
