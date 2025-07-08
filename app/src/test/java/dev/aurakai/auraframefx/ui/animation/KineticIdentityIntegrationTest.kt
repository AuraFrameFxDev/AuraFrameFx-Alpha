package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Integration tests for KineticIdentity composable in realistic UI scenarios.
 * Testing framework: JUnit 4 with Compose Testing utilities and Robolectric
 */
@RunWith(RobolectricTestRunner::class)
class KineticIdentityIntegrationTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun kineticIdentity_integratesWithComplexLayout() = runTest {
        var lastPosition: Offset? = null
        var interactionCount by mutableStateOf(0)
        
        composeTestRule.setContent {
            Column(modifier = Modifier.fillMaxSize()) {
                Text(
                    text = "Interactions: $interactionCount",
                    modifier = Modifier.testTag("interaction-counter")
                )
                Row(modifier = Modifier.fillMaxWidth()) {
                    KineticIdentity(
                        modifier = Modifier
                            .weight(1f)
                            .height(200.dp)
                            .testTag("kinetic-left"),
                        onPositionChange = { offset ->
                            lastPosition = offset
                            interactionCount++
                        }
                    )
                    KineticIdentity(
                        modifier = Modifier
                            .weight(1f)
                            .height(200.dp)
                            .testTag("kinetic-right"),
                        onPositionChange = { offset ->
                            lastPosition = offset
                            interactionCount++
                        }
                    )
                }
                Button(
                    onClick = { interactionCount = 0 },
                    modifier = Modifier.testTag("reset-button")
                ) {
                    Text("Reset")
                }
            }
        }
        
        // Test interactions with left kinetic identity
        composeTestRule.onNodeWithTag("kinetic-left")
            .performTouchInput {
                down(Offset(50f, 100f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify state updates
        composeTestRule.onNodeWithTag("interaction-counter")
            .assertTextContains("1")
        
        // Test interactions with right kinetic identity
        composeTestRule.onNodeWithTag("kinetic-right")
            .performTouchInput {
                down(Offset(50f, 100f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify state updates again
        composeTestRule.onNodeWithTag("interaction-counter")
            .assertTextContains("2")
        
        // Test reset functionality
        composeTestRule.onNodeWithTag("reset-button").performClick()
        composeTestRule.waitForIdle()
        
        composeTestRule.onNodeWithTag("interaction-counter")
            .assertTextContains("0")
    }

    @Test
    fun kineticIdentity_handlesStateChangesDuringInteraction() = runTest {
        var isEnabled by mutableStateOf(true)
        var capturedPositions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Column {
                Button(
                    onClick = { isEnabled = !isEnabled },
                    modifier = Modifier.testTag("toggle-button")
                ) {
                    Text(if (isEnabled) "Disable" else "Enable")
                }
                
                if (isEnabled) {
                    KineticIdentity(
                        modifier = Modifier
                            .size(200.dp)
                            .testTag("conditional-kinetic"),
                        onPositionChange = { offset ->
                            capturedPositions.add(offset)
                        }
                    )
                }
            }
        }
        
        // Initial state - component should be present
        composeTestRule.onNodeWithTag("conditional-kinetic").assertExists()
        
        // Interact with component
        composeTestRule.onNodeWithTag("conditional-kinetic")
            .performTouchInput {
                down(Offset(100f, 100f))
                up()
            }
        
        composeTestRule.waitForIdle()
        assertTrue(capturedPositions.isNotEmpty(), "Should capture position when enabled")
        
        // Disable component
        composeTestRule.onNodeWithTag("toggle-button").performClick()
        composeTestRule.waitForIdle()
        
        // Component should no longer exist
        composeTestRule.onNodeWithTag("conditional-kinetic").assertDoesNotExist()
        
        // Re-enable component
        composeTestRule.onNodeWithTag("toggle-button").performClick()
        composeTestRule.waitForIdle()
        
        // Component should exist again
        composeTestRule.onNodeWithTag("conditional-kinetic").assertExists()
    }

    @Test
    fun kineticIdentity_performanceWithManyInstances() = runTest {
        val instanceCount = 10
        val capturedEvents = mutableMapOf<Int, Int>()
        
        composeTestRule.setContent {
            LazyColumn {
                items(instanceCount) { index ->
                    KineticIdentity(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(50.dp)
                            .testTag("kinetic-$index"),
                        onPositionChange = {
                            capturedEvents[index] = (capturedEvents[index] ?: 0) + 1
                        }
                    )
                }
            }
        }
        
        // Interact with several instances
        repeat(5) { index ->
            composeTestRule.onNodeWithTag("kinetic-$index")
                .performTouchInput {
                    down(Offset(25f, 25f))
                    up()
                }
        }
        
        composeTestRule.waitForIdle()
        
        // Verify events were captured for interacted instances
        assertTrue(capturedEvents.isNotEmpty(), "Should capture events from multiple instances")
        assertTrue(capturedEvents.size <= 5, "Should only have events from interacted instances")
    }
}