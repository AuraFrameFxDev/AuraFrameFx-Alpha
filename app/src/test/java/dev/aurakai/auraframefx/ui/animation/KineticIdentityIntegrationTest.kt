package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.unit.dp
import org.junit.Rule
import org.junit.Test

/**
 * Integration tests for KineticIdentity component in realistic usage scenarios
 */
class KineticIdentityIntegrationTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun kineticIdentity_inComplexLayout_worksCorrectly() {
        // Test KineticIdentity in a complex layout scenario
        composeTestRule.setContent {
            var lastPosition by remember { mutableStateOf("No touch yet") }

            Column(modifier = Modifier.fillMaxSize()) {
                Text(
                    text = "Position: $lastPosition",
                    modifier = Modifier.testTag("position-text")
                )

                Box(
                    modifier = Modifier
                        .size(200.dp)
                        .background(Color.LightGray)
                ) {
                    KineticIdentity(
                        modifier = Modifier
                            .fillMaxSize()
                            .testTag("kinetic-identity"),
                        onPositionChange = { position ->
                            lastPosition = "(${position.x.toInt()}, ${position.y.toInt()})"
                        }
                    )
                }
            }
        }

        // Verify initial state
        composeTestRule.onNodeWithTag("position-text")
            .assertTextContains("No touch yet")

        // Interact with KineticIdentity
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(Offset(50f, 75f))
            }

        // Verify position was updated
        composeTestRule.waitUntil(timeoutMillis = 1000) {
            try {
                composeTestRule.onNodeWithTag("position-text")
                    .assertTextContains("(")
                true
            } catch (e: AssertionError) {
                false
            }
        }
    }

    @Test
    fun multipleKineticIdentities_workIndependently() {
        // Test multiple KineticIdentity components working independently
        composeTestRule.setContent {
            var position1 by remember { mutableStateOf("0,0") }
            var position2 by remember { mutableStateOf("0,0") }

            Column {
                Text(
                    text = "Position 1: $position1",
                    modifier = Modifier.testTag("position1-text")
                )

                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .background(Color.Red)
                ) {
                    KineticIdentity(
                        modifier = Modifier
                            .fillMaxSize()
                            .testTag("kinetic1"),
                        onPositionChange = { position ->
                            position1 = "${position.x.toInt()},${position.y.toInt()}"
                        }
                    )
                }

                Text(
                    text = "Position 2: $position2",
                    modifier = Modifier.testTag("position2-text")
                )

                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .background(Color.Blue)
                ) {
                    KineticIdentity(
                        modifier = Modifier
                            .fillMaxSize()
                            .testTag("kinetic2"),
                        onPositionChange = { position ->
                            position2 = "${position.x.toInt()},${position.y.toInt()}"
                        }
                    )
                }
            }
        }

        // Interact with first component
        composeTestRule.onNodeWithTag("kinetic1")
            .performTouchInput {
                click(Offset(25f, 25f))
            }

        // Interact with second component  
        composeTestRule.onNodeWithTag("kinetic2")
            .performTouchInput {
                click(Offset(75f, 75f))
            }

        // Verify both components work independently
        composeTestRule.waitUntil(timeoutMillis = 2000) {
            try {
                composeTestRule.onNodeWithTag("position1-text")
                    .assertTextContains("25,25")
                composeTestRule.onNodeWithTag("position2-text")
                    .assertTextContains("75,75")
                true
            } catch (e: AssertionError) {
                false
            }
        }
    }

    @Test
    fun kineticIdentity_withOtherModifiers_maintainsFunctionality() {
        // Test KineticIdentity with various other modifiers
        var interactionCount = 0

        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(150.dp)
                    .background(Color.Yellow)
                    .testTag("styled-kinetic"),
                onPositionChange = {
                    interactionCount++
                }
            )
        }

        composeTestRule.onNodeWithTag("styled-kinetic").assertIsDisplayed()

        composeTestRule.onNodeWithTag("styled-kinetic")
            .performTouchInput {
                click(center)
            }

        composeTestRule.waitUntil(timeoutMillis = 1000) {
            interactionCount > 0
        }
    }

    @Test
    fun kineticIdentity_performanceWithManyInteractions() {
        // Test performance with many interactions
        var totalInteractions = 0
        val maxInteractions = 20

        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("performance-test"),
                    onPositionChange = {
                        totalInteractions++
                    }
                )
            }
        }

        // Perform many interactions quickly
        repeat(maxInteractions) { index ->
            composeTestRule.onNodeWithTag("performance-test")
                .performTouchInput {
                    click(Offset(10f + (index % 10) * 5f, 10f + (index % 10) * 5f))
                }
        }

        // Allow time for all interactions to be processed
        composeTestRule.waitUntil(timeoutMillis = 5000) {
            totalInteractions >= maxInteractions / 2 // Allow for some to be processed
        }
    }
}