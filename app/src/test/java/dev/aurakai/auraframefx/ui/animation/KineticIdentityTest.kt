package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
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
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Comprehensive unit tests for KineticIdentity composable.
 * Testing framework: JUnit 4 with Compose Testing utilities and Robolectric
 */
@RunWith(RobolectricTestRunner::class)
class KineticIdentityTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun kineticIdentity_rendersWithoutCrashing() {
        // Test basic rendering without any crashes
        composeTestRule.setContent {
            KineticIdentity()
        }
        
        // Verify the component renders successfully
        composeTestRule.waitForIdle()
    }

    @Test
    fun kineticIdentity_acceptsModifierParameter() {
        // Test that the component properly accepts and applies modifiers
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("kinetic-identity")
            )
        }
        
        // Verify the test tag is applied (indicating modifier is working)
        composeTestRule.onNodeWithTag("kinetic-identity").assertExists()
    }

    @Test
    fun kineticIdentity_callsOnPositionChangeCallback() = runTest {
        var capturedPosition: Offset? = null
        var callbackInvoked = false
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { offset ->
                        capturedPosition = offset
                        callbackInvoked = true
                    }
                )
            }
        }
        
        // Simulate a pointer event
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                down(Offset(50f, 50f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify the callback was invoked
        assertTrue(callbackInvoked, "onPositionChange callback should be invoked")
        assertNotNull(capturedPosition, "Position should be captured")
    }

    @Test
    fun kineticIdentity_handlesMultiplePointerEvents() = runTest {
        val capturedPositions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { offset ->
                        capturedPositions.add(offset)
                    }
                )
            }
        }
        
        // Simulate multiple pointer events
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                down(Offset(25f, 25f))
                up()
            }
        
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                down(Offset(75f, 75f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify multiple positions were captured
        assertTrue(capturedPositions.size >= 1, "At least one position should be captured")
    }

    @Test
    fun kineticIdentity_worksWithEmptyCallback() {
        // Test that component works even when no callback is provided
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic-identity")
                // onPositionChange is using default empty implementation
            )
        }
        
        // Simulate pointer event with empty callback
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                down(Offset(10f, 10f))
                up()
            }
        
        // Should not crash
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("kinetic-identity").assertExists()
    }

    @Test
    fun kineticIdentity_handlesEdgeCasePositions() = runTest {
        val capturedPositions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { offset ->
                        capturedPositions.add(offset)
                    }
                )
            }
        }
        
        // Test edge positions
        val edgePositions = listOf(
            Offset(0f, 0f),      // Top-left corner
            Offset(199f, 0f),    // Top-right corner
            Offset(0f, 199f),    // Bottom-left corner
            Offset(199f, 199f),  // Bottom-right corner
            Offset(100f, 100f)   // Center
        )
        
        edgePositions.forEach { position ->
            composeTestRule.onNodeWithTag("kinetic-identity")
                .performTouchInput {
                    down(position)
                    up()
                }
            composeTestRule.waitForIdle()
        }
        
        // Verify positions were captured
        assertTrue(capturedPositions.isNotEmpty(), "Edge positions should be captured")
    }

    @Test
    fun kineticIdentity_layoutBehaviorWithDifferentSizes() {
        // Test layout behavior with different component sizes
        val testSizes = listOf(50.dp, 100.dp, 200.dp, 300.dp)
        
        testSizes.forEach { size ->
            composeTestRule.setContent {
                KineticIdentity(
                    modifier = Modifier
                        .size(size)
                        .testTag("kinetic-identity-$size")
                )
            }
            
            // Verify component exists and can handle events
            composeTestRule.onNodeWithTag("kinetic-identity-$size")
                .assertExists()
                .performTouchInput {
                    down(Offset(10f, 10f))
                    up()
                }
            
            composeTestRule.waitForIdle()
        }
    }

    @Test
    fun kineticIdentity_handlesRapidPointerEvents() = runTest {
        var eventCount = 0
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { 
                        eventCount++
                    }
                )
            }
        }
        
        // Simulate rapid pointer events
        repeat(5) { index ->
            composeTestRule.onNodeWithTag("kinetic-identity")
                .performTouchInput {
                    down(Offset(20f + index * 10f, 20f + index * 10f))
                    up()
                }
        }
        
        composeTestRule.waitForIdle()
        
        // Verify events were processed
        assertTrue(eventCount > 0, "Rapid events should be processed")
    }

    @Test
    fun kineticIdentity_preservesModifierChain() {
        // Test that the component preserves the modifier chain properly
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(150.dp)
                    .testTag("outer-tag")
                    .testTag("inner-tag") // Should preserve both tags
            )
        }
        
        // Verify both test tags are preserved
        composeTestRule.onNodeWithTag("outer-tag").assertExists()
        composeTestRule.onNodeWithTag("inner-tag").assertExists()
    }

    @Test
    fun kineticIdentity_layoutCalculationEdgeCases() {
        // Test layout calculation with edge cases
        composeTestRule.setContent {
            Box(modifier = Modifier.size(1.dp)) { // Very small size
                KineticIdentity(
                    modifier = Modifier.testTag("tiny-kinetic")
                )
            }
        }
        
        // Should handle very small sizes without crashing
        composeTestRule.onNodeWithTag("tiny-kinetic").assertExists()
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(1000.dp)) { // Very large size
                KineticIdentity(
                    modifier = Modifier.testTag("large-kinetic")
                )
            }
        }
        
        // Should handle very large sizes without crashing
        composeTestRule.onNodeWithTag("large-kinetic").assertExists()
    }

    @Test
    fun kineticIdentity_callbackReceivesValidOffsets() = runTest {
        val capturedPositions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { offset ->
                        capturedPositions.add(offset)
                    }
                )
            }
        }
        
        val testPosition = Offset(75f, 125f)
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                down(testPosition)
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify the captured position is valid
        assertTrue(capturedPositions.isNotEmpty(), "Should capture at least one position")
        val capturedPosition = capturedPositions.first()
        assertTrue(capturedPosition.x >= 0f, "X coordinate should be non-negative")
        assertTrue(capturedPosition.y >= 0f, "Y coordinate should be non-negative")
        assertTrue(capturedPosition.isFinite, "Position coordinates should be finite")
    }

    @Test
    fun kineticIdentity_behaviorWithNullModifier() {
        // Test behavior when default modifier is used
        composeTestRule.setContent {
            Box(modifier = Modifier.testTag("container")) {
                KineticIdentity() // Using default modifier
            }
        }
        
        // Should render without issues
        composeTestRule.onNodeWithTag("container").assertExists()
        composeTestRule.waitForIdle()
    }
}