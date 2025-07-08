package dev.aurakai.auraframefx.ui.animation

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.unit.IntOffset
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

@RunWith(JUnit4::class)
class KineticIdentityTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun kineticIdentity_rendersWithoutCrashing() {
        // Given
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic_identity")
            )
        }

        // When & Then
        composeTestRule.onNodeWithTag("kinetic_identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_defaultModifier_appliesCorrectly() {
        // Given
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic_identity")
            )
        }

        // When & Then
        composeTestRule.onNodeWithTag("kinetic_identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_customModifier_appliesCorrectly() {
        // Given
        val customModifier = Modifier.testTag("custom_kinetic_identity")
        
        composeTestRule.setContent {
            KineticIdentity(
                modifier = customModifier
            )
        }

        // When & Then
        composeTestRule.onNodeWithTag("custom_kinetic_identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_onPositionChangeCallback_triggersOnPointerInput() {
        // Given
        var capturedPosition: Offset? = null
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPosition = position
        }

        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic_identity"),
                onPositionChange = onPositionChange
            )
        }

        // When
        composeTestRule.onNodeWithTag("kinetic_identity").performTouchInput {
            down(Offset(100f, 200f))
            up()
        }

        // Then
        composeTestRule.waitForIdle()
        assertNotNull(capturedPosition)
    }

    @Test
    fun kineticIdentity_multiplePointerEvents_callsOnPositionChangeMultipleTimes() {
        // Given
        val capturedPositions = mutableListOf<Offset>()
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPositions.add(position)
        }

        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic_identity"),
                onPositionChange = onPositionChange
            )
        }

        // When
        composeTestRule.onNodeWithTag("kinetic_identity").performTouchInput {
            down(Offset(50f, 100f))
            up()
        }
        composeTestRule.onNodeWithTag("kinetic_identity").performTouchInput {
            down(Offset(150f, 250f))
            up()
        }

        // Then
        composeTestRule.waitForIdle()
        assertEquals(2, capturedPositions.size)
    }

    @Test
    fun kineticIdentity_defaultOnPositionChange_doesNotCrash() {
    
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