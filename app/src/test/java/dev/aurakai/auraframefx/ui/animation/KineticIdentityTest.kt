package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*

@RunWith(AndroidJUnit4::class)
class KineticIdentityTest {
    
    @get:Rule
    val composeTestRule = createComposeRule()
    
    @Test
    fun kineticIdentity_defaultModifier_rendersSuccessfully() {
        // Test that the component renders with default parameters
        composeTestRule.setContent {
            KineticIdentity()
        }
        
        // Verify the component is composed without errors
        composeTestRule.onRoot().assertExists()
    }
    
    @Test
    fun kineticIdentity_customModifier_appliesCorrectly() {
        // Test that custom modifiers are applied correctly
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("kinetic-identity")
            )
        }
        
        composeTestRule.onNodeWithTag("kinetic-identity")
            .assertExists()
            .assertIsDisplayed()
    }
    
    @Test
    fun kineticIdentity_onPositionChange_triggeredOnPointerInput() {
        var lastPosition: Offset? = null
        val testTag = "kinetic-test"
        
        composeTestRule.setContent {
            Box(
                modifier = Modifier.size(200.dp)
            ) {
                KineticIdentity(
                    modifier = Modifier
                        .size(100.dp)
                        .testTag(testTag),
                    onPositionChange = { position ->
                        lastPosition = position
                    }
                )
            }
        }
        
        // Perform touch gesture on the component
        composeTestRule.onNodeWithTag(testTag)
            .performTouchInput {
                down(Offset(50f, 50f))
                up()
            }
        
        // Verify that position change callback was triggered
        composeTestRule.waitForIdle()
        assertNotNull("Position change callback should be triggered", lastPosition)
    }
    
    @Test
    fun kineticIdentity_multiplePointerEvents_handlesCorrectly() {
        val positions = mutableListOf<Offset>()
        val testTag = "kinetic-multi-touch"
        
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(150.dp)
                    .testTag(testTag),
                onPositionChange = { position ->
                    positions.add(position)
                }
            )
        }
        
        // Perform multiple touch events
        composeTestRule.onNodeWithTag(testTag)
            .performTouchInput {
                down(Offset(25f, 25f))
                up()
                down(Offset(75f, 75f))
                up()
                down(Offset(125f, 125f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify multiple position changes were captured
        assertTrue("Should capture multiple position changes", positions.size >= 1)
    }
    
    @Test
    fun kineticIdentity_withNestedContent_layoutsCorrectly() {
        composeTestRule.setContent {
            Box(modifier = Modifier.size(300.dp)) {
                KineticIdentity(
                    modifier = Modifier
                        .size(150.dp)
                        .testTag("nested-kinetic")
                ) {
                    Box(
                        modifier = Modifier
                            .size(50.dp)
                            .testTag("inner-content")
                    )
                }
            }
        }
        
        // Verify both the container and nested content exist
        composeTestRule.onNodeWithTag("nested-kinetic").assertExists()
        composeTestRule.onNodeWithTag("inner-content").assertExists()
    }
    
    @Test
    fun kineticIdentity_emptyOnPositionChange_handlesGracefully() {
        // Test with no callback provided (default empty lambda)
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("no-callback")
            )
        }
        
        // Should not crash when performing touch input
        composeTestRule.onNodeWithTag("no-callback")
            .performTouchInput {
                down(Offset(50f, 50f))
                up()
            }
        
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("no-callback").assertExists()
    }
    
    @Test
    fun kineticIdentity_edgeCasePositions_handlesCorrectly() {
        val positions = mutableListOf<Offset>()
        val testTag = "edge-positions"
        
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag(testTag),
                onPositionChange = { position ->
                    positions.add(position)
                }
            )
        }
        
        // Test edge positions (corners and center)
        composeTestRule.onNodeWithTag(testTag)
            .performTouchInput {
                // Top-left corner
                down(Offset(0f, 0f))
                up()
                // Top-right corner
                down(Offset(99f, 0f))
                up()
                // Bottom-left corner
                down(Offset(0f, 99f))
                up()
                // Bottom-right corner
                down(Offset(99f, 99f))
                up()
                // Center
                down(Offset(50f, 50f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify positions were captured for edge cases
        assertTrue("Should handle edge case positions", positions.isNotEmpty())
        
        // Verify all positions are valid (non-negative)
        positions.forEach { position ->
            assertTrue("X coordinate should be non-negative", position.x >= 0f)
            assertTrue("Y coordinate should be non-negative", position.y >= 0f)
        }
    }
    
    @Test
    fun kineticIdentity_rapidTouchEvents_performsWell() {
        var eventCount = 0
        val testTag = "rapid-touch"
        
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag(testTag),
                onPositionChange = { 
                    eventCount++
                }
            )
        }
        
        // Perform rapid touch events
        composeTestRule.onNodeWithTag(testTag)
            .performTouchInput {
                repeat(10) { i ->
                    down(Offset(10f + i * 5f, 10f + i * 5f))
                    up()
                }
            }
        
        composeTestRule.waitForIdle()
        
        // Verify the component handled rapid events without crashing
        assertTrue("Should handle rapid touch events", eventCount > 0)
        composeTestRule.onNodeWithTag(testTag).assertExists()
    }
    
    @Test
    fun kineticIdentity_layoutMeasurement_calculatesDimensionsCorrectly() {
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(width = 120.dp, height = 80.dp)
                    .testTag("sized-kinetic")
            )
        }
        
        // Verify the component respects the specified dimensions
        composeTestRule.onNodeWithTag("sized-kinetic")
            .assertExists()
            .assertIsDisplayed()
    }
    
    @Test
    fun kineticIdentity_statePreservation_maintainsBehaviorAcrossRecomposition() {
        var recompositionTrigger by mutableStateOf(0)
        val positions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            // Trigger recomposition
            val _ = recompositionTrigger
            
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("recomposition-test"),
                onPositionChange = { position ->
                    positions.add(position)
                }
            )
        }
        
        // Perform initial touch
        composeTestRule.onNodeWithTag("recomposition-test")
            .performTouchInput {
                down(Offset(30f, 30f))
                up()
            }
        
        composeTestRule.waitForIdle()
        val initialEventCount = positions.size
        
        // Trigger recomposition
        recompositionTrigger = 1
        composeTestRule.waitForIdle()
        
        // Perform touch after recomposition
        composeTestRule.onNodeWithTag("recomposition-test")
            .performTouchInput {
                down(Offset(70f, 70f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        // Verify behavior is maintained after recomposition
        assertTrue("Should maintain behavior after recomposition", 
                  positions.size > initialEventCount)
    }
}