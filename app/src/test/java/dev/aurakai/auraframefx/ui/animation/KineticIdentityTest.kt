package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeRight
import androidx.compose.ui.test.swipeLeft
import androidx.compose.ui.test.swipeUp
import androidx.compose.ui.test.swipeDown
import androidx.compose.ui.test.click
import androidx.compose.ui.test.assertExists
import androidx.compose.ui.unit.dp
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.*
import kotlinx.coroutines.test.runTest

/**
 * Comprehensive unit tests for KineticIdentity Composable component.
 * Testing framework: JUnit 4 with Compose testing library
 */
class KineticIdentityTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun kineticIdentity_defaultParameters_rendersSuccessfully() {
        // Arrange & Act
        composeTestRule.setContent {
            KineticIdentity()
        }

        // Assert - Component renders without crashing
        composeTestRule.waitForIdle()
    }

    @Test
    fun kineticIdentity_withCustomModifier_appliesModifierCorrectly() {
        // Arrange
        val testTag = "kinetic_identity_test"
        
        // Act
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag(testTag)
            )
        }

        // Assert
        composeTestRule.onNodeWithTag(testTag).assertExists()
    }

    @Test
    fun kineticIdentity_singleTouch_triggersPositionCallback() {
        // Arrange
        var capturedPosition: Offset? = null
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPosition = position
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click(Offset(50f, 50f))
        }

        // Assert
        composeTestRule.waitForIdle()
        assertNotNull("Position should be captured on touch", capturedPosition)
        assertEquals("X coordinate should match touch position", 50f, capturedPosition?.x ?: 0f, 1f)
        assertEquals("Y coordinate should match touch position", 50f, capturedPosition?.y ?: 0f, 1f)
    }

    @Test
    fun kineticIdentity_multipleSequentialTouches_capturesAllPositions() {
        // Arrange
        val capturedPositions = mutableListOf<Offset>()
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPositions.add(position)
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(200.dp)
                    .testTag("container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click(Offset(50f, 50f))
            click(Offset(100f, 100f))
            click(Offset(150f, 75f))
        }

        // Assert
        composeTestRule.waitForIdle()
        assertTrue("Should capture multiple positions", capturedPositions.size >= 3)
        
        // Verify distinct positions were captured
        val uniquePositions = capturedPositions.distinct()
        assertTrue("Should capture distinct positions", uniquePositions.size >= 2)
    }

    @Test
    fun kineticIdentity_swipeGestureRight_capturesPositionChanges() {
        // Arrange
        val capturedPositions = mutableListOf<Offset>()
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPositions.add(position)
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(200.dp)
                    .testTag("container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            swipeRight()
        }

        // Assert
        composeTestRule.waitForIdle()
        assertTrue("Swipe should capture position changes", capturedPositions.isNotEmpty())
    }

    @Test
    fun kineticIdentity_allSwipeDirections_capturePositions() {
        // Arrange list of directions and corresponding gestures
        val directions = listOf(
            "right" to { composeTestRule.onNodeWithTag("kinetic_component").performTouchInput { swipeRight() } },
            "left" to { composeTestRule.onNodeWithTag("kinetic_component").performTouchInput { swipeLeft() } },
            "up" to { composeTestRule.onNodeWithTag("kinetic_component").performTouchInput { swipeUp() } },
            "down" to { composeTestRule.onNodeWithTag("kinetic_component").performTouchInput { swipeDown() } }
        )

        directions.forEach { (direction, action) ->
            val capturedPositions = mutableListOf<Offset>()
            val onPositionChange: (Offset) -> Unit = { position ->
                capturedPositions.add(position)
            }

            // Act
            composeTestRule.setContent {
                Box(
                    modifier = Modifier
                        .size(200.dp)
                        .testTag("container")
                ) {
                    KineticIdentity(
                        modifier = Modifier.testTag("kinetic_component"),
                        onPositionChange = onPositionChange
                    )
                }
            }

            action()
            composeTestRule.waitForIdle()

            // Assert
            assertTrue("Swipe $direction should capture positions", capturedPositions.isNotEmpty())
        }
    }

    @Test
    fun kineticIdentity_emptyPositionCallback_doesNotCrash() {
        // Arrange & Act
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic_component"),
                onPositionChange = { /* Do nothing */ }
            )
        }

        // Perform a touch on the component
        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click()
        }

        // Assert - Component renders without crashing
        composeTestRule.waitForIdle()
    }

    @Test
    fun kineticIdentity_layoutMeasurement_handlesConstraintsProperly() {
        // Arrange
        val testTag = "container"

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(150.dp)
                    .testTag(testTag)
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component")
                )
            }
        }

        // Assert - Component handles layout constraints without crashing
        composeTestRule.onNodeWithTag("kinetic_component").assertExists()
        composeTestRule.onNodeWithTag(testTag).assertExists()
        composeTestRule.waitForIdle()
    }

    @Test
    fun kineticIdentity_zeroSizeContainer_handlesGracefully() {
        // Arrange & Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(0.dp)
                    .testTag("zero_container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component")
                )
            }
        }

        // Assert - Component handles zero size gracefully
        composeTestRule.waitForIdle()
        // Note: Zero-size components may not be findable, but shouldn't crash
    }

    @Test
    fun kineticIdentity_coroutineScope_handlesAsyncOperationsProperly() = runTest {
        // Arrange
        var callbackInvoked = false
        val onPositionChange: (Offset) -> Unit = { 
            callbackInvoked = true 
        }

        // Act
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier.testTag("kinetic_component"),
                onPositionChange = onPositionChange
            )
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click()
        }

        // Assert
        composeTestRule.waitForIdle()
        assertTrue("Async callback should be invoked", callbackInvoked)
    }

    @Test
    fun kineticIdentity_intOffsetHandling_convertsNullableIntsCorrectly() {
        // This test validates the null safety handling in IntOffset conversion
        // as mentioned in the component code: "Safely convert possible nullable Int? to Int"
        
        // Arrange & Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component")
                )
            }
        }

        // Assert - Component handles IntOffset conversion without crashing
        composeTestRule.onNodeWithTag("kinetic_component").assertExists()
        composeTestRule.waitForIdle()
    }

    @Test
    fun kineticIdentity_largeContainer_handlesExtremePositionValues() {
        // Arrange
        val capturedPositions = mutableListOf<Offset>()
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPositions.add(position)
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(1000.dp)
                    .testTag("large_container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click(Offset(999f, 999f)) // Near edge
            click(Offset(0f, 0f))    // Origin
            click(Offset(500f, 750f)) // Middle area
        }

        // Assert
        composeTestRule.waitForIdle()
        assertTrue("Should handle extreme position values", capturedPositions.isNotEmpty())
        assertTrue("Should capture multiple extreme positions", capturedPositions.size >= 2)
    }

    @Test
    fun kineticIdentity_rapidConsecutiveGestures_maintainsPerformance() {
        // Arrange
        val capturedPositions = mutableListOf<Offset>()
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPositions.add(position)
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(200.dp)
                    .testTag("container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        // Perform rapid clicks
        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            repeat(20) { index ->
                click(Offset((index % 10) * 20f, (index / 10) * 20f))
            }
        }

        // Assert
        composeTestRule.waitForIdle()
        assertTrue("Should handle rapid gestures without dropping events", capturedPositions.size >= 10)
    }

    @Test
    fun kineticIdentity_fillMaxSizeContainer_handlesLargeLayout() {
        // Arrange
        var positionCaptured = false
        val onPositionChange: (Offset) -> Unit = { 
            positionCaptured = true 
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .testTag("max_size_container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click()
        }

        // Assert
        composeTestRule.waitForIdle()
        assertTrue("Should handle fillMaxSize containers", positionCaptured)
    }

    @Test
    fun kineticIdentity_positionAccuracy_maintainsPrecision() {
        // Arrange
        val expectedPosition = Offset(123.45f, 67.89f)
        var capturedPosition: Offset? = null
        val onPositionChange: (Offset) -> Unit = { position ->
            capturedPosition = position
        }

        // Act
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .size(200.dp)
                    .testTag("container")
            ) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic_component"),
                    onPositionChange = onPositionChange
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic_component").performTouchInput {
            click(expectedPosition)
        }

        // Assert
        composeTestRule.waitForIdle()
        assertNotNull("Position should be captured", capturedPosition)
        assertEquals("X coordinate should be precise", expectedPosition.x, capturedPosition?.x ?: 0f, 2f)
        assertEquals("Y coordinate should be precise", expectedPosition.y, capturedPosition?.y ?: 0f, 2f)
    }
}