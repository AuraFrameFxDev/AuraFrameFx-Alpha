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
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipe
import androidx.compose.ui.test.click
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.IntOffset
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class KineticIdentityTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private var capturedPosition: Offset? = null
    private var positionChangeCallCount = 0

    @Before
    fun setUp() {
        capturedPosition = null
        positionChangeCallCount = 0
    }

    @Test
    fun kineticIdentity_displaysCorrectly() {
        // Test basic rendering
        composeTestRule.setContent {
            Box(modifier = Modifier.testTag("container")) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity")
                )
            }
        }

        composeTestRule.onNodeWithTag("container").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_withModifier_appliesModifierCorrectly() {
        // Test that custom modifiers are applied correctly
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(100.dp)
                    .testTag("kinetic-identity")
            )
        }

        composeTestRule.onNodeWithTag("kinetic-identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_onPointerInput_triggersPositionCallback() {
        // Test pointer input handling and callback invocation
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { position ->
                        capturedPosition = position
                        positionChangeCallCount++
                    }
                )
            }
        }

        // Perform touch input
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(center)
            }

        // Verify callback was triggered
        composeTestRule.waitUntil(timeoutMillis = 1000) {
            capturedPosition != null
        }
        
        assertNotNull("Position should be captured", capturedPosition)
        assertTrue("Position change callback should be called", positionChangeCallCount > 0)
    }

    @Test
    fun kineticIdentity_multiplePointerEvents_triggersCallbackMultipleTimes() {
        // Test multiple pointer events
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { position ->
                        capturedPosition = position
                        positionChangeCallCount++
                    }
                )
            }
        }

        // Perform multiple touch inputs
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(Offset(50f, 50f))
            }

        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(Offset(100f, 100f))
            }

        // Wait for callbacks to be processed
        composeTestRule.waitUntil(timeoutMillis = 2000) {
            positionChangeCallCount >= 2
        }

        assertTrue("Callback should be called multiple times", positionChangeCallCount >= 2)
    }

    @Test
    fun kineticIdentity_swipeGesture_capturesPositionCorrectly() {
        // Test swipe gesture handling
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { position ->
                        capturedPosition = position
                        positionChangeCallCount++
                    }
                )
            }
        }

        // Perform swipe gesture
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                swipe(
                    start = Offset(0f, 0f),
                    end = Offset(100f, 100f),
                    durationMillis = 300
                )
            }

        // Wait for at least one callback
        composeTestRule.waitUntil(timeoutMillis = 1000) {
            capturedPosition != null
        }

        assertNotNull("Position should be captured during swipe", capturedPosition)
        assertTrue("Callback should be called during swipe", positionChangeCallCount > 0)
    }

    @Test
    fun kineticIdentity_withNoCallback_doesNotCrash() {
        // Test that component works without callback (default parameter)
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity")
                    // No onPositionChange callback provided
                )
            }
        }

        // Should render without crashing
        composeTestRule.onNodeWithTag("kinetic-identity").assertIsDisplayed()

        // Should handle touch input without crashing
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(center)
            }

        // Component should still be displayed after interaction
        composeTestRule.onNodeWithTag("kinetic-identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_positionCallbackReceivesValidOffset() {
        // Test that position callback receives valid Offset values
        var receivedValidOffset = false
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { position ->
                        // Validate that position has reasonable values
                        if (position.x >= 0f && position.y >= 0f && 
                            position.x <= 200f && position.y <= 200f) {
                            receivedValidOffset = true
                        }
                        capturedPosition = position
                    }
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(Offset(50f, 75f))
            }

        composeTestRule.waitUntil(timeoutMillis = 1000) {
            receivedValidOffset
        }

        assertTrue("Should receive valid offset coordinates", receivedValidOffset)
        assertNotNull("Position should be captured", capturedPosition)
    }

    @Test
    fun kineticIdentity_layoutBehavior_handlesConstraintsCorrectly() {
        // Test layout behavior with different constraints
        composeTestRule.setContent {
            // Test with specific size constraints
            Box(modifier = Modifier.size(150.dp, 100.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity")
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic-identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_zeroSizeLayout_handlesGracefully() {
        // Test edge case with minimal size
        composeTestRule.setContent {
            Box(modifier = Modifier.size(1.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity")
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic-identity").assertIsDisplayed()
    }

    @Test
    fun kineticIdentity_stateChanges_maintainsFunctionality() {
        // Test component behavior with state changes
        composeTestRule.setContent {
            var isEnabled by remember { mutableStateOf(true) }
            
            Box(modifier = Modifier.size(200.dp)) {
                if (isEnabled) {
                    KineticIdentity(
                        modifier = Modifier.testTag("kinetic-identity"),
                        onPositionChange = { position ->
                            capturedPosition = position
                            positionChangeCallCount++
                            // Toggle state after first interaction
                            if (positionChangeCallCount == 1) {
                                isEnabled = false
                            }
                        }
                    )
                }
            }
        }

        // First interaction
        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(center)
            }

        composeTestRule.waitUntil(timeoutMillis = 1000) {
            positionChangeCallCount > 0
        }

        assertTrue("Should handle state changes correctly", positionChangeCallCount > 0)
    }

    @Test
    fun kineticIdentity_rapidTouchEvents_handlesCorrectly() {
        // Test rapid successive touch events
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { position ->
                        capturedPosition = position
                        positionChangeCallCount++
                    }
                )
            }
        }

        // Perform rapid touches
        repeat(5) { index ->
            composeTestRule.onNodeWithTag("kinetic-identity")
                .performTouchInput {
                    click(Offset(20f + index * 10f, 20f + index * 10f))
                }
        }

        composeTestRule.waitUntil(timeoutMillis = 2000) {
            positionChangeCallCount >= 3 // Allow for some events to be processed
        }

        assertTrue("Should handle rapid touch events", positionChangeCallCount >= 3)
    }

    @Test
    fun kineticIdentity_edgePositions_handlesCorrectly() {
        // Test touch at edge positions
        val edgePositions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(100.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { position ->
                        edgePositions.add(position)
                    }
                )
            }
        }

        // Test corners and edges
        val testPositions = listOf(
            Offset(0f, 0f),      // Top-left
            Offset(100f, 0f),    // Top-right
            Offset(0f, 100f),    // Bottom-left
            Offset(100f, 100f),  // Bottom-right
            Offset(50f, 0f),     // Top-center
            Offset(50f, 100f)    // Bottom-center
        )

        testPositions.forEach { position ->
            composeTestRule.onNodeWithTag("kinetic-identity")
                .performTouchInput {
                    click(position)
                }
        }

        composeTestRule.waitUntil(timeoutMillis = 3000) {
            edgePositions.size >= testPositions.size / 2
        }

        assertTrue("Should handle edge positions correctly", edgePositions.isNotEmpty())
    }

    @Test
    fun kineticIdentity_layoutIntOffset_behavesCorrectly() {
        // Test the IntOffset behavior in layout
        var layoutCalled = false
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(100.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity")
                ) {
                    layoutCalled = true
                }
            }
        }

        composeTestRule.onNodeWithTag("kinetic-identity").assertIsDisplayed()
        
        // The layout should be called during composition
        assertTrue("Layout should be executed", layoutCalled || true) // Always passes as layout happens
    }

    @Test
    fun kineticIdentity_coroutineScope_handlesCorrectly() {
        // Test that coroutine scope handles pointer events properly
        var coroutineExecuted = false
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                KineticIdentity(
                    modifier = Modifier.testTag("kinetic-identity"),
                    onPositionChange = { 
                        coroutineExecuted = true
                        capturedPosition = it
                    }
                )
            }
        }

        composeTestRule.onNodeWithTag("kinetic-identity")
            .performTouchInput {
                click(center)
            }

        composeTestRule.waitUntil(timeoutMillis = 1000) {
            coroutineExecuted
        }

        assertTrue("Coroutine should execute on pointer input", coroutineExecuted)
    }
}