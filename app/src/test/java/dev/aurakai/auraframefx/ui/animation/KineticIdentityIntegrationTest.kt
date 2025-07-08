package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.*

class KineticIdentityIntegrationTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun kineticIdentity_integratesWithComplexLayout() = runTest {
        var positionUpdates = 0
        var lastPosition = Offset.Zero
        
        composeTestRule.setContent {
            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Header Text")
                
                Box(
                    modifier = Modifier
                        .size(200.dp)
                        .background(Color.LightGray)
                        .testTag("integration_container")
                ) {
                    KineticIdentity(
                        onPositionChange = { position ->
                            positionUpdates++
                            lastPosition = position
                        },
                        modifier = Modifier
                            .fillMaxSize()
                            .testTag("integrated_kinetic")
                    ) {
                        Text(
                            "Touch me!",
                            modifier = Modifier
                                .align(Alignment.Center)
                                .testTag("inner_text")
                        )
                    }
                }
                
                Text("Footer Text")
            }
        }
        
        // Verify all components are present
        composeTestRule.onNodeWithText("Header Text").assertExists()
        composeTestRule.onNodeWithText("Touch me!").assertExists()
        composeTestRule.onNodeWithText("Footer Text").assertExists()
        composeTestRule.onNodeWithTag("integrated_kinetic").assertExists()
        
        // Interact with the kinetic component
        composeTestRule.onNodeWithTag("integration_container").performTouchInput {
            down(Offset(100f, 100f))
            up()
        }
        
        composeTestRule.waitForIdle()
        
        // Verify integration worked
        assertTrue("Position updates should have occurred", positionUpdates > 0)
        assertNotEquals("Last position should be updated", Offset.Zero, lastPosition)
    }

    @Test
    fun kineticIdentity_worksWithNestedComposables() = runTest {
        val positionHistory = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .testTag("nested_container")
            ) {
                KineticIdentity(
                    onPositionChange = { position ->
                        positionHistory.add(position)
                    }
                ) {
                    Column {
                        repeat(3) { index ->
                            Row {
                                repeat(3) { subIndex ->
                                    Box(
                                        modifier = Modifier
                                            .size(30.dp)
                                            .background(
                                                if ((index + subIndex) % 2 == 0) Color.Blue else Color.Red
                                            )
                                            .testTag("grid_item_${index}_${subIndex}")
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Verify nested structure exists
        composeTestRule.onNodeWithTag("grid_item_0_0").assertExists()
        composeTestRule.onNodeWithTag("grid_item_1_1").assertExists()
        composeTestRule.onNodeWithTag("grid_item_2_2").assertExists()
        
        // Test interaction with different grid items
        composeTestRule.onNodeWithTag("grid_item_1_1").performTouchInput {
            down(center)
            up()
        }
        
        composeTestRule.waitForIdle()
        
        // Verify kinetic identity responds to nested content interactions
        assertTrue("Position history should contain entries", positionHistory.isNotEmpty())
    }

    @Test
    fun kineticIdentity_performanceWithRapidInteractions() = runTest {
        var interactionCount = 0
        val positions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            KineticIdentity(
                onPositionChange = { position ->
                    interactionCount++
                    positions.add(position)
                },
                modifier = Modifier
                    .size(300.dp)
                    .testTag("performance_test")
            )
        }
        
        // Simulate rapid interactions
        repeat(10) { index ->
            val x = (index * 30f).coerceAtMost(270f)
            val y = (index * 30f).coerceAtMost(270f)
            
            composeTestRule.onNodeWithTag("performance_test").performTouchInput {
                down(Offset(x, y))
                up()
            }
        }
        
        composeTestRule.waitForIdle()
        
        // Verify performance characteristics
        assertTrue("Should handle multiple rapid interactions", interactionCount > 0)
        assertTrue("Position list should contain multiple entries", positions.size > 0)
        assertEquals("Interaction count should match position count", interactionCount, positions.size)
    }

    @Test
    fun kineticIdentity_accessibilitySupport() {
        composeTestRule.setContent {
            KineticIdentity(
                modifier = Modifier
                    .size(150.dp)
                    .testTag("accessibility_test")
            ) {
                Text(
                    "Accessible Content",
                    modifier = Modifier.testTag("accessible_text")
                )
            }
        }
        
        // Verify accessibility features work with kinetic identity
        composeTestRule.onNodeWithTag("accessible_text")
            .assertExists()
            .assertIsDisplayed()
            .assertTextEquals("Accessible Content")
        
        // Verify the kinetic component doesn't interfere with accessibility
        composeTestRule.onNodeWithTag("accessibility_test")
            .assertExists()
            .assertIsDisplayed()
    }
}