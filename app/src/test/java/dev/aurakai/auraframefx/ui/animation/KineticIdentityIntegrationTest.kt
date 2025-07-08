package dev.aurakai.auraframefx.ui.animation

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.*
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
class KineticIdentityIntegrationTest {
    
    @get:Rule
    val composeTestRule = createComposeRule()
    
    @Test
    fun multipleKineticIdentities_interaction_worksIndependently() {
        val positions1 = mutableListOf<Offset>()
        val positions2 = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Column(modifier = Modifier.fillMaxSize()) {
                KineticIdentity(
                    modifier = Modifier
                        .size(100.dp)
                        .testTag("kinetic1"),
                    onPositionChange = { positions1.add(it) }
                )
                Spacer(modifier = Modifier.height(20.dp))
                KineticIdentity(
                    modifier = Modifier
                        .size(100.dp)
                        .testTag("kinetic2"),
                    onPositionChange = { positions2.add(it) }
                )
            }
        }
        
        composeTestRule.onNodeWithTag("kinetic1")
            .performTouchInput {
                down(Offset(25f, 25f))
                up()
            }
        
        composeTestRule.onNodeWithTag("kinetic2")
            .performTouchInput {
                down(Offset(75f, 75f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        assertTrue("First component should capture events", positions1.isNotEmpty())
        assertTrue("Second component should capture events", positions2.isNotEmpty())
    }
    
    @Test
    fun kineticIdentityInScrollableContent_maintainsInteractivity() {
        val capturedPositions = mutableListOf<Offset>()
        
        composeTestRule.setContent {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .testTag("scrollable-content")
            ) {
                repeat(5) { index ->
                    KineticIdentity(
                        modifier = Modifier
                            .size(80.dp)
                            .testTag("kinetic-$index"),
                        onPositionChange = { capturedPositions.add(it) }
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }
        }
        
        composeTestRule.onNodeWithTag("kinetic-2")
            .performTouchInput {
                down(Offset(40f, 40f))
                up()
            }
        
        composeTestRule.waitForIdle()
        
        assertTrue(
            "Should maintain interactivity in scrollable content",
            capturedPositions.isNotEmpty()
        )
    }
    
    @Test
    fun kineticIdentity_performanceUnderLoad_remainsResponsive() {
        var totalEvents = 0
        val testTag = "performance-test"
        
        composeTestRule.setContent {
            Box(modifier = Modifier.size(200.dp)) {
                repeat(3) { i ->
                    KineticIdentity(
                        modifier = Modifier
                            .size((150 - i * 30).dp)
                            .testTag("$testTag-$i"),
                        onPositionChange = { totalEvents++ }
                    )
                }
            }
        }
        
        repeat(3) { i ->
            composeTestRule.onNodeWithTag("$testTag-$i")
                .performTouchInput {
                    repeat(5) { j ->
                        down(Offset(20f + j * 10f, 20f + j * 10f))
                        up()
                    }
                }
        }
        
        composeTestRule.waitForIdle()
        
        assertTrue("Should handle intensive interactions", totalEvents > 0)
        
        repeat(3) { i ->
            composeTestRule.onNodeWithTag("$testTag-$i").assertExists()
        }
    }
}