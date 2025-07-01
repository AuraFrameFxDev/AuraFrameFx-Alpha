package dev.aurakai.auraframefx.ui.animation

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import dev.aurakai.auraframefx.ui.theme.NeonBlue
import dev.aurakai.auraframefx.ui.theme.NeonCyan
import dev.aurakai.auraframefx.ui.theme.NeonPink
import kotlin.random.Random

/**
 * Custom Digital Transitions inspired by cyberpunk aesthetics
 * These transitions create a "digitalization" effect for screen changes
 */
object DigitalTransitions {

    /**
     * Creates a Navigation Animation to materialize content with a digital effect
     */
    val EnterDigitalMaterialization: EnterTransition = fadeIn(
        animationSpec = tween(
            durationMillis = 500,
            easing = FastOutSlowInEasing
        )
    ) + scaleIn(
        animationSpec = tween(
            durationMillis = 400,
            delayMillis = 100,
            easing = FastOutSlowInEasing
        ),
        initialScale = 0.95f
    )

    /**
     * Creates a Navigation Animation to dematerialize content with a digital effect
     */
    val ExitDigitalDematerialization: ExitTransition = fadeOut(
        animationSpec = tween(
            durationMillis = 300,
            easing = FastOutLinearInEasing
        )
    ) + scaleOut(
        animationSpec = tween(
            durationMillis = 300,
            easing = FastOutLinearInEasing
        ),
        targetScale = 1.05f
    )

    /**
     * Applies a digital pixelation effect modifier to a composable
     * Creates a dissolving/materializing effect using digital particles
     */
    fun Modifier.digitalPixelEffect(
        visible: Boolean = true,
        pixelsCount: Int = 400,
        colors: List<Color> = listOf(NeonBlue, NeonPink, NeonCyan),
    ) = composed {
        val animatedVisible by animateFloatAsState(
            targetValue = if (visible) 1f else 0f,
            animationSpec = tween(durationMillis = 800, easing = FastOutSlowInEasing),
            label = "pixelationAnimation"
        )

        this.drawWithCache {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val pixelSize = 4f
            val random = Random(System.currentTimeMillis().toInt())

            onDrawWithContent {
                drawContent()

                if (animatedVisible > 0.01f) {
                    // Create digital materialization effect
                    val numPixels = (pixelsCount * animatedVisible).toInt()
                    for (i in 0 until numPixels) {
                        val x = random.nextFloat() * canvasWidth
                        val y = random.nextFloat() * canvasHeight
                        val color = colors[random.nextInt(colors.size)]

                        drawRect(
                            color = color.copy(alpha = random.nextFloat() * 0.5f * animatedVisible),
                            topLeft = Offset(x, y),
                            size = Size(
                                pixelSize * random.nextFloat(),
                                pixelSize * random.nextFloat()
                            ),
                            blendMode = BlendMode.Screen
                        )
                    }

                    // Add some digital circuit lines
                    for (i in 0 until (15 * animatedVisible).toInt()) {
                        val startX = random.nextFloat() * canvasWidth
                        val startY = random.nextFloat() * canvasHeight
                        val endX = startX + random.nextFloat() * canvasWidth / 3
                        val endY = startY + (random.nextFloat() - 0.5f) * 60

                        drawLine(
                            color = colors[random.nextInt(colors.size)].copy(alpha = 0.6f * animatedVisible),
                            start = Offset(startX, startY),
                            end = Offset(endX, endY),
                            strokeWidth = 2f,
                            pathEffect = PathEffect.dashPathEffect(floatArrayOf(5f, 3f), 0f),
                            blendMode = BlendMode.Screen
                        )
                    }
                }
            }
        }
    }

    /**
     * Applies a scanning line effect similar to old CRT monitors
     */
    fun Modifier.digitalScanlineEffect(visible: Boolean = true) = composed {
        var offsetY by remember { mutableFloatStateOf(0f) }
        val infiniteTransition = rememberInfiniteTransition(label = "scanline")
        val scanlinePosition by infiniteTransition.animateFloat(
            initialValue = -200f,
            targetValue = 2000f,
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 2000, easing = LinearEasing),
                repeatMode = RepeatMode.Restart
            ),
            label = "scanlinePosition"
        )

        this.drawWithCache {
            val canvasHeight = size.height
            val scanlineHeight = 2f

            onDrawWithContent {
                drawContent()

                if (visible) {
                    offsetY = scanlinePosition

                    // Draw scan line
                    drawRect(
                        color = Color.White.copy(alpha = 0.2f),
                        topLeft = Offset(0f, offsetY % canvasHeight),
                        size = Size(size.width, scanlineHeight),
                        blendMode = BlendMode.Screen
                    )
                }
            }
        }
    }

    /**
     * Applies an edge glow effect around a composable
     * Creates a cyberpunk-style glowing border
     */
    fun Modifier.cyberEdgeGlow(
        visible: Boolean = true,
        primaryColor: Color = NeonCyan,
        secondaryColor: Color = NeonPink,
        cornerRadius: Float = 12f,
    ) = composed {
        val animatedVisible by animateFloatAsState(
            targetValue = if (visible) 1f else 0f,
            animationSpec = tween(durationMillis = 800),
            label = "edgeGlowAnimation"
        )

        val infiniteTransition = rememberInfiniteTransition(label = "glow")
        val glowIntensity by infiniteTransition.animateFloat(
            initialValue = 0.7f,
            targetValue = 1.0f,
            animationSpec = infiniteRepeatable(
                animation = tween(1000, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "glowIntensity"
        )

        this.drawWithCache {
            val width = size.width
            val height = size.height

            val borderPath = Path().apply {
                moveTo(cornerRadius, 0f)
                lineTo(width - cornerRadius, 0f)

                // Top-right corner
                cubicTo(
                    width, 0f,
                    width, 0f,
                    width, cornerRadius
                )

                lineTo(width, height - cornerRadius)

                // Bottom-right corner
                cubicTo(
                    width, height,
                    width, height,
                    width - cornerRadius, height
                )

                lineTo(cornerRadius, height)

                // Bottom-left corner
                cubicTo(
                    0f, height,
                    0f, height,
                    0f, height - cornerRadius
                )

                lineTo(0f, cornerRadius)

                // Top-left corner
                cubicTo(
                    0f, 0f,
                    0f, 0f,
                    cornerRadius, 0f
                )

                close()
            }

            onDrawWithContent {
                drawContent()

                if (animatedVisible > 0.01f) {
                    // Draw inner border
                    drawPath(
                        path = borderPath,
                        color = primaryColor.copy(alpha = 0.8f * glowIntensity * animatedVisible),
                        style = Stroke(width = 2f),
                        blendMode = BlendMode.Screen
                    )

                    // Draw outer glow
                    drawPath(
                        path = borderPath,
                        color = secondaryColor.copy(alpha = 0.5f * glowIntensity * animatedVisible),
                        style = Stroke(width = 4f),
                        blendMode = BlendMode.Screen
                    )

                    // Draw nodes at corners
                    val nodeSize = 6f
                    val corners = listOf(
                        Offset(cornerRadius, 0f),
                        Offset(width - cornerRadius, 0f),
                        Offset(width, cornerRadius),
                        Offset(width, height - cornerRadius),
                        Offset(width - cornerRadius, height),
                        Offset(cornerRadius, height),
                        Offset(0f, height - cornerRadius),
                        Offset(0f, cornerRadius)
                    )

                    corners.forEachIndexed { index, offset ->
                        // Alternate colors for corner nodes
                        val nodeColor = if (index % 2 == 0) primaryColor else secondaryColor
                        drawCircle(
                            color = nodeColor.copy(alpha = glowIntensity * animatedVisible),
                            radius = nodeSize,
                            center = offset,
                            blendMode = BlendMode.Screen
                        )
                    }
                }
            }
        }
    }

    /**
     * Digital glitch effect that randomly displaces small portions of the UI
     */
    @Composable
    fun Modifier.digitalGlitchEffect(visible: Boolean = true): Modifier {
        if (!visible) return this

        val infiniteTransition = rememberInfiniteTransition(label = "glitch")
        val glitchTiming by infiniteTransition.animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(5000),
                repeatMode = RepeatMode.Restart
            ),
            label = "glitchTiming"
        )

        return this.then(
            Modifier.drawWithCache {
                val width = size.width
                val height = size.height
                val random = Random(glitchTiming.hashCode())

                // Determine if a glitch should occur (randomly with timing)
                val shouldGlitch = glitchTiming > 0.95f ||
                        (glitchTiming > 0.4f && glitchTiming < 0.42f) ||
                        (glitchTiming > 0.7f && glitchTiming < 0.71f)

                onDrawWithContent {
                    if (!shouldGlitch) {
                        drawContent()
                        return@onDrawWithContent
                    }

                    // Create glitch by shifting random horizontal slices
                    val sliceCount = 8
                    val sliceHeight = height / sliceCount

                    for (i in 0 until sliceCount) {
                        val yOffset = i * sliceHeight
                        val xOffset = if (random.nextFloat() > 0.7f) {
                            (random.nextFloat() * 20 - 10)
                        } else 0f

                        val glitchColor = if (random.nextFloat() > 0.9f) {
                            when (random.nextInt(3)) {
                                0 -> NeonCyan.copy(alpha = 0.3f)
                                1 -> NeonPink.copy(alpha = 0.3f)
                                else -> NeonBlue.copy(alpha = 0.3f)
                            }
                        } else null

                        // Save the canvas state
                        save()

                        // Clip to just this slice
                        clipRect(
                            left = 0f,
                            top = yOffset,
                            right = width,
                            bottom = yOffset + sliceHeight
                        )

                        // Translate to create offset effect
                        translate(left = xOffset, top = 0f)

                        // Draw the content for this slice
                        drawContent()

                        // Add color overlay for some slices
                        glitchColor?.let {
                            drawRect(
                                color = it,
                                topLeft = Offset(0f, yOffset),
                                size = Size(width, sliceHeight),
                                blendMode = BlendMode.Screen
                            )
                        }

                        // Restore canvas for next slice
                        restore()
                    }
                }
            }
        )
    }
}
