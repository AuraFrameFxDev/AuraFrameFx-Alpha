package dev.aurakai.auraframefx.utils

import java.time.Clock
import java.time.Duration
import java.time.Instant

/**
 * Time utilities for the AuraFrameFX system
 */
object TimeUtils {
    val systemClock: Clock = Clock.systemUTC()
    
    /**
 * Retrieves the current instant in UTC using the system clock.
 *
 * @return The current time as an [Instant].
 */
fun now(): Instant = Instant.now(systemClock)
    
    /**
 * Returns the current time in milliseconds since the Unix epoch.
 *
 * @return The number of milliseconds elapsed since January 1, 1970 UTC.
 */
fun currentTimestamp(): Long = System.currentTimeMillis()
    
    /**
     * Calculates the duration that has elapsed since the specified instant.
     *
     * @param instant The starting point in time from which to measure.
     * @return The duration between the provided instant and the current time.
     */
    fun durationSince(instant: Instant): Duration {
        return Duration.between(instant, now())
    }
}

// Type aliases for common time types
typealias AuraInstant = Instant
typealias AuraClock = Clock
typealias AuraDuration = Duration
