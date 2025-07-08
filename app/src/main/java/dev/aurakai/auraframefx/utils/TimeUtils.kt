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
 * Returns the current instant in time using the system UTC clock.
 *
 * @return The current time as an [Instant].
 */
fun now(): Instant = Instant.now(systemClock)
    
    /**
 * Returns the current system time in milliseconds since the Unix epoch (January 1, 1970 UTC).
 *
 * @return The number of milliseconds since the Unix epoch.
 */
fun currentTimestamp(): Long = System.currentTimeMillis()
    
    /**
     * Calculates the elapsed duration from the specified instant to the current time.
     *
     * @param instant The starting instant to measure from.
     * @return The duration between the given instant and the current instant.
     */
    fun durationSince(instant: Instant): Duration {
        return Duration.between(instant, now())
    }
}

// Type aliases for common time types
typealias AuraInstant = Instant
typealias AuraClock = Clock
typealias AuraDuration = Duration
