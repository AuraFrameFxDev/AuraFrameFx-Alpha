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
 * Retrieves the current instant in time using the system UTC clock.
 *
 * @return The current time as an [Instant].
 */
fun now(): Instant = Instant.now(systemClock)
    
    /**
 * Returns the current system time in milliseconds since the Unix epoch.
 *
 * @return The number of milliseconds elapsed since January 1, 1970 UTC.
 */
fun currentTimestamp(): Long = System.currentTimeMillis()
    
    /**
     * Returns the duration that has elapsed from the given instant to the current time.
     *
     * @param instant The starting instant from which to measure elapsed time.
     * @return The duration between the specified instant and now.
     */
    fun durationSince(instant: Instant): Duration {
        return Duration.between(instant, now())
    }
}

// Type aliases for common time types
typealias AuraInstant = Instant
typealias AuraClock = Clock
typealias AuraDuration = Duration
