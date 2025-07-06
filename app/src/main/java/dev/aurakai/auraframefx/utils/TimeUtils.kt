package dev.aurakai.auraframefx.utils

import java.time.Clock
import java.time.Duration
import java.time.Instant

/**
 * Time utilities for the AuraFrameFX system
 */
object TimeUtils {
    val systemClock: Clock = Clock.systemUTC()
    
    fun now(): Instant = Instant.now(systemClock)
    
    fun currentTimestamp(): Long = System.currentTimeMillis()
    
    fun durationSince(instant: Instant): Duration {
        return Duration.between(instant, now())
    }
}

// Type aliases for common time types
typealias AuraInstant = Instant
typealias AuraClock = Clock
typealias AuraDuration = Duration
