package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {
    
    /**
     * Log debug information for development and troubleshooting.
     */
    fun debug(tag: String, message: String, throwable: Throwable? = null)
    
    /**
     * Log informational messages for normal operation tracking.
     */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
     * Log warning messages for potential issues.
     */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
     * Log error messages for failures and exceptions.
     */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
     * Log critical security events that require immediate attention.
     */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
     * Log performance metrics and timing information.
     */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
     * Log user interactions for analytics and UX optimization.
     */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
     * Log AI operations and decision making processes.
     */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
     * Enable or disable logging at runtime.
     */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
     * Set minimum log level to reduce noise.
     */
    fun setLogLevel(level: LogLevel)
    
    /**
     * Force flush all pending log entries to storage.
     */
    suspend fun flush()
    
    /**
     * Clean up resources and stop logging.
     */
    fun cleanup()
}

/**
 * Log levels for filtering and prioritization.
 */
enum class LogLevel(val priority: Int) {
    DEBUG(0),
    INFO(1),
    WARN(2),
    ERROR(3),
    SECURITY(4)
}
