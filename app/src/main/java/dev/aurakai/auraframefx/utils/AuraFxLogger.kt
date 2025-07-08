package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting.
 *
 * @param tag Identifies the source or component of the log entry.
 * @param message The message to log.
 * @param throwable An optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message indicating normal application operation.
 *
 * @param tag Identifier for the log source or component.
 * @param message The message describing the informational event.
 * @param throwable Optional exception or error associated with the event.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message to indicate a potential issue or abnormal condition.
 *
 * @param tag The category or source of the warning.
 * @param message The warning message content.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message to indicate a failure or exception.
 *
 * Use this method to record errors that require attention or investigation.
 *
 * @param tag The category or source of the log entry.
 * @param message The error message to log.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event for immediate attention.
 *
 * Use this method to record security-related incidents or breaches.
 *
 * @param tag The category or source of the security event.
 * @param message The security-related message to log.
 * @param throwable An optional exception or error associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs performance metrics for an operation, including its duration and optional contextual metadata.
     *
     * @param tag Identifier for the source or component generating the log.
     * @param operation Name or description of the operation being measured.
     * @param durationMs Duration of the operation in milliseconds.
     * @param metadata Additional contextual data for the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Logs a user interaction event for analytics and user experience tracking.
 *
 * @param tag Identifier for the category or component related to the event.
 * @param action The specific user action to record.
 * @param metadata Optional contextual data about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation with its name, confidence score, and optional metadata for analytics or auditing.
     *
     * @param tag Identifies the category or component related to the AI operation.
     * @param operation Describes the AI operation performed.
     * @param confidence The confidence score associated with the AI result.
     * @param metadata Additional context or details about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables logging at runtime.
 *
 * When disabled, no log entries will be recorded until logging is re-enabled.
 *
 * @param enabled True to enable logging; false to disable it.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for recording log entries.
 *
 * Only log messages at or above the specified level will be processed and stored.
 *
 * @param level The minimum log level to enable.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Forces all buffered log entries to be persisted to storage.
 *
 * This suspend function may perform I/O operations and ensures that all pending logs are written immediately.
 */
    suspend fun flush()

    /**
 * Releases all resources and shuts down the logger.
 *
 * Call this method to properly terminate logging operations and free any associated resources.
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
