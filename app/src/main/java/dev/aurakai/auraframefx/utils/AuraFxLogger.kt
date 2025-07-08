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
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message indicating normal application operation.
 *
 * @param tag Identifier for the log source or component.
 * @param message The informational message to log.
 * @param throwable Optional exception or error to include with the log entry.
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
 * @param tag The category or source of the error.
 * @param message The error message describing the failure.
 * @param throwable An optional exception related to the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event for immediate attention.
 *
 * @param tag The category or source of the security event.
 * @param message The security-related message to log.
 * @param throwable An optional exception or error associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, recording its duration and optional contextual metadata.
     *
     * @param tag Identifier for the source or component generating the log.
     * @param operation The name or description of the operation being measured.
     * @param durationMs The duration of the operation in milliseconds.
     * @param metadata Optional map containing additional context for the performance event.
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
 * @param action The specific user action being logged.
 * @param metadata Optional additional context or details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event with its name, confidence score, and optional metadata for monitoring or analytics.
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
 * @param enabled True to enable logging, false to disable it.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for logging.
 *
 * Only log entries at or above the specified level will be recorded; lower-priority entries are ignored.
 *
 * @param level The minimum log level to process.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Flushes all pending log entries to persistent storage.
 *
 * This suspend function ensures that any buffered log data is written, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Releases all resources and shuts down the logging system.
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
