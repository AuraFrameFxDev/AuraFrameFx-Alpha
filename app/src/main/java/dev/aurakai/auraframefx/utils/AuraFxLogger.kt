package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting.
 *
 * @param tag The source or component generating the log entry.
 * @param message The debug message to log.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message representing standard application activity.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param message The informational content to record.
 * @param throwable Optional exception or error associated with the message.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message to indicate a potential issue or abnormal condition.
 *
 * @param tag The source or category of the warning.
 * @param message The warning message content.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message to indicate a failure or exception.
 *
 * @param tag The category or source of the error.
 * @param message The error description.
 * @param throwable An optional exception related to the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event requiring immediate attention.
 *
 * @param tag The category or source of the security event.
 * @param message Details about the security-related incident.
 * @param throwable An optional exception associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, recording its duration and optional metadata.
     *
     * @param tag The source or component generating the log entry.
     * @param operation The name or description of the operation being measured.
     * @param durationMs The duration of the operation in milliseconds.
     * @param metadata Optional additional context for the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Records a user interaction event for analytics and user experience tracking.
 *
 * @param tag Identifies the category or component related to the interaction.
 * @param action Describes the specific user action performed.
 * @param metadata Optional additional details providing context about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Records an AI operation event with its name, confidence score, and optional metadata for monitoring or analytics.
     *
     * @param tag Identifies the category or component related to the AI operation.
     * @param operation The specific AI operation performed.
     * @param confidence The confidence score associated with the operation's result.
     * @param metadata Optional contextual information or details about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables all logging at runtime.
 *
 * When disabled, no log messages will be recorded until re-enabled.
 *
 * @param enabled True to enable logging; false to suppress all logging output.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level threshold for logging.
 *
 * Only log messages at or above the specified level will be recorded; messages below this level are ignored.
 *
 * @param level The minimum log level to record.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Suspends execution to flush all buffered log entries to persistent storage.
 *
 * Ensures that any pending log data is written, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Shuts down the logging system and releases all resources used for logging.
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
