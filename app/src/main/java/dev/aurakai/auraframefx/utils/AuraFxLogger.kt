package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting.
 *
 * @param tag Identifies the source or component generating the log entry.
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
 * Logs a warning message indicating a potential issue or abnormal condition.
 *
 * @param tag The source or category of the warning.
 * @param message The warning message to record.
 * @param throwable An optional exception associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message representing a failure or exception, optionally including an associated exception.
 *
 * @param tag The category or source of the error.
 * @param message The error description.
 * @param throwable An optional exception related to the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a security-related event that may indicate a critical issue or breach.
 *
 * @param tag The category or source of the security event.
 * @param message Details about the security incident.
 * @param throwable An optional exception associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, recording its duration and optional contextual metadata.
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
 * Logs a user interaction event for analytics and user experience monitoring.
 *
 * @param tag The category or component associated with the interaction.
 * @param action The user action being recorded.
 * @param metadata Optional additional context or details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event with the given operation name, confidence score, and optional metadata.
     *
     * @param tag The category or component related to the AI operation.
     * @param operation The name or description of the AI operation performed.
     * @param confidence The confidence score of the AI operation result.
     * @param metadata Optional additional context or details about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables the logging system during application runtime.
 *
 * @param enabled True to enable logging; false to suppress all log output.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level threshold for logging.
 *
 * Only log messages at or above the specified level will be processed; messages below this level are ignored.
 *
 * @param level The minimum log level to allow.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Suspends execution to flush all buffered log entries to persistent storage.
 *
 * Ensures that any pending log data is written, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Shuts down the logging system and releases all associated resources.
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
