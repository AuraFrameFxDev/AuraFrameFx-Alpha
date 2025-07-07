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
 * Logs an informational message indicating normal application operations.
 *
 * @param tag Identifier for the log source or component.
 * @param message The message describing the informational event.
 * @param throwable Optional exception or error associated with the event.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a warning message to indicate a potential issue or abnormal condition.
 *
 * @param tag Identifies the source or category of the warning.
 * @param message The warning message content.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an error-level message to indicate a failure or exception event.
 *
 * @param tag The category or source of the log entry.
 * @param message The error message to record.
 * @param throwable An optional exception related to the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a security-related event that may indicate a critical issue or breach.
 *
 * @param tag Identifies the source or category of the security event.
 * @param message Describes the security event being logged.
 * @param throwable An optional exception or error related to the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a performance event for an operation, recording its duration and optional contextual metadata.
 *
 * @param operation The name or description of the operation being measured.
 * @param durationMs The duration of the operation in milliseconds.
 * @param metadata Additional contextual data for the performance event.
 */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Records a user interaction event for analytics or behavioral tracking.
 *
 * @param tag Identifies the category or component associated with the interaction.
 * @param action Describes the specific user action performed.
 * @param metadata Optional contextual data providing additional details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs an AI operation event with its name, confidence score, and optional contextual metadata.
 *
 * @param tag Identifies the category or component related to the AI operation.
 * @param operation Describes the AI operation performed.
 * @param confidence The confidence score associated with the AI result.
 * @param metadata Additional context or details about the operation.
 */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Toggles logging on or off at runtime.
 *
 * @param enabled True to enable logging; false to disable it.
 */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
 * Sets the minimum severity level for log entries to be recorded.
 *
 * Only log messages at or above the specified log level will be processed and stored.
 *
 * @param level The minimum log level threshold.
 */
    fun setLogLevel(level: LogLevel)
    
    /**
 * Forces all buffered log entries to be written to persistent storage.
 *
 * This suspend function may perform I/O operations and should be called to ensure all logs are saved, especially before shutdown or critical transitions.
 */
    suspend fun flush()
    
    /**
 * Releases all resources and finalizes the logging system, terminating any ongoing logging operations.
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
