package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {
    
    /**
 * Logs a debug-level message for development and troubleshooting purposes.
 *
 * @param tag Identifier for the log source or component.
 * @param message The debug message to log.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an informational message related to normal application operation.
 *
 * @param tag A category or source identifier for the log entry.
 * @param message The informational message to log.
 * @param throwable An optional exception or error to include with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a warning message indicating a potential issue.
 *
 * @param tag The category or source of the log entry.
 * @param message The warning message to log.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an error message indicating a failure or exception.
 *
 * @param tag The category or source of the log entry.
 * @param message The error message to log.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a critical security event that requires immediate attention.
 *
 * @param tag The category or source of the security event.
 * @param message The security-related message to log.
 * @param throwable An optional exception or error associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs performance metrics for a specific operation, including its duration and optional metadata.
 *
 * @param tag Identifier for the log entry source or category.
 * @param operation Name or description of the operation being measured.
 * @param durationMs Duration of the operation in milliseconds.
 * @param metadata Optional additional data related to the operation.
 */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs a user interaction event for analytics and user experience optimization.
 *
 * @param tag The category or source of the interaction.
 * @param action The specific user action being recorded.
 * @param metadata Optional additional data related to the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs an AI-related operation, including the operation name, confidence score, and optional metadata.
 *
 * @param tag Identifier for the log source or component.
 * @param operation Description of the AI operation or decision.
 * @param confidence Confidence score associated with the AI operation.
 * @param metadata Optional additional information relevant to the AI operation.
 */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Enables or disables logging at runtime.
 *
 * @param enabled If true, logging is enabled; if false, logging is disabled.
 */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
 * Sets the minimum log level for logging.
 *
 * Only log entries at or above the specified level will be recorded.
 *
 * @param level The lowest log level that will be processed.
 */
    fun setLogLevel(level: LogLevel)
    
    /**
 * Forces all pending log entries to be written to storage.
 *
 * This suspend function ensures that any buffered or queued log data is persisted immediately.
 */
    suspend fun flush()
    
    /**
 * Releases resources and terminates logging operations.
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
