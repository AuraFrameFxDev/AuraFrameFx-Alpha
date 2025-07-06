package dev.aurakai.auraframefx.serialization

import kotlinx.datetime.Instant
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * Kotlinx.serialization serializer for kotlinx.datetime.Instant
 */
object InstantSerializer : KSerializer<Instant> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("Instant", PrimitiveKind.STRING)
    
    /**
     * Serializes an [Instant] as an ISO-8601 string using the provided encoder.
     *
     * Converts the [Instant] to its string representation before encoding.
     */
    override fun serialize(encoder: Encoder, value: Instant) {
        encoder.encodeString(value.toString())
    }
    
    /**
     * Deserializes an ISO-8601 formatted string from the decoder into an [Instant] object.
     *
     * @return The resulting [Instant] parsed from the input string.
     */
    override fun deserialize(decoder: Decoder): Instant {
        return Instant.parse(decoder.decodeString())
    }
}
