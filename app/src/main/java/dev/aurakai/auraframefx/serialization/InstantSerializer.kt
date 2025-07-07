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
     * Serializes an [Instant] value as its ISO-8601 string representation using the provided encoder.
     *
     * @param encoder The encoder to write the string representation to.
     * @param value The [Instant] value to serialize.
     */
    override fun serialize(encoder: Encoder, value: Instant) {
        encoder.encodeString(value.toString())
    }
    
    /**
     * Decodes a string from the given decoder and parses it into an `Instant` object.
     *
     * @return The deserialized `Instant` value.
     */
    override fun deserialize(decoder: Decoder): Instant {
        return Instant.parse(decoder.decodeString())
    }
}
