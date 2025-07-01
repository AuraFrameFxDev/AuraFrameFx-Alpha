package dev.aurakai.auraframefx.serialization

import kotlinx.datetime.Instant
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

object InstantSerializer : KSerializer<Instant> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("Instant", PrimitiveKind.STRING)

    /**
     * Serializes an [Instant] object by encoding its string representation.
     *
     * @param encoder The encoder used to write the serialized data.
     * @param value The [Instant] instance to serialize.
     */
    override fun serialize(encoder: Encoder, value: Instant) {
        encoder.encodeString(value.toString())
    }

    /**
     * Decodes a string from the given decoder and parses it into an `Instant` object.
     *
     * @return The deserialized `Instant` instance.
     */
    override fun deserialize(decoder: Decoder): Instant {
        return Instant.parse(decoder.decodeString())
    }
}

// Potentially add Clock and Duration serializers here if needed, though they weren't explicitly in the error
// For now, only Instant is addressed as per the error message.
