package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log"
	"time"
)

// SecretKey is the shared key used for packet signature verification.
// Must match the key used by the hospital/client when signing packets.
const SecretKey = "federated_secret_2024"

// MaxTimestampAge is the maximum age (in seconds) a packet's timestamp
// may have before the server rejects it as stale.
const MaxTimestampAge int64 = 30

// verifySignature recomputes the SHA256 hash over the packet's metadata
// and compares it against the signature carried in the packet.
// Returns true if the signature is valid.
func verifySignature(packet UpdatePacket) bool {
	metaJSON, err := json.Marshal(packet.Metadata)
	if err != nil {
		log.Printf("[security] Failed to marshal metadata for verification: %v", err)
		return false
	}

	hash := sha256.Sum256(append(metaJSON, []byte(SecretKey)...))
	expected := hex.EncodeToString(hash[:])

	if expected != packet.Signature {
		log.Printf("[security] Signature mismatch for %s: expected %s, got %s",
			packet.Metadata.HospitalID, expected, packet.Signature)
		return false
	}

	log.Printf("[security] Signature verified for %s", packet.Metadata.HospitalID)
	return true
}

// validateTimestamp checks that the packet's timestamp is within the
// acceptable freshness window (MaxTimestampAge seconds from now).
// Returns true if the timestamp is valid (not stale).
func validateTimestamp(packet UpdatePacket) bool {
	now := time.Now().Unix()
	age := now - packet.Metadata.Timestamp

	if age > MaxTimestampAge {
		log.Printf("[security] Stale packet from %s: timestamp age %ds exceeds %ds limit",
			packet.Metadata.HospitalID, age, MaxTimestampAge)
		return false
	}

	if age < 0 {
		log.Printf("[security] Future timestamp from %s: %ds ahead — rejecting",
			packet.Metadata.HospitalID, -age)
		return false
	}

	log.Printf("[security] Timestamp valid for %s (age: %ds)", packet.Metadata.HospitalID, age)
	return true
}
