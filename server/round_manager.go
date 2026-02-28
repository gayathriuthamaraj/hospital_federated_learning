package main

import (
	"log"
	"sync"
)

// RoundState represents the current phase of a federated learning round.
type RoundState int

const (
	// RoundWaiting means the round is open and collecting updates.
	RoundWaiting RoundState = iota
	// RoundAggregating means quorum was met and aggregation is in progress.
	RoundAggregating
	// RoundComplete means aggregation finished and the next round has begun.
	RoundComplete
)

func (s RoundState) String() string {
	switch s {
	case RoundWaiting:
		return "WAITING"
	case RoundAggregating:
		return "AGGREGATING"
	case RoundComplete:
		return "COMPLETE"
	default:
		return "UNKNOWN"
	}
}

// RoundManager tracks the state of the current federated learning round.
// It is the single source of truth for whether aggregation should fire.
//
// Fields:
//   - CurrentRound    — monotonically incrementing round counter (starts at 0)
//   - ExpectedClients — minimum number of updates required to trigger aggregation (quorum)
//   - ReceivedClients — set of hospital IDs that have submitted in the current round
//   - State           — current phase of the round
type RoundManager struct {
	mu              sync.Mutex
	CurrentRound    int
	ExpectedClients int
	ReceivedClients map[string]bool // keyed by hospital_id to avoid duplicate counting
	State           RoundState
}

// NewRoundManager creates a RoundManager for round 0 with the given quorum size.
func NewRoundManager(quorum int) *RoundManager {
	return &RoundManager{
		CurrentRound:    0,
		ExpectedClients: quorum,
		ReceivedClients: make(map[string]bool),
		State:           RoundWaiting,
	}
}

// RecordUpdate registers an incoming update from hospitalID for the given roundID.
//
// Returns:
//   - accepted  bool   — false if the update is rejected (wrong round or duplicate)
//   - quorumMet bool   — true if this submission caused quorum to be reached
//
// Caller must call TriggerAggregation() in a goroutine when quorumMet is true.
func (rm *RoundManager) RecordUpdate(hospitalID string, roundID int) (accepted bool, quorumMet bool) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	// Reject updates that belong to a different round.
	if roundID != rm.CurrentRound {
		log.Printf("[RoundManager] Rejected update from %s: round mismatch (got %d, current %d)",
			hospitalID, roundID, rm.CurrentRound)
		return false, false
	}

	// Reject if aggregation already triggered for this round.
	if rm.State != RoundWaiting {
		log.Printf("[RoundManager] Rejected update from %s: round %d is in state %s",
			hospitalID, rm.CurrentRound, rm.State)
		return false, false
	}

	// Reject duplicate submissions from the same hospital within a round.
	if rm.ReceivedClients[hospitalID] {
		log.Printf("[RoundManager] Rejected duplicate from %s in round %d", hospitalID, rm.CurrentRound)
		return false, false
	}

	rm.ReceivedClients[hospitalID] = true
	received := len(rm.ReceivedClients)

	log.Printf("[RoundManager] Round %d — %s submitted (%d/%d)",
		rm.CurrentRound, hospitalID, received, rm.ExpectedClients)

	if received >= rm.ExpectedClients {
		rm.State = RoundAggregating
		log.Printf("[RoundManager] Quorum met (%d/%d). Triggering aggregation for round %d.",
			received, rm.ExpectedClients, rm.CurrentRound)
		return true, true
	}

	return true, false
}

// AdvanceRound moves the RoundManager into the next round.
// Must be called by the aggregation routine after a successful aggregation.
func (rm *RoundManager) AdvanceRound() {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.CurrentRound++
	rm.ReceivedClients = make(map[string]bool)
	rm.State = RoundWaiting

	log.Printf("[RoundManager] Advanced to round %d. Waiting for %d clients.",
		rm.CurrentRound, rm.ExpectedClients)
}

// Status returns a snapshot of the current round state (safe to call at any time).
func (rm *RoundManager) Status() (round, expected, received int, state RoundState) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	return rm.CurrentRound, rm.ExpectedClients, len(rm.ReceivedClients), rm.State
}
