# Review, 2026-09-24

The 297-test audit reproduced one timing-dependent assertion: with a 10 ms TTL,
the runtime correctly returned an immediate timeout fallback before the first
expected wait. The test incorrectly assumed thread dispatch always took less
than 10 ms. Replace its wall-clock sleep with a controlled module clock and an
unresolved future; assert hold before expiry and fallback after it. The separate
real-arrival test continues to verify asynchronous completion and applied effects.
This changes no package runtime, model, checkpoint or published evaluation source.
Reviewed against the existing planner TTL contract before editing the test.
