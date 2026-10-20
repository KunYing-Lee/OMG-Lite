# Tracking

HoloMotion consumes native50Hz reference poses. `Tracker` verifies a50Hz
control period and ten future preview frames. `PlanTimeline` replaces only the
future suffix at fixed frame50 boundaries. Reference derivatives are computed
on the concatenated raw timeline, without modifying the poses.

The velocity stencil reads one additional sample beyond the ten-frame preview.
Thus a request at25 must finish before39. Same-rate input validation replaces
the old generic resampling function. Robot sensor sampling and root-state
estimation remain the hardware controller's responsibility.

See [deployment](realtime_g1.md) for preview packets and the consumer hook.
