# Changelog

## Unreleased
### Added
- text deduplication to strip consecutive duplicate phrases
- durable trusted-sample speaker review tracking, including migration of legacy review-added samples and review-screen visibility after reload
### Fixed
- add pydantic-settings dependency to unbreak CI
- regenerate requirements with hashes for deterministic installs
- speaker decisions now keep confirm match, keep unknown, local label only, and add trusted sample semantically separate
